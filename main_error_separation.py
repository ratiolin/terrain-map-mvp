"""
Error Separation Drift Scan — full pipeline.

  1. Run drift-rate scan with repetitions (≥ 3 seeds)
  2. Aggregate → D_sep_mean(d), loss_mean(d), eta_mean(d)
  3. Plot three final curves (drift-rate level)
  4. Normalised alignment plot (D_sep_norm & -loss_norm)
  5. Detect W_struct: dD/ddrift > 0 ∧ dloss/ddrift < 0
  6. Stability validation: repeat for multiple kappa
  7. Three diagnostic outputs (printed):
       a. D_sep monotonicity
       b. W_struct existence
       c. W_struct cross-kappa stability

Usage:
  uv run python main_error_separation.py
  uv run python main_error_separation.py --kappa "0.5,1.0,1.5"
"""
import copy
import json
import random
import time

import numpy as np
import torch
import torch.optim as optim

from env_drifting_double_well import DriftingDoubleWell
from agent import Agent
from controller import GatingGrowthController
from error_separation import (
    ErrorSeparationTracker,
    train_with_tracker,
    evaluate_with_tracker,
    aggregate_scan_results,
    find_W_struct,
    check_monotonic,
    plot_scan_curves,
    plot_alignment,
)


def reset_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _state_dim(env):
    dim = 2 if env.add_context else 1
    mk = getattr(env, "memory_k", None)
    if mk is not None and mk > 1:
        dim = mk
        if env.add_context:
            dim += 1
    elif hasattr(env, "memory_alpha") and env.memory_alpha is not None:
        dim = 2
        if env.add_context:
            dim += 1
    return dim


def make_agent(env):
    return Agent(obs_dim=_state_dim(env), act_dim=1, hidden_dim=2, lr=1e-3)


def make_ctrl(agent, K_budget=4, inertia=0.0):
    ctrl = GatingGrowthController(
        check_interval=200, env_type="doublewell",
        merge_thresh=0.2, prune_thresh=0.03, max_models=8,
        use_z=True, inertia=inertia,
    )
    ctrl.init_models(agent)
    for _ in range(K_budget - 1):
        child = copy.deepcopy(ctrl.models[0])
        child.optimizer = optim.Adam(child.parameters(), lr=1e-3)
        with torch.no_grad():
            for p in child.predictor.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        ctrl.models.append(child)
        ctrl.gating.expand()
        ctrl.gating_optimizer = optim.Adam(ctrl.gating.parameters(), lr=1e-3)
    K = len(ctrl.models)
    ctrl.usage = [0] * K
    ctrl.errors = [[] for _ in range(K)]
    ctrl.birth_step = [0] * K
    ctrl.region_bias = [0.0] * K
    ctrl.weight_history = [[] for _ in range(K)]
    return ctrl


# ── single-condition run ─────────────────────────────────────────────

def run_single(kappa, drift_rate, seed, train_steps=1200, test_steps=300,
               K_budget=4, inertia=0.0, flip_mode="deterministic",
               add_context=False, omega=None, alpha_ema=0.1, epsilon=0.01):
    """
    Run one (kappa, drift_rate, seed) experiment.
    Returns tracker + summary dict.
    """
    reset_seed(seed)

    env_train = DriftingDoubleWell(
        kappa=kappa, drift_rate=drift_rate,
        flip_mode=flip_mode, add_context=add_context, omega=omega,
    )
    agent = make_agent(env_train)
    ctrl = make_ctrl(agent, K_budget=K_budget, inertia=inertia)

    tracker = ErrorSeparationTracker(alpha_ema=alpha_ema, epsilon=epsilon)
    train_with_tracker(env_train, ctrl, agent, steps=train_steps,
                       tracker=tracker, z_loss=True, inertia=inertia)

    reset_seed(seed + 10000)
    env_test = DriftingDoubleWell(
        kappa=kappa, drift_rate=drift_rate,
        flip_mode=flip_mode, add_context=add_context, omega=omega,
    )
    evaluate_with_tracker(env_test, ctrl, steps=test_steps, tracker=tracker)

    return tracker, tracker.summary()


# ── per-kappa scan ───────────────────────────────────────────────────

def scan_one_kappa(kappa, drift_list, seeds, train_steps=1200, test_steps=300,
                   K_budget=4, inertia=0.0, flip_mode="deterministic",
                   add_context=False, alpha_ema=0.1, epsilon=0.01, verbose=True):
    """
    For a fixed kappa, run all drift_rates with all seeds.
    Returns:
      run_summaries   : {drift_rate: [summary, ...]}   (raw per-run data)
      drift_vals, D_sep, loss, eta, D_std, L_std, E_std   (aggregated)
      monotonic       : dict from check_monotonic
      W_struct        : list from find_W_struct
    """
    run_summaries = {}
    n_total = len(drift_list) * len(seeds)
    done = 0

    for drift_rate in drift_list:
        per_drift = []
        for seed in seeds:
            t0 = time.time()
            tracker, summary = run_single(
                kappa=kappa, drift_rate=drift_rate, seed=seed,
                train_steps=train_steps, test_steps=test_steps,
                K_budget=K_budget, inertia=inertia,
                flip_mode=flip_mode, add_context=add_context,
                alpha_ema=alpha_ema, epsilon=epsilon,
            )
            per_drift.append(summary)
            done += 1
            elapsed = time.time() - t0
            if verbose:
                print(f"  [{done}/{n_total}] k={kappa} d={drift_rate} s={seed}  "
                      f"D_sep={summary['D_sep_mean']:.4f}  "
                      f"loss={summary['loss_mean']:.3f}  "
                      f"({elapsed:.1f}s)")
        run_summaries[drift_rate] = per_drift

    drift_vals, D_sep, loss, eta, D_std, L_std, E_std = aggregate_scan_results(run_summaries)
    monotonic = check_monotonic(drift_vals, D_sep)
    W_struct = find_W_struct(drift_vals, D_sep, loss)

    return run_summaries, (drift_vals, D_sep, loss, eta, D_std, L_std, E_std), monotonic, W_struct


# ── stability validation across kappas ───────────────────────────────

def stability_validation(kappa_list, drift_list, seeds, output_prefix="error_sep_scan", **kwargs):
    """
    Run full scan for each kappa, then compare W_struct across kappas.
    Returns dict keyed by kappa with all results.
    """
    all_results = {}

    for kappa in kappa_list:
        print(f"\n{'#'*60}")
        print(f"  SCANNING kappa = {kappa}")
        print(f"{'#'*60}")

        summaries, curves, monotonic, W_struct = scan_one_kappa(
            kappa=kappa, drift_list=drift_list, seeds=seeds, **kwargs,
        )
        dv, D, L, E, Ds, Ls, Es = curves

        tag = f"k{kappa}"
        pfx = f"{output_prefix}_{tag}"

        plot_scan_curves(dv, D, Ds, L, Ls, E, Es,
                         save_path=f"{pfx}_curves.png")
        plot_alignment(dv, D, L, W_struct=W_struct,
                       save_path=f"{pfx}_alignment.png")

        all_results[kappa] = {
            "kappa": kappa,
            "drift_vals": dv.tolist(),
            "D_sep_mean": D.tolist(),
            "loss_mean": L.tolist(),
            "eta_mean": E.tolist(),
            "D_sep_std": Ds.tolist(),
            "loss_std": Ls.tolist(),
            "eta_std": Es.tolist(),
            "monotonic": monotonic,
            "W_struct": W_struct,
        }

        print(f"\n── kappa={kappa}  diagnostics ──")
        print(f"  monotonic:      {monotonic['is_monotonic']}")
        print(f"  direction:      {monotonic['direction']}")
        print(f"  n_segments:     {len(monotonic.get('segments', []))}")
        print(f"  W_struct found: {len(W_struct)}")
        for w in W_struct:
            print(f"    d ∈ [{w['drift_start']:.4f}, {w['drift_end']:.4f}]")

    return all_results


def print_diagnostics(all_results):
    """
    Print the three required diagnostic outputs.
    """
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC OUTPUT")
    print(f"{'='*70}")

    # ── 1. D_sep monotonicity ──
    print(f"\n 1) D_sep monotonicity")
    print(f"    {'─'*40}")
    for kappa, r in sorted(all_results.items()):
        m = r["monotonic"]
        seg_info = ", ".join(
            f"[{s['drift_start']:.3f},{s['drift_end']:.3f}] {s['direction']}"
            for s in m.get("segments", [])
        )
        print(f"    kappa={kappa}:  monotonic={m['is_monotonic']}  "
              f"direction={m['direction']}")
        if not m["is_monotonic"]:
            print(f"      piecewise segments: {seg_info}")

    # ── 2. W_struct existence ──
    print(f"\n 2) W_struct existence")
    print(f"    {'─'*40}")
    for kappa, r in sorted(all_results.items()):
        W = r["W_struct"]
        if W:
            ws = ", ".join(f"[{w['drift_start']:.4f}, {w['drift_end']:.4f}]" for w in W)
            print(f"    kappa={kappa}:  EXISTS  —  {ws}")
        else:
            print(f"    kappa={kappa}:  NOT FOUND")

    # ── 3. W_struct cross-kappa stability ──
    print(f"\n 3) W_struct cross-kappa stability")
    print(f"    {'─'*40}")

    kappas = sorted(all_results.keys())
    all_intervals = {}
    for kappa in kappas:
        all_intervals[kappa] = [(w["drift_start"], w["drift_end"])
                                for w in all_results[kappa]["W_struct"]]

    def overlap(a, b):
        a_start, a_end = a
        b_start, b_end = b
        return max(0, min(a_end, b_end) - max(a_start, b_start))

    if len(kappas) >= 2:
        for i in range(len(kappas)):
            for j in range(i + 1, len(kappas)):
                ki, kj = kappas[i], kappas[j]
                common = []
                for ai in all_intervals[ki]:
                    for aj in all_intervals[kj]:
                        ov = overlap(ai, aj)
                        if ov > 0:
                            common.append({
                                "k1": ki, "k2": kj,
                                "overlap_start": max(ai[0], aj[0]),
                                "overlap_end": min(ai[1], aj[1]),
                                "overlap_width": ov,
                            })
                if common:
                    cs = ", ".join(
                        f"[{c['overlap_start']:.4f}, {c['overlap_end']:.4f}]"
                        f" (w={c['overlap_width']:.4f})" for c in common
                    )
                    print(f"    kappa {ki} ↔ {kj}:  STABLE  —  overlapping: {cs}")
                else:
                    print(f"    kappa {ki} ↔ {kj}:  UNSTABLE  —  no overlap")

    stable_overall = True
    if len(kappas) >= 2:
        common_all = all_intervals[kappas[0]]
        for k in kappas[1:]:
            new_common = []
            for a in common_all:
                for b in all_intervals[k]:
                    ov = overlap(a, b)
                    if ov > 0:
                        new_common.append((max(a[0], b[0]), min(a[1], b[1])))
            common_all = new_common
        if common_all:
            cs = ", ".join(f"[{c[0]:.4f}, {c[1]:.4f}]" for c in common_all)
            print(f"    OVERALL stable W_struct: {cs}")
        else:
            stable_overall = False
            print(f"    OVERALL: no region stable across ALL kappas")

    return stable_overall


# ── main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Error Separation Drift Scan")
    parser.add_argument("--kappa", type=str, default="1.0",
                        help="Comma-separated kappa values (e.g. 0.5,1.0,1.5)")
    parser.add_argument("--drift-min", type=float, default=0.02)
    parser.add_argument("--drift-max", type=float, default=0.14)
    parser.add_argument("--drift-step", type=float, default=0.02)
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Repetitions per condition (≥ 3)")
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--test-steps", type=int, default=300)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--inertia", type=float, default=0.0)
    parser.add_argument("--alpha-ema", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--flip-mode", type=str, default="deterministic",
                        choices=["deterministic", "random"])
    parser.add_argument("--add-context", action="store_true")
    parser.add_argument("--output", type=str, default="error_sep_scan")
    parser.add_argument("--sweep", action="store_true",
                        help="Run drift scan (default behavior; kept for compatibility)")
    args = parser.parse_args()

    kappa_list = [float(x.strip()) for x in args.kappa.split(",")]
    drift_list = list(np.arange(args.drift_min, args.drift_max + 1e-12,
                                args.drift_step))
    seeds = list(range(args.n_seeds))

    assert args.n_seeds >= 3, "Need at least 3 seeds per condition"

    print(f"")
    print(f"{'='*60}")
    print(f"  ERROR-SEPARATION DRIFT SCAN")
    print(f"{'='*60}")
    print(f"  kappa list:   {kappa_list}")
    print(f"  drift list:   {[f'{d:.2f}' for d in drift_list]}")
    print(f"  n_seeds:      {args.n_seeds}")
    print(f"  train steps:  {args.train_steps}")
    print(f"  test steps:   {args.test_steps}")
    print(f"  n_conditions: {len(kappa_list) * len(drift_list) * args.n_seeds}")
    print(f"{'='*60}")

    all_results = stability_validation(
        kappa_list=kappa_list,
        drift_list=drift_list,
        seeds=seeds,
        output_prefix=args.output,
        train_steps=args.train_steps,
        test_steps=args.test_steps,
        K_budget=args.K,
        inertia=args.inertia,
        flip_mode=args.flip_mode,
        add_context=args.add_context,
        alpha_ema=args.alpha_ema,
        epsilon=args.epsilon,
    )

    with open(f"{args.output}_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results → {args.output}_results.json")

    print_diagnostics(all_results)
