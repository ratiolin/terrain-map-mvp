"""Layer 1 Supplementary Analyses.

1. Control Signal Decomposition — unit-action effect, net effect vs d.
2. Action Effective Dimension — PCA spectrum, effective rank, k90.
3. Environment Sensitivity — ∂s/∂a Jacobian, Frobenius norm, condition number.
4. Direction Alignment — cos(a, ∇_a risk_pred), projection efficiency.

All analyses reuse the ClosedLoopModel architecture and MultiModeEnv.
Outputs go to core_mvp_v4/results/layer1_supplement/.

Usage:
  uv run python core_mvp_v4/experiments/layer1_supplement.py
  uv run python core_mvp_v4/experiments/layer1_supplement.py --quick
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core_mvp_v4.core_env import MultiModeEnv, PseudoActionGenerator
from core_mvp_v4.core_models import ClosedLoopModel, get_designed_hidden_dim
from core_mvp_v4.core_metrics import compute_w2_gaussian

# ---------------------------------------------------------------------------
K = 2
D_DIMS = [2, 5, 10, 20, 50, 100]
N_SEEDS = 5
N_EPISODES = 5
EP_LENGTH = 2000
ROLLOUT_STEPS = 2000
DEFAULT_LR = 1e-3
LAMBDA_CTRL = 0.1
DRIFT = 0.5
QUICK_DIMS = [2, 8, 16, 32]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results", "layer1_supplement")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(d, seed, mode="closed"):
    env = MultiModeEnv(d_total=d, k_controlled=K, mode=mode,
                       theta=0.5, noise_std=0.05, coupling=0.05,
                       drift=DRIFT, force_scale=0.1, action_scale=0.1,
                       seed=seed, pseudo_affects_noise=False)
    env.calibrate_noise_scales()
    return env


def _make_model(d, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return ClosedLoopModel(state_dim=d, hidden_dim=get_designed_hidden_dim(d), action_dim=K)


def train_model(model, env, episodes, steps, lr, lam, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    env.set_seed(seed)
    action_norms = []
    for ep in range(episodes):
        state = env.reset()
        for _ in range(steps):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            action_norms.append(float(np.linalg.norm(a_np)))
            noise = np.random.randn(*a_np.shape) * 0.03
            ns, risk, _, _ = env.step(a_np + noise)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            ploss = nn.functional.mse_loss(risk_pred, risk_t)
            aloss = torch.mean(action ** 2)
            loss = ploss + lam * aloss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state = ns
    return np.array(action_norms)


def rollout_closed(env, model, steps, seed):
    env.set_seed(seed)
    state = env.reset()
    states, actions, next_states = [], [], []
    for _ in range(steps):
        a_np = model.act_numpy(state)
        ns, _, _, _ = env.step(a_np)
        states.append(state.copy())
        actions.append(a_np.copy())
        next_states.append(ns.copy())
        state = ns
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
    }


def rollout_open(env, steps, seed):
    env.set_seed(seed)
    state = env.reset()
    states, next_states = [], []
    for _ in range(steps):
        a_zero = np.zeros(K, dtype=np.float32)
        ns, _, _, _ = env.step(a_zero)
        states.append(state.copy())
        next_states.append(ns.copy())
        state = ns
    return {
        "states": np.array(states, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Analysis 1: Control Signal Decomposition
# ---------------------------------------------------------------------------

def analyze_control_efficiency(
    traj_cl: Dict, model, env_cl: MultiModeEnv, env_open: MultiModeEnv,
    seed: int, steps: int,
) -> Dict:
    acts = traj_cl["actions"]       # (T, K)
    ns_cl = traj_cl["next_states"]  # (T, d)
    st_cl = traj_cl["states"]       # (T, d)

    eps = 1e-6
    T = len(acts)

    # per-step unit-action effect
    s_ctrl_now = st_cl[:, :K]
    s_ctrl_next = ns_cl[:, :K]
    delta_s = np.linalg.norm(s_ctrl_next - s_ctrl_now, axis=1)
    a_norm = np.linalg.norm(acts, axis=1)
    eff_raw = delta_s / (a_norm + eps)

    # ---- Net action effect via open-loop baseline ----
    open_env = deepcopy(env_open)
    traj_op = rollout_open(open_env, steps=steps, seed=seed)
    s_ctrl_next_op = traj_op["next_states"][:, :K]

    # Direct difference: ns_cl - ns_op isolates action contribution
    action_effect = ns_cl[:, :K] - s_ctrl_next_op
    net_effect = np.linalg.norm(action_effect, axis=1)
    eff_net_raw = net_effect / (a_norm + eps)

    # Filter degenerate steps where a_norm is tiny
    valid_mask = a_norm > 1e-4
    eff_net_filt = eff_net_raw[valid_mask] if valid_mask.sum() > 10 else eff_net_raw
    net_effect_filt = net_effect[valid_mask] if valid_mask.sum() > 10 else net_effect

    delta_s_op = np.linalg.norm(s_ctrl_next_op - traj_op["states"][:, :K], axis=1)

    return {
        "eff_raw_mean": float(np.mean(eff_raw)),
        "eff_raw_std": float(np.std(eff_raw)),
        "eff_net_mean": float(np.mean(eff_net_filt)),
        "eff_net_std": float(np.std(eff_net_filt)),
        "delta_s_cl_mean": float(np.mean(delta_s)),
        "delta_s_op_mean": float(np.mean(delta_s_op)),
        "net_effect_mean": float(np.mean(net_effect_filt)),
        "action_norm_mean": float(np.mean(a_norm)),
    }


# ---------------------------------------------------------------------------
# Analysis 2: Action Effective Dimension (PCA)
# ---------------------------------------------------------------------------

def analyze_action_pca(actions: np.ndarray, ns_ctrl: np.ndarray) -> Dict:
    """PCA on action vectors. Also align top PC with state-change direction."""
    a = actions.astype(np.float64)
    if a.shape[0] < 3:
        return {"eff_rank": 1.0, "k90": a.shape[1], "cos_sim_top_pc_ds": 0.0}

    a_centered = a - a.mean(axis=0)
    cov = a_centered.T @ a_centered / (a.shape[0] - 1)
    try:
        _, S, Vt = np.linalg.svd(cov, full_matrices=False)
    except np.linalg.LinAlgError:
        S = np.ones(a.shape[1])
        Vt = np.eye(a.shape[1])

    S = np.maximum(S, 0.0)
    p = S / (S.sum() + 1e-12)
    p = p[p > 0]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))) if len(p) > 0 else 1.0)

    cumsum = np.cumsum(S) / (S.sum() + 1e-12)
    k90 = int(np.searchsorted(cumsum, 0.90) + 1)
    k90 = min(k90, len(S))

    # ---- Alignment with Δs_ctrl ----
    if len(ns_ctrl) >= 2:
        ds = np.diff(ns_ctrl.astype(np.float64), axis=0)
        ds_mean = ds.mean(axis=0)
        ds_norm = np.linalg.norm(ds_mean)
        if ds_norm > 1e-12:
            ds_dir = ds_mean / ds_norm
        else:
            ds_dir = np.zeros_like(ds_mean)
    else:
        ds_dir = np.zeros(ns_ctrl.shape[1])

    top_pc = Vt[0, :]  # dominant PC direction
    top_norm = np.linalg.norm(top_pc)
    if top_norm > 1e-12 and np.linalg.norm(ds_dir) > 1e-12:
        cos_sim = float(np.abs(np.dot(top_pc / top_norm, ds_dir)))
    else:
        cos_sim = 0.0

    return {
        "eff_rank": eff_rank,
        "k90": k90,
        "singular_values": S.tolist(),
        "variance_explained": (S / (S.sum() + 1e-12)).tolist(),
        "cos_sim_top_pc_ds": cos_sim,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Environment Sensitivity (∂s/∂a Jacobian)
# ---------------------------------------------------------------------------

def analyze_env_sensitivity(
    env: MultiModeEnv, model, states: np.ndarray, n_sample: int = 200,
) -> Dict:
    """Compute ||∂s_ctrl/∂a||_F and condition number via finite differences."""
    eps_fd = 1e-3
    n = min(n_sample, len(states))
    idx = np.linspace(0, len(states) - 1, n, dtype=int)
    sampled = states[idx]

    frob_vals = []
    cond_vals = []
    for s in sampled:
        a_ref = model.act_numpy(s)
        J = np.zeros((K, K), dtype=np.float64)

        saved = env.save_state()
        env.state = s.copy()
        env._rng.set_state(saved["rng_state"])

        for j in range(K):
            e = np.zeros(K, dtype=np.float32)
            e[j] = eps_fd

            env.restore_state(saved)
            ns_plus, _, _, _ = env.step(a_ref + e)
            ns_plus_ctrl = ns_plus[:K].astype(np.float64)

            env.restore_state(saved)
            ns_minus, _, _, _ = env.step(a_ref - e)
            ns_minus_ctrl = ns_minus[:K].astype(np.float64)

            J[:, j] = (ns_plus_ctrl - ns_minus_ctrl) / (2.0 * eps_fd)

        env.restore_state(saved)

        frob = np.linalg.norm(J, ord='fro')
        frob_vals.append(frob)
        try:
            sv = np.linalg.svd(J, compute_uv=False)
            if sv[-1] > 1e-12:
                cond_vals.append(sv[0] / sv[-1])
            else:
                cond_vals.append(np.inf)
        except np.linalg.LinAlgError:
            cond_vals.append(np.inf)

    finite_cond = [c for c in cond_vals if np.isfinite(c)]
    return {
        "jac_frob_mean": float(np.mean(frob_vals)),
        "jac_frob_std": float(np.std(frob_vals)),
        "jac_cond_mean": float(np.mean(finite_cond)) if finite_cond else float('inf'),
        "jac_cond_median": float(np.median(finite_cond)) if finite_cond else float('inf'),
        "n_samples": n,
        "n_finite_cond": len(finite_cond),
    }


# ---------------------------------------------------------------------------
# Analysis 4: Direction Alignment (action vs risk gradient)
# ---------------------------------------------------------------------------

def analyze_action_gradient_alignment(
    model: ClosedLoopModel, states: np.ndarray, n_sample: int = 500,
) -> Dict:
    """cos(actor_out, ∇_a risk_pred) and projection efficiency."""
    n = min(n_sample, len(states))
    idx = np.linspace(0, len(states) - 1, n, dtype=int)
    sampled = states[idx]

    cos_sims = []
    proj_effs = []

    for s in sampled:
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        s_t.requires_grad_(False)

        h = model.encoder(s_t)
        action = model.actor(h)
        action_detached = action.detach().requires_grad_(True)

        risk_pred = model.predictor(torch.cat([h, action_detached], dim=-1))

        model.zero_grad()
        risk_pred.sum().backward(retain_graph=True)
        grad = action_detached.grad

        if grad is None:
            model.zero_grad()
            continue

        a_np = action.detach().squeeze(0).numpy().astype(np.float64)
        g_np = grad.squeeze(0).numpy().astype(np.float64)

        a_norm = np.linalg.norm(a_np)
        g_norm = np.linalg.norm(g_np)

        if a_norm > 1e-12 and g_norm > 1e-12:
            cos_sim = float(np.dot(a_np, g_np) / (a_norm * g_norm))
            proj = float(np.dot(a_np, g_np) / g_norm)  # scalar projection of a onto g
            proj_eff = float(abs(proj) / a_norm) if a_norm > 1e-12 else 0.0
        else:
            cos_sim = 0.0
            proj_eff = 0.0

        cos_sims.append(cos_sim)
        proj_effs.append(proj_eff)

        model.zero_grad()

    cos_sims = np.array(cos_sims)
    proj_effs = np.array(proj_effs)

    return {
        "cos_sim_mean": float(np.mean(cos_sims)),
        "cos_sim_std": float(np.std(cos_sims)),
        "cos_sim_median": float(np.median(cos_sims)),
        "proj_efficiency_mean": float(np.mean(proj_effs)),
        "proj_efficiency_std": float(np.std(proj_effs)),
        "frac_cos_positive": float(np.mean(cos_sims > 0)),
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_supplements(
    d_dims=None, seeds=None,
    episodes=N_EPISODES, steps=EP_LENGTH,
    rollout_steps=ROLLOUT_STEPS,
    results_dir=None,
):
    if d_dims is None:
        d_dims = D_DIMS
    if seeds is None:
        seeds = list(range(N_SEEDS))
    if results_dir is None:
        results_dir = OUT_DIR
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Layer 1 Supplementary Analyses")
    print(f"  d ∈ {d_dims}  |  seeds = {seeds}")
    print(f"  training: {episodes}ep × {steps}steps  |  rollout: {rollout_steps}")
    print(f"{'=' * 70}\n")

    all_results: Dict[str, Dict] = {}

    for d in d_dims:
        print(f"\n--- d = {d} ---")
        t0 = time.time()

        ce_seeds, ae_seeds, es_seeds, ag_seeds = [], [], [], []

        for seed in seeds:
            print(f"  seed={seed}  ", end="", flush=True)

            # Train
            train_env = _make_env(d, seed, mode="closed")
            model = _make_model(d, seed)
            anorms = train_model(model, train_env, episodes, steps, DEFAULT_LR, LAMBDA_CTRL, seed)
            model.eval()

            final_an = float(np.mean(anorms[-500:])) if len(anorms) >= 500 else float(np.mean(anorms))
            if final_an < 1e-3:
                print(f"SKIP (|a|={final_an:.5f} < 1e-3)")
                continue

            # Rollout
            base_env = _make_env(d, seed + 10000, mode="closed")
            cl_env = deepcopy(base_env)
            cl_env.set_mode("closed")
            traj_cl = rollout_closed(cl_env, model, steps=rollout_steps, seed=seed + 20000)

            base_open = _make_env(d, seed + 10000, mode="open")
            base_open.calibrate_noise_scales()

            # -------- Analysis 1: Control Efficiency --------
            ce = analyze_control_efficiency(
                traj_cl, model, cl_env, base_open, seed=seed + 20000, steps=rollout_steps,
            )
            ce["seed"] = seed
            ce_seeds.append(ce)

            # -------- Analysis 2: Action PCA --------
            ae = analyze_action_pca(
                traj_cl["actions"],
                traj_cl["next_states"][:, :K],
            )
            ae["seed"] = seed
            ae_seeds.append(ae)

            # -------- Analysis 3: Env Sensitivity --------
            es = analyze_env_sensitivity(
                deepcopy(cl_env), model, traj_cl["states"], n_sample=200,
            )
            es["seed"] = seed
            es_seeds.append(es)

            # -------- Analysis 4: Action-Gradient Alignment --------
            ag = analyze_action_gradient_alignment(
                model, traj_cl["states"], n_sample=500,
            )
            ag["seed"] = seed
            ag_seeds.append(ag)

            print(f"eff_net={ce['eff_net_mean']:.4f} "
                  f"eff_rank={ae['eff_rank']:.2f} "
                  f"jac_f={es['jac_frob_mean']:.4f} "
                  f"cos={ag['cos_sim_mean']:.3f}")

        # Aggregate each analysis
        agg_ce = _aggregate_list(ce_seeds, ["eff_raw_mean", "eff_net_mean", "net_effect_mean", "action_norm_mean"])
        agg_ae = _aggregate_list(ae_seeds, ["eff_rank", "k90", "cos_sim_top_pc_ds"])
        agg_es = _aggregate_list(es_seeds, ["jac_frob_mean", "jac_cond_mean", "jac_cond_median"])
        agg_ag = _aggregate_list(ag_seeds, ["cos_sim_mean", "proj_efficiency_mean", "frac_cos_positive"])

        all_results[f"d{d}"] = {
            "control_efficiency": agg_ce,
            "action_effective_rank": agg_ae,
            "env_sensitivity": agg_es,
            "action_gradient_alignment": agg_ag,
        }

        elapsed = time.time() - t0
        print(f"  d={d} done in {elapsed:.1f}s  "
              f"eff_net={agg_ce.get('eff_net_mean',{}).get('mean','?'):.3f}  "
              f"eff_rank={agg_ae.get('eff_rank',{}).get('mean','?'):.2f}  "
              f"cos={agg_ag.get('cos_sim_mean',{}).get('mean','?'):.3f}")

    # ---- Save per-analysis JSON ----
    _save_analysis(results_dir, "control_efficiency", all_results, d_dims,
                   ["eff_raw_mean", "eff_net_mean", "net_effect_mean", "action_norm_mean"])
    _save_analysis(results_dir, "action_effective_rank", all_results, d_dims,
                   ["eff_rank", "k90", "cos_sim_top_pc_ds"])
    _save_analysis(results_dir, "env_sensitivity", all_results, d_dims,
                   ["jac_frob_mean", "jac_cond_mean", "jac_cond_median"])
    _save_analysis(results_dir, "action_gradient_alignment", all_results, d_dims,
                   ["cos_sim_mean", "proj_efficiency_mean", "frac_cos_positive"])

    # ---- Plotting ----
    try:
        _make_plots(all_results, d_dims, results_dir)
    except Exception as e:
        print(f"  [warn] Plotting failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"All supplement analyses complete. Results in {results_dir}/")
    print(f"{'=' * 70}")
    return all_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_list(entries, keys):
    result = {}
    for key in keys:
        vals = [e.get(key, 0.0) for e in entries if key in e]
        if vals:
            a = np.array(vals, dtype=np.float64)
            # Filter outliers: drop values with |z-score| > 5
            finite = a[np.isfinite(a)]
            if len(finite) >= 2:
                med = np.median(finite)
                mad = np.median(np.abs(finite - med))
                if mad > 1e-12:
                    z = np.abs(finite - med) / mad
                    finite = finite[z < 5.0]
                a = finite
            result[key] = {"mean": float(np.mean(a)), "std": float(np.std(a)), "n": int(len(a))}
    result["n_seeds"] = len(entries)
    return result


def _save_analysis(out_dir, name, all_results, d_dims, keys):
    """Save per-d summary and full spectrum for one analysis."""
    summary = {}
    for key in keys:
        summary[f"{key}_vs_d"] = {}
        for d in d_dims:
            entry = all_results.get(f"d{d}", {}).get(name, {})
            v = entry.get(key, {})
            summary[f"{key}_vs_d"][f"d{d}"] = {"mean": v.get("mean", None), "std": v.get("std", None)}

    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(all_results, d_dims, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def _get(key, sub):
        m, s = [], []
        for d in d_dims:
            e = all_results.get(f"d{d}", {}).get(key, {}).get(sub, {})
            m.append(e.get("mean", np.nan))
            s.append(e.get("std", np.nan))
        return np.array(m, dtype=np.float64), np.array(s, dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # eff_net(d)
    ax = axes[0, 0]
    m, s = _get("control_efficiency", "eff_net_mean")
    ax.errorbar(d_dims, m, yerr=s, marker="o", capsize=3, color="C0")
    ax.set_xlabel("d"); ax.set_ylabel("net eff"); ax.set_title("Net Unit-Action Effect")
    ax.grid(True, alpha=0.3)

    # eff_rank(d)
    ax = axes[0, 1]
    m, s = _get("action_effective_dim", "eff_rank")
    ax.errorbar(d_dims, m, yerr=s, marker="D", capsize=3, color="C1")
    ax.set_xlabel("d"); ax.set_ylabel("effective rank"); ax.set_title("Action Effective Rank")
    ax.grid(True, alpha=0.3)

    # jac_frob(d)
    ax = axes[1, 0]
    m, s = _get("env_sensitivity", "jac_frob_mean")
    ax.errorbar(d_dims, m, yerr=s, marker="s", capsize=3, color="C2")
    ax.set_xlabel("d"); ax.set_ylabel("||J_ctrl||_F"); ax.set_title("Env Sensitivity (Jacobian)")
    ax.grid(True, alpha=0.3)

    # cos_sim(d)
    ax = axes[1, 1]
    m, s = _get("action_gradient_alignment", "cos_sim_mean")
    ax.errorbar(d_dims, m, yerr=s, marker="^", capsize=3, color="C3")
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("d"); ax.set_ylabel("cos(a, ∇risk)"); ax.set_title("Action-Gradient Alignment")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Layer 1 Supplement: Control Analysis", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "layer1_supplement.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Layer 1 Supplementary Analyses")
    parser.add_argument("--d-dims", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--steps", type=int, default=EP_LENGTH)
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        d_dims = QUICK_DIMS
        seeds = list(range(3))
        episodes = 2
        steps = 500
        rollout_steps = 500
    else:
        d_dims = args.d_dims or D_DIMS
        seeds = list(range(args.seeds))
        episodes = args.episodes
        steps = args.steps
        rollout_steps = args.rollout_steps

    run_supplements(
        d_dims=d_dims, seeds=seeds,
        episodes=episodes, steps=steps,
        rollout_steps=rollout_steps,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
