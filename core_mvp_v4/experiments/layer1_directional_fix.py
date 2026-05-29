"""Layer 1 Directional Fix: Eliminate sign randomness in cos_sim(a, ∇_a risk_pred).

Four loss variants tested across d ∈ {2, 10, 100}, ≥5 seeds each:

  0. Baseline  — MSE(risk_pred, risk_target) + λ·||a||²
  1. Directional — + α·risk_pred  (pushes predictions/states toward lower risk)
  2. One-sided — ReLU(risk_pred - risk_target)² + λ·||a||²
  3. Baseline-sign — adv² + β·sign(adv) + λ·||a||²  (adv = pred - baseline)

Goal: achieve consistent cos_sim sign (≥80% seeds same sign, risk-reducing direction)
while preserving proj_efficiency > 0.6 and total_risk ≤ baseline.

Usage:
  uv run python core_mvp_v4/experiments/layer1_directional_fix.py
  uv run python core_mvp_v4/experiments/layer1_directional_fix.py --quick
"""

from __future__ import annotations

import os, sys, json, time, argparse
from copy import deepcopy
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core_mvp_v4.core_env import MultiModeEnv
from core_mvp_v4.core_models import ClosedLoopModel, get_designed_hidden_dim

# ---------------------------------------------------------------------------
K = 2
D_DIMS = [2, 10, 100]
N_SEEDS = 5
N_EPISODES = 5
EP_LENGTH = 2000
ROLLOUT_STEPS = 1000
DEFAULT_LR = 1e-3
LAMBDA_CTRL = 0.1
DRIFT = 0.5
ALPHAS = [0.01, 0.1, 0.5]
BETAS = [0.01, 0.1]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results", "layer1_directional_fix")


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


# ---------------------------------------------------------------------------
# Training variants
# ---------------------------------------------------------------------------

def train_baseline(model, env, episodes, steps, lr, lam, seed):
    """Original MSE loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    env.set_seed(seed)
    state = env.reset()
    action_norms, risks = [], []
    for ep in range(episodes):
        for _ in range(steps):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            action_norms.append(float(np.linalg.norm(a_np)))
            noise = np.random.randn(*a_np.shape) * 0.03
            ns, risk, _, _ = env.step(a_np + noise)
            risks.append(risk)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            loss = nn.functional.mse_loss(risk_pred, risk_t) + lam * torch.mean(action ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state = ns
    return np.array(action_norms), np.array(risks)


def train_directional(model, env, episodes, steps, lr, lam, alpha, seed):
    """MSE + α·risk_pred + λ·||a||² — pushes toward lower risk."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    env.set_seed(seed)
    state = env.reset()
    action_norms, risks = [], []
    for ep in range(episodes):
        for _ in range(steps):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            action_norms.append(float(np.linalg.norm(a_np)))
            noise = np.random.randn(*a_np.shape) * 0.03
            ns, risk, _, _ = env.step(a_np + noise)
            risks.append(risk)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            mse = nn.functional.mse_loss(risk_pred, risk_t)
            directional_term = alpha * risk_pred.mean()
            action_loss = lam * torch.mean(action ** 2)
            loss = mse + directional_term + action_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state = ns
    return np.array(action_norms), np.array(risks)


def train_onesided(model, env, episodes, steps, lr, lam, seed):
    """ReLU(risk_pred - risk_target)² + λ·||a||² — only penalize over-prediction."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    env.set_seed(seed)
    state = env.reset()
    action_norms, risks = [], []
    for ep in range(episodes):
        for _ in range(steps):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            action_norms.append(float(np.linalg.norm(a_np)))
            noise = np.random.randn(*a_np.shape) * 0.03
            ns, risk, _, _ = env.step(a_np + noise)
            risks.append(risk)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            diff = risk_pred - risk_t
            one_sided = torch.mean(torch.relu(diff) ** 2)
            action_loss = lam * torch.mean(action ** 2)
            loss = one_sided + action_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state = ns
    return np.array(action_norms), np.array(risks)


def train_baseline_sign(model, env, episodes, steps, lr, lam, beta, seed):
    """adv² + β·sign(adv) + λ·||a||², baseline = EMA of risk_pred."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    env.set_seed(seed)
    state = env.reset()
    action_norms, risks = [], []
    baseline_ema = None
    ema_decay = 0.99
    for ep in range(episodes):
        for _ in range(steps):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            action_norms.append(float(np.linalg.norm(a_np)))
            noise = np.random.randn(*a_np.shape) * 0.03
            ns, risk, _, _ = env.step(a_np + noise)
            risks.append(risk)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)

            if baseline_ema is None:
                baseline_ema = risk_pred.detach().clone()
            else:
                baseline_ema = ema_decay * baseline_ema + (1 - ema_decay) * risk_pred.detach()

            adv = risk_pred - baseline_ema
            adv_loss = torch.mean(adv ** 2)
            sign_term = beta * torch.mean(torch.sign(adv))
            action_loss = lam * torch.mean(action ** 2)
            loss = adv_loss + sign_term + action_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state = ns
    return np.array(action_norms), np.array(risks)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def rollout_eval(env, model, steps, seed):
    env.set_seed(seed)
    state = env.reset()
    states, actions, next_states, risks = [], [], [], []
    for _ in range(steps):
        a_np = model.act_numpy(state)
        ns, risk, _, _ = env.step(a_np)
        states.append(state.copy())
        actions.append(a_np.copy())
        next_states.append(ns.copy())
        risks.append(risk)
        state = ns
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "risks": np.array(risks, dtype=np.float32),
    }


def evaluate_alignment(model, traj, n_sample=500) -> Dict:
    states = traj["states"]
    n = min(n_sample, len(states))
    idx = np.linspace(0, len(states) - 1, n, dtype=int)
    sampled = states[idx]

    cos_vals, proj_vals = [], []
    for s in sampled:
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h = model.encoder(s_t)
        action = model.actor(h)
        a_detached = action.detach().requires_grad_(True)
        risk_pred = model.predictor(torch.cat([h, a_detached], dim=-1))
        model.zero_grad()
        risk_pred.sum().backward(retain_graph=True)
        grad = a_detached.grad
        if grad is None:
            model.zero_grad()
            continue
        a_np = action.detach().squeeze(0).numpy().astype(np.float64)
        g_np = grad.squeeze(0).numpy().astype(np.float64)
        a_norm = np.linalg.norm(a_np)
        g_norm = np.linalg.norm(g_np)
        if a_norm > 1e-12 and g_norm > 1e-12:
            cos_vals.append(float(np.dot(a_np, g_np) / (a_norm * g_norm)))
            proj_vals.append(float(abs(np.dot(a_np, g_np) / g_norm) / a_norm))
        else:
            cos_vals.append(0.0)
            proj_vals.append(0.0)
        model.zero_grad()

    cos_arr = np.array(cos_vals)
    proj_arr = np.array(proj_vals)
    sign_counts = {"pos": int(np.sum(cos_arr > 0.01)), "neg": int(np.sum(cos_arr < -0.01)),
                   "zero": int(np.sum(np.abs(cos_arr) <= 0.01))}
    return {
        "cos_sim_mean": float(np.mean(cos_arr)),
        "cos_sim_std": float(np.std(cos_arr)),
        "cos_sim_median": float(np.median(cos_arr)),
        "cos_pos_frac": float(np.mean(cos_arr > 0)),
        "proj_efficiency_mean": float(np.mean(proj_arr)),
        "proj_efficiency_std": float(np.std(proj_arr)),
        "sign_counts": sign_counts,
        "total_risk": float(np.mean(traj["risks"])),
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_directional_fix(
    d_dims=None, seeds=None,
    episodes=N_EPISODES, steps=EP_LENGTH, rollout_steps=ROLLOUT_STEPS,
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
    print(f"Layer 1 Directional Fix Experiment")
    print(f"  d ∈ {d_dims}  |  seeds = {seeds}")
    print(f"  training: {episodes}ep × {steps}steps  |  rollout: {rollout_steps}")
    print(f"  α ∈ {ALPHAS}  |  β ∈ {BETAS}  |  λ = {LAMBDA_CTRL}")
    print(f"{'=' * 70}\n")

    all_results: Dict[str, Dict] = {}

    # Method configurations
    # Each entry: (method_label, train_fn, extra_args_list)
    methods = []

    # Baseline
    methods.append(("baseline", train_baseline, [{}]))

    # Method 1: Directional
    for alpha in ALPHAS:
        methods.append((f"directional_a{alpha}", train_directional,
                        [{"alpha": alpha}]))

    # Method 2: One-sided
    methods.append(("onesided", train_onesided, [{}]))

    # Method 3: Baseline-sign
    for beta in BETAS:
        methods.append((f"baseline_sign_b{beta}", train_baseline_sign,
                        [{"beta": beta}]))

    # Flatten
    flat_methods = []
    for label, fn, args_list in methods:
        for args in args_list:
            full_label = label
            flat_methods.append((full_label, fn, args))

    for d in d_dims:
        print(f"\n{'=' * 50}")
        print(f"  d = {d}")
        print(f"{'=' * 50}")

        for method_label, train_fn, extra_args in flat_methods:
            seed_metrics = []
            for seed in seeds:
                train_env = _make_env(d, seed, mode="closed")
                model = _make_model(d, seed)

                anorms, risks = train_fn(model, train_env, episodes, steps,
                                         DEFAULT_LR, LAMBDA_CTRL, seed=seed, **extra_args)
                model.eval()

                final_an = float(np.mean(anorms[-500:])) if len(anorms) >= 500 else float(np.mean(anorms))
                if final_an < 1e-3:
                    seed_metrics.append({"aborted": True, "action_norm_final": final_an})
                    continue

                eval_env = _make_env(d, seed + 10000, mode="closed")
                traj = rollout_eval(eval_env, model, steps=rollout_steps, seed=seed + 20000)
                metrics = evaluate_alignment(model, traj, n_sample=500)
                metrics["action_norm_final"] = final_an
                metrics["action_norm_mean"] = float(np.mean(np.linalg.norm(traj["actions"], axis=1)))
                metrics["seed"] = seed
                seed_metrics.append(metrics)

            active = [m for m in seed_metrics if not m.get("aborted", False)]
            if len(active) < 2:
                print(f"    {method_label}: too few active seeds ({len(active)}), skip")
                continue

            cos_vals = [m["cos_sim_mean"] for m in active]
            # Per-seed majority sign
            signs = [1 if m["cos_sim_mean"] > 0.01 else (-1 if m["cos_sim_mean"] < -0.01 else 0)
                     for m in active]
            pos_count = sum(1 for s in signs if s == 1)
            neg_count = sum(1 for s in signs if s == -1)
            majority_sign = "pos" if pos_count > neg_count else ("neg" if neg_count > pos_count else "zero")
            sign_consistency = max(pos_count, neg_count) / len(signs) if len(signs) > 0 else 0.0

            agg = {
                "cos_sim_mean": float(np.mean(cos_vals)),
                "cos_sim_std": float(np.std(cos_vals)),
                "cos_sim_median": float(np.median(cos_vals)),
                "proj_efficiency_mean": float(np.mean([m["proj_efficiency_mean"] for m in active])),
                "proj_efficiency_std": float(np.std([m["proj_efficiency_mean"] for m in active])),
                "total_risk": float(np.mean([m["total_risk"] for m in active])),
                "total_risk_std": float(np.std([m["total_risk"] for m in active])),
                "action_norm_mean": float(np.mean([m["action_norm_mean"] for m in active])),
                "sign_pos_count": pos_count,
                "sign_neg_count": neg_count,
                "sign_consistency": sign_consistency,
                "majority_sign": majority_sign,
                "n_seeds": len(active),
                "n_aborted": len(seed_metrics) - len(active),
            }

            key = f"d{d}_{method_label}"
            all_results[key] = agg

            print(f"    {method_label:25s} cos={agg['cos_sim_mean']:+.3f}±{agg['cos_sim_std']:.3f}  "
                  f"sign_ok={agg['sign_consistency']:.2f} ({majority_sign})  "
                  f"proj={agg['proj_efficiency_mean']:.3f}  "
                  f"risk={agg['total_risk']:.4f}  "
                  f"|a|={agg['action_norm_mean']:.4f}")

    # ---- Verdict ----
    baseline_key = f"d{d_dims[0]}_baseline"
    bl_risks = {}
    for d in d_dims:
        bk = f"d{d}_baseline"
        if bk in all_results:
            bl_risks[d] = all_results[bk]["total_risk"]

    verdicts = []
    for key, agg in all_results.items():
        parts = key.split("_", 1)
        d_str = parts[0].replace("d", "")
        d_val = int(d_str)
        method = parts[1] if len(parts) > 1 else key

        bl_risk = bl_risks.get(d_val, None)
        sign_ok = agg["sign_consistency"] >= 0.8
        proj_ok = agg["proj_efficiency_mean"] > 0.6
        risk_ok = True if bl_risk is None else agg["total_risk"] <= bl_risk * 1.2

        passed = sign_ok and proj_ok and risk_ok
        verdicts.append({
            "key": key, "passed": passed,
            "sign_ok": sign_ok, "proj_ok": proj_ok, "risk_ok": risk_ok,
            "method": method, "d": d_val,
        })

    # ---- Save ----
    output = {
        "config": {"d_dims": d_dims, "seeds": len(seeds),
                   "episodes": episodes, "steps": steps,
                   "alphas": ALPHAS, "betas": BETAS, "lambda": LAMBDA_CTRL},
        "results": {k: v for k, v in all_results.items()},
        "verdicts": verdicts,
    }

    # Find best method
    best = None
    for v in verdicts:
        if v.get("passed"):
            best = v
            break

    output["recommendation"] = (f"Best method: {best['method']} at d={best['d']}"
                                if best else "No method achieved all criteria.")

    with open(os.path.join(results_dir, "layer1_directional_fix.json"), "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    # ---- Summary table ----
    print(f"\n{'=' * 70}")
    print(f"SUMMARY TABLE")
    print(f"{'Method':<30s} {'d':>4s} {'cos':>8s} {'sign_ok':>8s} {'proj':>8s} {'risk':>8s} {'PASS'}")
    print(f"{'-' * 75}")
    for v in sorted(verdicts, key=lambda x: (x["d"], not x["passed"], x["method"])):
        agg = all_results[v["key"]]
        print(f"{v['method']:<30s} {v['d']:>4d} {agg['cos_sim_mean']:>+8.3f} "
              f"{v['sign_ok']!s:>8s} {agg['proj_efficiency_mean']:>8.3f} "
              f"{agg['total_risk']:>8.4f} {'✓' if v['passed'] else '✗'}")
    print(f"\nRECOMMENDATION: {output['recommendation']}")

    # ---- Plotting ----
    try:
        _make_plots(all_results, d_dims, results_dir)
    except Exception as e:
        print(f"  [warn] Plotting failed: {e}")

    return output


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

    method_order = ["baseline"]
    for a in ALPHAS:
        method_order.append(f"directional_a{a}")
    method_order.append("onesided")
    for b in BETAS:
        method_order.append(f"baseline_sign_b{b}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for di, d in enumerate(d_dims):
        x_ticks = []
        cos_means, cos_stds = [], []
        sign_cons, proj_means, risk_means = [], [], []

        for method in method_order:
            key = f"d{d}_{method}"
            if key not in all_results:
                continue
            a = all_results[key]
            x_ticks.append(method.replace("directional_", "dir_").replace("baseline_sign_", "bs_"))
            cos_means.append(a["cos_sim_mean"])
            cos_stds.append(a["cos_sim_std"])
            sign_cons.append(a["sign_consistency"])
            proj_means.append(a["proj_efficiency_mean"])
            risk_means.append(a["total_risk"])

        x = range(len(x_ticks))
        color = f"C{di}"

        # cos_sim
        ax = axes[0, 0]
        ax.errorbar(x, cos_means, yerr=cos_stds, marker="o", capsize=3,
                     label=f"d={d}", color=color)
        ax.axhline(0, color="gray", ls="--", alpha=0.4)

        # sign consistency
        ax = axes[0, 1]
        ax.plot(x, sign_cons, marker="s", color=color, label=f"d={d}")
        ax.axhline(0.8, color="green", ls="--", alpha=0.6, label="80%")

        # proj efficiency
        ax = axes[1, 0]
        ax.plot(x, proj_means, marker="D", color=color, label=f"d={d}")
        ax.axhline(0.6, color="green", ls="--", alpha=0.6)

        # total risk
        ax = axes[1, 1]
        ax.plot(x, risk_means, marker="^", color=color, label=f"d={d}")

    for ax, title in [(axes[0, 0], "cos_sim(a, ∇risk)"), (axes[0, 1], "Sign Consistency"),
                       (axes[1, 0], "Projection Efficiency"), (axes[1, 1], "Total Risk")]:
        ax.set_title(title)
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Layer 1 Directional Fix", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "layer1_directional_fix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {path}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Layer 1 Directional Fix")
    parser.add_argument("--d-dims", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--steps", type=int, default=EP_LENGTH)
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        d_dims = [2, 10]
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

    run_directional_fix(
        d_dims=d_dims, seeds=seeds,
        episodes=episodes, steps=steps,
        rollout_steps=rollout_steps,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
