"""Layer 1 (Fixed): Closed-Loop Dominance Verification.

Key fix from V4→V5: predictor consumes [h, action], creating a gradient
pathway from prediction loss back to the actor.  This enables genuine
closed-loop control — not just statistical correlation.

Pipeline:
  Step 1 — Lambda calibration (scan {0.01, 0.1, 0.5}) per d.
  Step 2 — Train ClosedLoopModel on mode='closed' for each (d, seed).
           Abort early if mean|a| < 1e-3 at convergence.
  Step 3 — Rollout in closed / open / pseudo modes.
  Step 4 — Compute r, W2, pseudo_ratio, gain, grad_norm.
  Step 5 — Aggregate, plot, verdict.

Usage:
  uv run python core_mvp_v4/experiments/layer1_closed_loop_dominance.py
  uv run python core_mvp_v4/experiments/layer1_closed_loop_dominance.py --quick
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core_mvp_v4.core_env import MultiModeEnv, PseudoActionGenerator
from core_mvp_v4.core_models import ClosedLoopModel, get_designed_hidden_dim
from core_mvp_v4.core_metrics import (
    compute_wasserstein,
    compute_w2_gaussian,
    compute_spearman_correlation,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
K = 2
D_DIMS = [2, 5, 10, 20, 50, 100]
N_SEEDS = 5
N_EPISODES = 5
EP_LENGTH = 2000
ROLLOUT_STEPS = 1000
DEFAULT_LR = 1e-3
LAMBDA_SCAN = [0.01, 0.1, 0.5]
LAMBDA_CALIB_STEPS = 3000
DRIFT = 0.5
QUICK_DIMS = [2, 8, 16, 32]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "layer1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(d: int, seed: int, mode: str = "closed") -> MultiModeEnv:
    env = MultiModeEnv(
        d_total=d, k_controlled=K, mode=mode,
        theta=0.5, noise_std=0.05, coupling=0.05,
        drift=DRIFT, force_scale=0.1, action_scale=0.1,
        seed=seed, pseudo_affects_noise=False,
    )
    env.calibrate_noise_scales()
    return env


def _make_model(d: int, seed: int) -> ClosedLoopModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden_dim = get_designed_hidden_dim(d)
    return ClosedLoopModel(state_dim=d, hidden_dim=hidden_dim, action_dim=K)


# ---------------------------------------------------------------------------
# Lambda calibration
# ---------------------------------------------------------------------------

def _calibrate_lambda(
    d: int,
    candidates: List[float] = LAMBDA_SCAN,
    steps: int = LAMBDA_CALIB_STEPS,
    seed: int = 9999,
    target_action_norm: float = 0.05,
) -> Tuple[float, Dict[str, float]]:
    """Short training run for each lambda candidate; pick the one yielding
    non-zero action norms closest to target_action_norm."""
    results = {}
    for lam in candidates:
        env = _make_env(d, seed, mode="closed")
        model = _make_model(d, seed)
        action_norms = _quick_train(model, env, steps=steps,
                                    lambda_ctrl=lam, seed=seed)
        final_an = float(np.mean(action_norms[-200:])) if len(action_norms) >= 200 else float(np.mean(action_norms))
        results[str(lam)] = final_an
        print(f"    λ={lam}: final mean|a|={final_an:.5f}")

    nonzero = {lam: v for lam, v in results.items() if v > 1e-4}
    if not nonzero:
        best_lam = candidates[-1]
    else:
        best_lam = float(min(nonzero, key=lambda k: abs(nonzero[k] - target_action_norm)))
    print(f"    selected λ={best_lam}")

    return best_lam, results


def _quick_train(model, env, steps=3000, lambda_ctrl=0.1, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)
    env.set_seed(seed)
    state = env.reset()
    action_norms = []
    for _ in range(steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        action, h, risk_pred = model(s_t)
        a_np = action.squeeze(0).detach().numpy()
        action_norms.append(float(np.linalg.norm(a_np)))

        noise = np.random.randn(*a_np.shape) * 0.03
        a_noisy = a_np + noise
        next_state, risk_actual, _, _ = env.step(a_noisy)

        risk_t = torch.tensor([[risk_actual]], dtype=torch.float32)
        pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
        action_loss = torch.mean(action ** 2)
        loss = pred_loss + lambda_ctrl * action_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        state = next_state

    return np.array(action_norms)


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def train_closed_loop(
    model: ClosedLoopModel,
    env: MultiModeEnv,
    num_episodes: int = N_EPISODES,
    episode_length: int = EP_LENGTH,
    lr: float = DEFAULT_LR,
    lambda_ctrl: float = 0.1,
    seed: int = 0,
    exploration_noise: float = 0.03,
) -> Tuple[List[Dict], np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    env.set_seed(seed)

    all_action_norms: List[float] = []
    logs: List[Dict] = []

    for ep in range(num_episodes):
        state = env.reset()
        for _ in range(episode_length):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            all_action_norms.append(float(np.linalg.norm(a_np)))

            noise = np.random.randn(*a_np.shape) * exploration_noise
            a_noisy = a_np + noise
            next_state, risk_actual, _, _ = env.step(a_noisy)

            risk_t = torch.tensor([[risk_actual]], dtype=torch.float32)
            pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
            action_loss = torch.mean(action ** 2)
            loss = pred_loss + lambda_ctrl * action_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            logs.append({"pred_loss": float(pred_loss.item()),
                         "action_loss": float(action_loss.item())})
            state = next_state

    return logs, np.array(all_action_norms)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def rollout_closed(
    env: MultiModeEnv, model: ClosedLoopModel, steps: int = ROLLOUT_STEPS, seed: int = 0,
) -> Dict[str, np.ndarray]:
    env.set_seed(seed)
    state = env.reset()
    states, actions, next_states, risks = [], [], [], []
    for _ in range(steps):
        a_np = model.act_numpy(state)
        next_state, risk, _, _ = env.step(a_np)
        states.append(state.copy())
        actions.append(a_np.copy())
        next_states.append(next_state.copy())
        risks.append(risk)
        state = next_state
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "risks": np.array(risks, dtype=np.float32),
    }


def rollout_open(
    env: MultiModeEnv, steps: int = ROLLOUT_STEPS, seed: int = 0,
) -> Dict[str, np.ndarray]:
    env.set_seed(seed)
    state = env.reset()
    states, actions, next_states, risks = [], [], [], []
    for _ in range(steps):
        a_np = np.zeros(K, dtype=np.float32)
        next_state, risk, _, _ = env.step(a_np)
        states.append(state.copy())
        actions.append(a_np.copy())
        next_states.append(next_state.copy())
        risks.append(risk)
        state = next_state
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "risks": np.array(risks, dtype=np.float32),
    }


def rollout_pseudo(
    env: MultiModeEnv,
    pseudo_gen: PseudoActionGenerator,
    steps: int = ROLLOUT_STEPS,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    env.set_seed(seed)
    state = env.reset()
    pseudo_gen.reset(state)
    states, actions, next_states, risks = [], [], [], []
    for _ in range(steps):
        a_np = pseudo_gen.step(state)
        next_state, risk, _, _ = env.step(a_np)
        states.append(state.copy())
        actions.append(a_np.copy())
        next_states.append(next_state.copy())
        risks.append(risk)
        state = next_state
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "risks": np.array(risks, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_correlation_r(actions: np.ndarray, next_states: np.ndarray, k: int = K) -> Dict:
    if actions.shape[0] < 2:
        return {"r_max_abs": 0.0, "r_mean_abs": 0.0}
    s_ctrl = next_states[:, :k]
    n_a, n_c = actions.shape[1], s_ctrl.shape[1]
    corr_matrix = np.zeros((n_a, n_c))
    for i in range(n_a):
        for j in range(n_c):
            rho, _ = compute_spearman_correlation(actions[:, i], s_ctrl[:, j])
            corr_matrix[i, j] = rho if not np.isnan(rho) else 0.0
    abs_corr = np.abs(corr_matrix)
    return {
        "r_max_abs": float(np.max(abs_corr)),
        "r_mean_abs": float(np.mean(abs_corr)),
    }


def compute_distribution_metrics(
    traj_cl: Dict, traj_open: Dict, traj_pseudo: Dict, k: int = K,
) -> Dict:
    s_cl = traj_cl["next_states"][:, :k]
    s_op = traj_open["next_states"][:, :k]
    s_ps = traj_pseudo["next_states"][:, :k]

    w1_cl_op = compute_wasserstein(s_cl, s_op)
    w1_ps_op = compute_wasserstein(s_ps, s_op)
    w2_cl_op = compute_w2_gaussian(s_cl, s_op)
    w2_ps_op = compute_w2_gaussian(s_ps, s_op)

    pseudo_ratio = w2_ps_op / max(w2_cl_op, 1e-12)

    return {
        "w1_close_open": float(w1_cl_op),
        "w1_pseudo_open": float(w1_ps_op),
        "w2_close_open": float(w2_cl_op),
        "w2_pseudo_open": float(w2_ps_op),
        "pseudo_ratio": float(pseudo_ratio),
    }


def compute_gain(traj_cl: Dict, traj_open: Dict, k: int = K) -> float:
    s_cl = traj_cl["next_states"][:, :k]
    s_op = traj_open["next_states"][:, :k]
    diffs = np.linalg.norm(s_cl - s_op, axis=1)
    return float(np.mean(diffs))


def compute_grad_norm(model: ClosedLoopModel, states_batch: np.ndarray,
                       n_sample: int = 200) -> float:
    n = min(n_sample, len(states_batch))
    if n == 0:
        return 0.0
    idx = np.linspace(0, len(states_batch) - 1, n, dtype=int)
    batch = states_batch[idx]
    return model.compute_grad_norm(batch)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def compute_verdict(
    r_closed: float, r_pseudo: float,
    w2: float, pseudo_ratio: float,
    grad_norm: float, gain: float,
    action_norm: float, d: int,
) -> Dict:
    c1_w2 = w2 > 1e-4
    c2_pr = pseudo_ratio < 0.3
    c3_r = r_closed > 0.05 and (r_closed > r_pseudo * 1.1 or r_pseudo < 0.01)
    c4_gn = grad_norm > 1e-6
    c5_gain = gain > 1e-4
    c6_an = action_norm > 1e-3

    if d == K:
        c2_pr = True

    all_pass = all([c1_w2, c2_pr, c3_r, c4_gn, c5_gain, c6_an])
    return {
        "pass_all": bool(all_pass),
        "criteria": {
            "c1_w2": {"pass": c1_w2, "value": w2},
            "c2_pseudo_ratio": {"pass": c2_pr, "value": pseudo_ratio},
            "c3_correlation": {"pass": c3_r, "r_closed": r_closed, "r_pseudo": r_pseudo},
            "c4_grad_norm": {"pass": c4_gn, "value": grad_norm},
            "c5_gain": {"pass": c5_gain, "value": gain},
            "c6_action_norm": {"pass": c6_an, "value": action_norm},
        },
        "fail_mode": _classify_failure(c1_w2, c2_pr, c3_r, c4_gn, c5_gain, c6_an),
    }


def _classify_failure(c1, c2, c3, c4, c5, c6) -> str:
    if all([c1, c2, c3, c4, c5, c6]):
        return "ALL_PASS"
    if not c6:
        return "ZERO_ACTION: mean|a| → 0, policy collapsed, no control signal."
    if not c4:
        return "ZERO_GRADIENT: ∂risk_pred/∂action ≈ 0, gradient pathway not utilized."
    if not c5:
        return "ZERO_GAIN: actions do not alter physical state."
    if not c1:
        return "ZERO_W2: no distribution shift between closed and open."
    if not c2:
        return "HIGH_PSEUDO_RATIO: effect explained by statistical correlation."
    if not c3:
        return "LOW_CORRELATION: action-state correlation too weak."
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_l1_fixed(
    d_dims: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    episodes: int = N_EPISODES,
    steps: int = EP_LENGTH,
    rollout_steps: int = ROLLOUT_STEPS,
    results_dir: Optional[str] = None,
    lambda_override: Optional[float] = None,
) -> Dict:
    if d_dims is None:
        d_dims = D_DIMS
    if seeds is None:
        seeds = list(range(N_SEEDS))
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Layer 1 (Fixed): Closed-Loop Dominance — Action-Aware Predictor")
    print(f"  d ∈ {d_dims}")
    print(f"  seeds = {seeds}")
    print(f"  training: {episodes}ep × {steps}steps")
    print(f"  rollout: {rollout_steps} steps")
    print(f"  k = {K}")
    print(f"{'=' * 70}\n")

    aggregated: Dict[str, Dict] = {}
    lambda_selections: Dict[str, float] = {}

    for d in d_dims:
        print(f"\n--- d = {d} ---")
        t0 = time.time()

        # Step 1: Lambda calibration
        if lambda_override is not None:
            best_lambda = lambda_override
            print(f"    λ override = {best_lambda}")
        else:
            print(f"  [calibrating λ]")
            best_lambda, lam_results = _calibrate_lambda(d)
        lambda_selections[f"d{d}"] = best_lambda

        seed_results: Dict[int, Dict] = {}

        for seed in seeds:
            print(f"  seed={seed}  ", end="", flush=True)

            # Step 2: Train
            train_env = _make_env(d, seed, mode="closed")
            model = _make_model(d, seed)
            _, action_norms = train_closed_loop(
                model, train_env, num_episodes=episodes,
                episode_length=steps, lr=DEFAULT_LR,
                lambda_ctrl=best_lambda, seed=seed,
            )

            final_an = float(np.mean(action_norms[-500:])) if len(action_norms) >= 500 else float(np.mean(action_norms))
            std_an = float(np.std(action_norms[-500:])) if len(action_norms) >= 500 else float(np.std(action_norms))

            if final_an < 1e-3:
                print(f"|a|={final_an:.5f} < 1e-3 → ABORT (no control formed)")
                seed_results[seed] = {
                    "d": d, "seed": seed,
                    "action_norm_mean": final_an,
                    "action_norm_std": std_an,
                    "best_lambda": best_lambda,
                    "aborted": True,
                    "fail_mode": "ZERO_ACTION",
                }
                continue

            model.eval()

            # Step 3: Rollouts
            base_env = _make_env(d, seed + 10000, mode="closed")
            base_env.calibrate_noise_scales()

            cl_env = deepcopy(base_env)
            cl_env.set_mode("closed")
            traj_cl = rollout_closed(cl_env, model, steps=rollout_steps, seed=seed + 20000)

            open_env = deepcopy(base_env)
            open_env.set_mode("open")
            traj_open = rollout_open(open_env, steps=rollout_steps, seed=seed + 20000)

            pseudo_env = deepcopy(base_env)
            pseudo_env.set_mode("pseudo")
            pseudo_gen = PseudoActionGenerator(d, K, seed=seed + 30000, scale=0.1)
            traj_pseudo = rollout_pseudo(pseudo_env, pseudo_gen, steps=rollout_steps, seed=seed + 20000)

            # Step 4: Metrics
            r_closed = compute_correlation_r(traj_cl["actions"], traj_cl["next_states"])
            r_pseudo = compute_correlation_r(traj_pseudo["actions"], traj_pseudo["next_states"])
            dist_m = compute_distribution_metrics(traj_cl, traj_open, traj_pseudo)
            gain_val = compute_gain(traj_cl, traj_open)
            grad_norm_val = compute_grad_norm(model, traj_cl["states"])
            action_norm_cl = float(np.mean(np.linalg.norm(traj_cl["actions"], axis=1)))

            verdict = compute_verdict(
                r_closed=r_closed["r_max_abs"],
                r_pseudo=r_pseudo["r_max_abs"],
                w2=dist_m["w2_close_open"],
                pseudo_ratio=dist_m["pseudo_ratio"],
                grad_norm=grad_norm_val,
                gain=gain_val,
                action_norm=action_norm_cl,
                d=d,
            )

            result = {
                "d": d, "seed": seed,
                "r_closed": r_closed, "r_pseudo": r_pseudo,
                "dist_metrics": dist_m,
                "gain": gain_val,
                "grad_norm": grad_norm_val,
                "action_norm_mean": action_norm_cl,
                "action_norm_train_final": final_an,
                "action_norm_train_std": std_an,
                "best_lambda": best_lambda,
                "aborted": False,
                "verdict": verdict,
            }
            seed_results[seed] = result

            print(f"r_cl={r_closed['r_max_abs']:.3f} r_ps={r_pseudo['r_max_abs']:.3f} "
                  f"w2={dist_m['w2_close_open']:.4f} pr={dist_m['pseudo_ratio']:.3f} "
                  f"gain={gain_val:.4f} gn={grad_norm_val:.4f} |a|={action_norm_cl:.4f} "
                  f"{'PASS' if verdict['pass_all'] else verdict.get('fail_mode','FAIL')[:20]}")

        agg = _aggregate_seeds(seed_results, d)
        aggregated[f"d{d}"] = agg

        d_out = {"per_seed": _serialise(seed_results), "aggregated": agg, "lambda": best_lambda}
        with open(os.path.join(results_dir, f"d{d}_aggregated.json"), "w") as f:
            json.dump(d_out, f, indent=2, default=_json_default)

        elapsed = time.time() - t0
        print(f"  d={d} done in {elapsed:.1f}s  "
              f"W2={agg.get('w2_close_open',{}).get('mean','?'):.4f}  "
              f"pr={agg.get('pseudo_ratio',{}).get('mean','?'):.3f}  "
              f"pass={agg.get('pass_rate',{}).get('mean','?')}")

    # Summary
    summary = _build_summary(aggregated, d_dims, lambda_selections)
    with open(os.path.join(results_dir, "layer1_closed_loop_dominance.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    try:
        _make_plots(aggregated, d_dims, results_dir)
    except Exception as e:
        print(f"  [warn] Plotting failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"VERDICT: {summary.get('verdict','')}")
    print(f"{'=' * 70}")
    return summary


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_seeds(seed_results: Dict, d: int) -> Dict:
    active = {k: v for k, v in seed_results.items() if not v.get("aborted", False)}
    n_seeds = len(seed_results)
    n_active = len(active)

    def _vals(key, default=0.0):
        return [s.get(key, default) for s in seed_results.values()]

    def _vals_active(key, default=0.0):
        return [s.get(key, default) for s in active.values()]

    r_cl_max = _vals_active("r_closed", {})
    r_cl_max_v = [v.get("r_max_abs", 0.0) if isinstance(v, dict) else 0.0 for v in r_cl_max]
    r_ps_max = _vals_active("r_pseudo", {})
    r_ps_max_v = [v.get("r_max_abs", 0.0) if isinstance(v, dict) else 0.0 for v in r_ps_max]

    dm = [_vals_active("dist_metrics", {})]
    w2co = [s.get("dist_metrics", {}).get("w2_close_open", 0.0) for s in active.values()]
    w2po = [s.get("dist_metrics", {}).get("w2_pseudo_open", 0.0) for s in active.values()]
    prs  = [s.get("dist_metrics", {}).get("pseudo_ratio", 0.0) for s in active.values()]
    gains= [s.get("gain", 0.0) for s in active.values()]
    gn   = [s.get("grad_norm", 0.0) for s in active.values()]
    an   = [s.get("action_norm_mean", 0.0) for s in active.values()]
    an_t = [s.get("action_norm_train_final", 0.0) for s in active.values()]

    passes = [1.0 if s.get("verdict", {}).get("pass_all", False) else 0.0 for s in active.values()]

    aborted_count = n_seeds - n_active

    return {
        "n_seeds": n_seeds,
        "n_active": n_active,
        "n_aborted": aborted_count,
        "r_closed_max_abs": _mean_std(r_cl_max_v) if r_cl_max_v else {"mean": 0.0, "std": 0.0, "n": 0},
        "r_pseudo_max_abs": _mean_std(r_ps_max_v) if r_ps_max_v else {"mean": 0.0, "std": 0.0, "n": 0},
        "w2_close_open": _mean_std(w2co) if w2co else {"mean": 0.0, "std": 0.0, "n": 0},
        "w2_pseudo_open": _mean_std(w2po) if w2po else {"mean": 0.0, "std": 0.0, "n": 0},
        "pseudo_ratio": _mean_std(prs) if prs else {"mean": 0.0, "std": 0.0, "n": 0},
        "gain": _mean_std(gains) if gains else {"mean": 0.0, "std": 0.0, "n": 0},
        "grad_norm": _mean_std(gn) if gn else {"mean": 0.0, "std": 0.0, "n": 0},
        "action_norm": _mean_std(an) if an else {"mean": 0.0, "std": 0.0, "n": 0},
        "action_norm_train": _mean_std(an_t) if an_t else {"mean": 0.0, "std": 0.0, "n": 0},
        "pass_rate": _mean_std(passes) if passes else {"mean": 0.0, "std": 0.0, "n": 0},
        "fail_modes": _count_fail_modes(active),
    }


def _count_fail_modes(active: Dict) -> Dict:
    counts: Dict[str, int] = {}
    for s in active.values():
        fm = s.get("verdict", {}).get("fail_mode", "UNKNOWN")
        counts[fm] = counts.get(fm, 0) + 1
    return counts


def _mean_std(vals: List[float]) -> Dict:
    a = np.array(vals, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)), "n": int(len(a))}


def _serialise(d: Dict) -> Dict:
    return {str(k): v for k, v in d.items()}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Summary & plots
# ---------------------------------------------------------------------------

def _build_summary(aggregated: Dict, d_dims: List[int],
                   lambda_sel: Dict[str, float]) -> Dict:
    def _pull(key):
        out = {}
        for d in d_dims:
            e = aggregated.get(f"d{d}", {}).get(key, {})
            out[f"d{d}"] = {"mean": e.get("mean", None), "std": e.get("std", None)}
        return out

    verdict = _make_overall_verdict(aggregated, d_dims)
    return {
        "lambda_selected": lambda_sel,
        "r_vs_d": _pull("r_closed_max_abs"),
        "w2_vs_d": _pull("w2_close_open"),
        "pseudo_ratio_vs_d": _pull("pseudo_ratio"),
        "gain_vs_d": _pull("gain"),
        "grad_norm_vs_d": _pull("grad_norm"),
        "action_norm_vs_d": _pull("action_norm"),
        "pass_rate_vs_d": _pull("pass_rate"),
        "verdict": verdict,
    }


def _make_overall_verdict(aggregated: Dict, d_dims: List[int]) -> str:
    pass_means, w2_means, pr_means, gn_means, gain_means, an_means = [], [], [], [], [], []
    for d in d_dims:
        e = aggregated.get(f"d{d}", {})
        pass_means.append(e.get("pass_rate", {}).get("mean", 0.0))
        w2_means.append(e.get("w2_close_open", {}).get("mean", 0.0))
        pr_means.append(e.get("pseudo_ratio", {}).get("mean", 0.0))
        gn_means.append(e.get("grad_norm", {}).get("mean", 0.0))
        gain_means.append(e.get("gain", {}).get("mean", 0.0))
        an_means.append(e.get("action_norm", {}).get("mean", 0.0))
        ab = e.get("n_aborted", 0)

    avg_pass = float(np.mean(pass_means))
    avg_w2 = float(np.mean(w2_means))
    avg_pr = float(np.mean(pr_means))
    avg_gn = float(np.mean(gn_means))
    avg_gain = float(np.mean(gain_means))
    avg_an = float(np.mean(an_means))

    diag = (f"pass={avg_pass:.2f} W2={avg_w2:.4f} pr={avg_pr:.3f} "
            f"gain={avg_gain:.4f} gn={avg_gn:.4f} |a|={avg_an:.4f}")

    if avg_pass >= 0.5:
        return f"CLOSED_LOOP_CONFIRMED | {diag}"
    elif avg_an < 0.01 and avg_gn < 1e-4:
        return f"FAILED_ACTION_COLLAPSE | {diag} | actions→0, gradient unused."
    elif avg_gn < 1e-4:
        return f"FAILED_ZERO_GRADIENT | {diag} | predictor gradient not flowing to actor."
    elif avg_gain < 1e-3:
        return f"FAILED_ZERO_GAIN | {diag} | actions do not shift physical state."
    elif avg_w2 < 0.001:
        return f"FAILED_NO_DISTRIBUTION_SHIFT | {diag}"
    else:
        return f"FAILED_PASS_RATE_TOO_LOW | {diag}"


def _make_plots(aggregated: Dict, d_dims: List[int], results_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    def _vals(key):
        m, s = [], []
        for d in d_dims:
            e = aggregated.get(f"d{d}", {}).get(key, {})
            m.append(e.get("mean", np.nan))
            s.append(e.get("std", np.nan))
        return np.array(m, dtype=np.float64), np.array(s, dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # r(d)
    ax = axes[0, 0]
    mr, sr = _vals("r_closed_max_abs")
    mrp, srp = _vals("r_pseudo_max_abs")
    ax.errorbar(d_dims, mr, yerr=sr, marker="o", capsize=3, label="closed r")
    ax.errorbar(d_dims, mrp, yerr=srp, marker="s", capsize=3, linestyle="--", label="pseudo r")
    ax.set_xlabel("d"); ax.set_ylabel("max |r(a, s')|"); ax.set_title("Correlation r(d)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # W2(d)
    ax = axes[0, 1]
    mw, sw = _vals("w2_close_open")
    mwp, swp = _vals("w2_pseudo_open")
    ax.errorbar(d_dims, mw, yerr=sw, marker="o", capsize=3, label="W2(closed,open)")
    ax.errorbar(d_dims, mwp, yerr=swp, marker="s", capsize=3, linestyle="--", label="W2(pseudo,open)")
    ax.set_xlabel("d"); ax.set_ylabel("W2"); ax.set_title("Distribution Shift")
    ax.legend(); ax.grid(True, alpha=0.3)

    # pseudo_ratio(d)
    ax = axes[0, 2]
    mpr, spr = _vals("pseudo_ratio")
    ax.errorbar(d_dims, mpr, yerr=spr, marker="o", capsize=3, color="C2")
    ax.axhline(0.3, color="red", linestyle="--", alpha=0.6, label="thresh=0.3")
    ax.set_xlabel("d"); ax.set_ylabel("pseudo_ratio"); ax.set_title("Pseudo Ratio")
    ax.legend(); ax.grid(True, alpha=0.3)

    # gain(d)
    ax = axes[1, 0]
    mg, sg = _vals("gain")
    ax.errorbar(d_dims, mg, yerr=sg, marker="D", capsize=3, color="C3")
    ax.set_xlabel("d"); ax.set_ylabel("gain"); ax.set_title("Control Gain")
    ax.grid(True, alpha=0.3)

    # grad_norm(d)
    ax = axes[1, 1]
    mgn, sgn = _vals("grad_norm")
    ax.errorbar(d_dims, mgn, yerr=sgn, marker="^", capsize=3, color="C4")
    ax.set_xlabel("d"); ax.set_ylabel("||∂risk/∂a||"); ax.set_title("Gradient Strength")
    ax.grid(True, alpha=0.3)

    # pass_rate(d)
    ax = axes[1, 2]
    mp, sp = _vals("pass_rate")
    ax.bar(range(len(d_dims)), mp, yerr=sp, tick_label=[str(d) for d in d_dims],
           capsize=3, color="C5", alpha=0.7)
    ax.set_xlabel("d"); ax.set_ylabel("pass rate"); ax.set_title("Pass Rate")
    ax.set_ylim(0, 1.1); ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Layer 1 (Fixed): Closed-Loop Dominance", fontsize=14)
    fig.tight_layout()
    path = os.path.join(results_dir, "layer1_dominance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Layer 1 Closed-Loop Dominance (Fixed)")
    parser.add_argument("--d-dims", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--steps", type=int, default=EP_LENGTH)
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--lambda-ctrl", type=float, default=None, dest="lam",
                        help="Override lambda calibration with a fixed value")
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

    run_l1_fixed(
        d_dims=d_dims, seeds=seeds,
        episodes=episodes, steps=steps,
        rollout_steps=rollout_steps,
        results_dir=args.results_dir,
        lambda_override=args.lam,
    )


if __name__ == "__main__":
    main()
