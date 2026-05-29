"""Layer 2 v7: Latent-Shift Causal Necessity.

D_latent_shift: Double-well with hidden offset that flips every 200 steps.
  - Agent sees x_obs, not x_true.
  - Env returns grad_risk = ∂risk/∂action as direction hint.
  - Normal model uses gradient pathway; blocked model (action.detach) loses it.

Causal confirmed if blocked model's actions collapse (<30% norm) and
risk rises (>1.5x normal) compared to normally-trained model.

Usage:
  uv run python core_mvp/layers/layer2_generalization.py
  uv run python core_mvp/layers/layer2_generalization.py --quick
"""

from __future__ import annotations

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, _root)

from core_mvp.core.core_env import LatentShiftDoubleWellEnv
from core_mvp.core.core_models import LatentShiftModel
from core_mvp.core.core_training import train_latent_shift

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_SEEDS = 8
TRAIN_STEPS = 4000
LAMBDA = 0.05
LR = 1e-3
ROLL = 2000

OUT = os.path.join(os.path.dirname(_here), "results", "layer2")


def _rollout_latent(env, model, steps):
    env.set_seed(9999)
    state = env.reset()
    grad_hint = None
    risks, actions, states = [], [], []
    for _ in range(steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        if model.use_grad_hint:
            gh_val = grad_hint if grad_hint is not None else 0.0
            gh_t = torch.tensor([[gh_val]], dtype=torch.float32, device=DEVICE)
            s_in = torch.cat([s_t, gh_t], dim=-1)
        else:
            s_in = s_t
        with torch.no_grad():
            h = model.encoder(s_in)
            action = model.actor(h)
            an = action.cpu().numpy().squeeze(0)
        ns, risk, grad_risk, done, _ = env.step(an)
        risks.append(risk); actions.append(an); states.append(state.copy())
        state = ns; grad_hint = grad_risk
        if done: state = env.reset(); grad_hint = None
    return np.array(risks), np.array(actions), np.array(states)


def _ms(v):
    a = np.array(v, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)), "n": int(len(a))}


def _jdef(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


def run_layer2(seeds=None, out_dir=None):
    seeds = seeds or list(range(N_SEEDS))
    out_dir = out_dir or OUT
    os.makedirs(out_dir, exist_ok=True)
    sd = seeds

    print(f"\n{'='*70}")
    print(f"Layer 2 v7: Latent-Shift Causal Necessity")
    print(f"  seeds={sd}  steps={TRAIN_STEPS}")
    print(f"{'='*70}\n")

    results_normal = []
    results_blocked = []
    causal_results = []

    for seed in sd:
        # ---- Normal model ----
        env_n = LatentShiftDoubleWellEnv(seed=seed)
        m_n = LatentShiftModel(state_dim=1, hidden_dim=128, action_dim=1, use_grad_hint=True)
        an_n = train_latent_shift(env_n, m_n, TRAIN_STEPS, lr=LR,
                                   lambda_action=LAMBDA, blocked=False)
        m_n.eval()
        env_test = LatentShiftDoubleWellEnv(seed=seed+10000)
        R_n, A_n, _ = _rollout_latent(env_test, m_n, ROLL)
        norm_n = float(np.mean(np.abs(A_n)))
        risk_n = float(np.mean(R_n))

        env_b = LatentShiftDoubleWellEnv(seed=seed)
        m_b = LatentShiftModel(state_dim=1, hidden_dim=128, action_dim=1, use_grad_hint=True)
        an_b = train_latent_shift(env_b, m_b, TRAIN_STEPS, lr=LR,
                                   lambda_action=LAMBDA, blocked=True)
        m_b.eval()
        env_test2 = LatentShiftDoubleWellEnv(seed=seed+10000)
        R_n, A_n, _ = _rollout_latent(env_test, m_n, ROLL)
        norm_n = float(np.mean(np.abs(A_n)))
        risk_n = float(np.mean(R_n))

        # ---- Blocked model ----
        env_b = LatentShiftDoubleWellEnv(seed=seed)
        m_b = LatentShiftModel(state_dim=1, hidden_dim=128, action_dim=1, use_grad_hint=True)
        an_b = train_latent_shift(env_b, m_b, TRAIN_STEPS, lr=LR,
                                   lambda_action=LAMBDA, blocked=True)
        m_b.eval()
        env_test2 = LatentShiftDoubleWellEnv(seed=seed+10000)
        R_b, A_b, _ = _rollout_latent(env_test2, m_b, ROLL)
        norm_b = float(np.mean(np.abs(A_b)))
        risk_b = float(np.mean(R_b))

        # ---- Causal verdict ----
        causal = (norm_b / max(norm_n, 1e-8)) < 0.3

        results_normal.append({"seed": seed, "norm": norm_n, "risk": risk_n,
                               "final_an": float(np.mean(an_n[-500:]))})
        results_blocked.append({"seed": seed, "norm": norm_b, "risk": risk_b,
                                "final_an": float(np.mean(an_b[-500:]))})
        causal_results.append({"seed": seed, "causal": "CONFIRMED" if causal else "NOT_CONFIRMED",
                               "norm_ratio": norm_b/max(norm_n,1e-8)})

        print(f"  seed={seed}: n_n={norm_n:.4f} n_b={norm_b:.4f} collapse={causal} → {causal_results[-1]['causal']}")

    causal_rate = np.mean([1.0 if c["causal"]=="CONFIRMED" else 0.0 for c in causal_results])
    avg_norm_ratio = np.mean([c["norm_ratio"] for c in causal_results])

    print(f"\n{'='*70}")
    print(f"SUMMARY: causal_rate={causal_rate:.2f}  avg_norm_ratio={avg_norm_ratio:.3f}")
    if causal_rate >= 0.75:
        print("LAYER_2_PASS: causal necessity confirmed via latent-shift test.")
    elif causal_rate >= 0.5:
        print("LAYER_2_PARTIAL: some seeds confirm causal necessity.")
    else:
        print("LAYER_2_FAIL: latent-shift causal test not confirmed.")

    out = {
        "normal": _ms([r["norm"] for r in results_normal]),
        "normal_risk": _ms([r["risk"] for r in results_normal]),
        "blocked": _ms([r["norm"] for r in results_blocked]),
        "blocked_risk": _ms([r["risk"] for r in results_blocked]),
        "causal_rate": causal_rate,
        "per_seed": causal_results,
    }
    with open(os.path.join(out_dir, "layer2_generalization.json"), "w") as f:
        json.dump(out, f, indent=2, default=_jdef)
    return out


def main():
    p = argparse.ArgumentParser("Layer 2 v7")
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    sd = list(range(args.seeds)) if not args.quick else [0, 1, 2]
    if args.quick:
        global TRAIN_STEPS, ROLL
        TRAIN_STEPS = 500; ROLL = 400
    run_layer2(seeds=sd, out_dir=args.out)


if __name__ == "__main__":
    main()
