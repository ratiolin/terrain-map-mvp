"""Layer 4 Dual-Path: closed-loop actively suppresses wrong-gradient channels.

Model: action = k1 * grad_real + k2 * grad_fake.
Prediction: k1 grows, k2 → 0 as system learns to ignore fake gradient.

Usage: uv run python core_mvp/layers/layer4_dual_path.py
"""

from __future__ import annotations

import os, sys, json, time, argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, _root)

from core_mvp.core.core_env import QuadraticDriftEnv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
OUT = os.path.join(os.path.dirname(_here), "results", "layer4")


class DualPathModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.k1 = nn.Parameter(torch.tensor(0.5))
        self.k2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, grad_real, grad_fake):
        return self.k1 * grad_real + self.k2 * grad_fake

    def suppress_k2(self):
        with torch.no_grad():
            self.k2.data *= 0.0


def _jdef(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


def run_dual_path(n_seeds=8, n_steps=5000, drift=0.1, action_scale=0.05, dt=0.1):
    print(f"\n{'='*70}")
    print(f"Layer 4 Dual-Path: k1*g_real + k2*g_fake  |  {n_seeds} seeds × {n_steps} steps")
    print(f"{'='*70}\n")

    fake_modes = [
        ("flip_1.0", "flip100"),
        ("noise_1.0", "noise"),
    ]

    all_results = {}

    for mode_name, mode_type in fake_modes:
        print(f"\n--- Fake mode: {mode_name} ---")
        sr_base, sr_dual = [], []
        k1_vals, k2_vals = [], []

        for s in range(n_seeds):
            env = QuadraticDriftEnv(drift=drift, action_scale=action_scale, dt=dt, seed=s)

            # ---- Baseline: k1 only ----
            m_base = DualPathModel(); m_base.to(DEVICE)
            m_base.k2.requires_grad = False
            opt = torch.optim.Adam([m_base.k1], lr=0.01)
            state = env.reset(); gr_prev = 0.0; gr_buf = deque(maxlen=10)
            for _ in range(n_steps):
                gr_real = gr_prev
                gr_buf.append(gr_real)
                gr_fake = -gr_real if mode_type == "flip100" else (np.random.randn() * abs(gr_real) if mode_type == "noise" else gr_real)
                gr_r = torch.tensor([gr_real], dtype=torch.float32, device=DEVICE)
                gr_f = torch.tensor([gr_fake], dtype=torch.float32, device=DEVICE)
                a = m_base(gr_r, gr_f); an = a.detach().cpu().numpy().item()
                ns, risk, gr_new, _, _ = env.step(np.float32(an))
                gr_t = torch.tensor([gr_new], dtype=torch.float32, device=DEVICE)
                opt.zero_grad()
                torch.autograd.backward([a], [gr_t.view(1)], retain_graph=True)
                torch.nn.utils.clip_grad_norm_([m_base.k1], 1.0); opt.step()
                state = ns; gr_prev = gr_new
            r_base = float(np.mean(_rollout_simple(env, m_base, mode_type, 500)))
            sr_base.append(r_base)

            # ---- Dual-path ----
            m_dual = DualPathModel(); m_dual.to(DEVICE)
            opt = torch.optim.Adam([m_dual.k1, m_dual.k2], lr=0.01)
            state = env.reset(); gr_prev = 0.0; gr_buf.clear()
            for _ in range(n_steps):
                gr_real = gr_prev
                gr_buf.append(gr_real)
                gr_fake = -gr_real if mode_type == "flip100" else (np.random.randn() * abs(gr_real) if mode_type == "noise" else gr_real)
                gr_r = torch.tensor([gr_real], dtype=torch.float32, device=DEVICE)
                gr_f = torch.tensor([gr_fake], dtype=torch.float32, device=DEVICE)
                a = m_dual(gr_r, gr_f); an = a.detach().cpu().numpy().item()
                ns, risk, gr_new, _, _ = env.step(np.float32(an))
                gr_t = torch.tensor([gr_new], dtype=torch.float32, device=DEVICE)
                opt.zero_grad()
                torch.autograd.backward([a], [gr_t.view(1)], retain_graph=True)
                torch.nn.utils.clip_grad_norm_([m_dual.k1, m_dual.k2], 1.0); opt.step()
                state = ns; gr_prev = gr_new
            k1_vals.append(m_dual.k1.item()); k2_vals.append(m_dual.k2.item())
            r_dual = float(np.mean(_rollout_simple(env, m_dual, mode_type, 500)))
            sr_dual.append(r_dual)

            k1 = m_dual.k1.item(); k2 = m_dual.k2.item()
            ratio = abs(k2) / max(abs(k1), 1e-8)
            print(f"  seed={s}: k1={k1:+.4f} k2={k2:+.4f} |k2/k1|={ratio:.4f}  r_base={sr_base[-1]:.4f} r_dual={sr_dual[-1]:.4f}")

        avg_k1 = float(np.mean(k1_vals)); avg_k2 = float(np.mean(k2_vals))
        suppressed = abs(avg_k2) < 0.1 or abs(avg_k2) / max(abs(avg_k1), 1e-8) < 0.2
        risk_ok = float(np.mean(sr_dual)) < float(np.mean(sr_base)) * 1.2
        all_results[mode_name] = {
            "k1_mean": avg_k1, "k2_mean": avg_k2,
            "k2_per_seed": [float(v) for v in k2_vals],
            "ratio_mean": float(np.mean([abs(k2)/max(abs(k1),1e-8) for k1,k2 in zip(k1_vals,k2_vals)])),
            "risk_baseline": float(np.mean(sr_base)),
            "risk_dual": float(np.mean(sr_dual)),
            "suppressed": suppressed, "risk_ok": risk_ok,
            "pass": suppressed and risk_ok,
        }
        print(f"  → suppressed={suppressed} risk_ok={risk_ok} PASS={all_results[mode_name]['pass']}")

    path = os.path.join(OUT, "dual_path.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=_jdef)
    print(f"\nSaved → {path}")
    return all_results


def _rollout_simple(env, model, mode_type, steps):
    env.set_seed(9999); s = env.reset(); gr_prev = 0.0
    gr_buf = deque(maxlen=10); R = []
    for _ in range(steps):
        gr_real = gr_prev; gr_buf.append(gr_real)
        gr_fake = -gr_real if mode_type == "flip100" else (np.random.randn() * abs(gr_real) if mode_type == "noise" else gr_real)
        gr_r = torch.tensor([gr_real], dtype=torch.float32, device=DEVICE)
        gr_f = torch.tensor([gr_fake], dtype=torch.float32, device=DEVICE)
        with torch.no_grad(): a = model(gr_r, gr_f); an = a.cpu().numpy().item()
        ns, risk, gr_new, _, _ = env.step(np.float32(an))
        R.append(risk); s = ns; gr_prev = gr_new
    return np.array(R)


def main():
    p = argparse.ArgumentParser("Dual Path")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    run_dual_path(n_seeds=args.seeds, n_steps=args.steps)


if __name__ == "__main__":
    main()
