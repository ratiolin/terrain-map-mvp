"""Layer 4 Corrected: recalibrated env (drift=0.1, action_scale=0.05, dt=0.1).

Requires |a|≈2.0 to maintain stability — interventions now have visible effects.

Usage: uv run python core_mvp/layers/layer4_advanced.py
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
from core_mvp.core.core_models import ExistenceDrivenModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
OUT = os.path.join(os.path.dirname(_here), "results", "layer4")

# Calibrated: steady-state action = drift/(action_scale*dt) = 0.03/(0.1*0.15) = 2.0
# But model learns ~0.5-1.0 which partially counters drift → residual risk > 0
# This creates a regime where interventions have visible effects
DRIFT = 0.03
ACTION_SCALE = 0.1
DT = 0.15


def _env(seed=0, drift=None, action_scale=None, dt=None, noise_std=0.0):
    return QuadraticDriftEnv(
        drift=drift if drift is not None else DRIFT,
        action_scale=action_scale if action_scale is not None else ACTION_SCALE,
        dt=dt if dt is not None else DT, seed=seed)


def _model(hd=64):
    return ExistenceDrivenModel(state_dim=1, hidden_dim=hd, action_dim=1)


def _jdef(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


# ===================================================================
# Shared training
# ===================================================================

def _train(env, model, steps, lr=1e-3, lam=0.01, blocked=False):
    model.to(DEVICE); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    s = env.reset(); s_buf = np.empty((1, 1), dtype=np.float32)
    for _ in range(steps):
        s_buf[0, 0] = s[0]; st = torch.from_numpy(s_buf).to(DEVICE)
        a = model(st); an = a.detach().cpu().numpy().item()
        ns, risk, gr, _, _ = env.step(np.float32(an))
        gr_t = torch.tensor([[gr]], dtype=torch.float32, device=DEVICE)
        action_loss = lam * (a**2).sum()
        opt.zero_grad(); action_loss.backward(retain_graph=True)
        if not blocked:
            torch.autograd.backward([a], [gr_t.view(1, 1)], retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        s = ns
    model.eval()


def _rollout(env, model, steps):
    env.set_seed(9999); s = env.reset(); s_buf = np.empty((1, 1), dtype=np.float32)
    R, A = [], []
    for _ in range(steps):
        s_buf[0, 0] = s[0]; st = torch.from_numpy(s_buf).to(DEVICE)
        with torch.no_grad(): a = model(st); an = a.cpu().numpy().item()
        ns, risk, _, _, _ = env.step(np.float32(an)); R.append(risk); A.append(abs(an)); s = ns
    return np.array(R), np.array(A)


# ===================================================================
# EXP 1: Delay Breakdown (recalibrated)
# ===================================================================

def run_delay(n_seeds=5, n_steps=3000):
    print(f"\n{'='*70}\nEXP 1: Delay Breakdown (drift=0.1)\n{'='*70}")
    delays = [0, 5, 10, 20]
    results = {}
    for delta in delays:
        sr = []
        for s in range(n_seeds):
            env = _env(seed=s+500)
            m = _model(); _train(env, m, n_steps)
            env_t = _env(seed=s+600); env_t.set_seed(9999); state = env_t.reset()
            s_buf = np.empty((1, 1), dtype=np.float32)
            buf = deque(maxlen=delta+1); risks = []
            for _ in range(1000):
                s_buf[0, 0] = state[0]; st = torch.from_numpy(s_buf).to(DEVICE)
                a = m(st); an = a.detach().cpu().numpy().item()
                ns, risk, gr, _, _ = env_t.step(np.float32(an))
                risks.append(risk)
                buf.append(gr)
                delay_gr = buf[0] if delta > 0 and len(buf) > delta else gr
                gr_t = torch.tensor([[delay_gr]], dtype=torch.float32, device=DEVICE)
                action_loss = 0.01 * (a**2).sum()
                opt_s = torch.optim.Adam(m.parameters(), lr=1e-4)
                opt_s.zero_grad(); action_loss.backward(retain_graph=True)
                torch.autograd.backward([a], [gr_t.view(1, 1)], retain_graph=True)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt_s.step()
                state = ns
            sr.append(float(np.mean(risks[-500:])))
        results[str(delta)] = {"risk": float(np.mean(sr)), "std": float(np.std(sr))}
        print(f"  Δ={delta}: risk={results[str(delta)]['risk']:.4f}±{results[str(delta)]['std']:.4f}")
    path = os.path.join(OUT, "delay_breakdown.json")
    with open(path, "w") as f: json.dump(results, f, indent=2, default=_jdef)
    return results


# ===================================================================
# EXP 2: Minimal Linear (fixed: a = k * grad_hint, no tanh)
# ===================================================================

class MinimalDirectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(0.1))
    def forward(self, grad_hint):
        return self.k * grad_hint


def run_minimal(n_seeds=5, n_steps=5000):
    print(f"\n{'='*70}\nEXP 2: Minimal Direct (manual grad injection)\n{'='*70}")
    sr = []
    for s in range(n_seeds):
        env = _env(seed=s+700); m = MinimalDirectModel(); m.to(DEVICE)
        opt = torch.optim.SGD([m.k], lr=1.0)
        state = env.reset(); gr_prev = 0.0; log = []
        for _ in range(n_steps):
            gh = torch.tensor([[gr_prev]], dtype=torch.float32, device=DEVICE)
            a = m(gh); an = a.detach().cpu().numpy().item()
            ns, risk, gr_new, _, _ = env.step(np.float32(an))
            gr_t = torch.tensor([[gr_new]], dtype=torch.float32, device=DEVICE)
            opt.zero_grad()
            torch.autograd.backward([a], [gr_t.view(1, 1)], retain_graph=True)
            torch.nn.utils.clip_grad_norm_([m.k], 1.0)
            opt.step()
            state = ns; gr_prev = gr_new; log.append({"risk": risk, "k": m.k.item()})
        sr.append({"final_risk": float(np.mean([l["risk"] for l in log[-500:]])),
                    "final_k": m.k.item()})
        print(f"  seed={s}: k={sr[-1]['final_k']:.4f} risk={sr[-1]['final_risk']:.4f}")
    env_o = _env(seed=9999); ro = []
    for _ in range(500): _, risk, _, _, _ = env_o.step(np.float32(0.0)); ro.append(risk)
    open_r = float(np.mean(ro))
    avg_risk = float(np.mean([r["final_risk"] for r in sr]))
    res = {"minimal_risk": avg_risk, "open_risk": open_r, "seeds": sr,
           "pass": avg_risk < open_r * 0.5}
    print(f"  open_risk={open_r:.4f} minimal={avg_risk:.4f} pass={res['pass']}")
    path = os.path.join(OUT, "minimal_loop.json")
    with open(path, "w") as f: json.dump(res, f, indent=2, default=_jdef)
    return res


# ===================================================================
# EXP 3: Complexity Sweep (kept)
# ===================================================================

def run_complexity(n_seeds=5, n_steps=3000):
    print(f"\n{'='*70}\nEXP 3: Complexity Sweep\n{'='*70}")
    drifts = [0.05, 0.1, 0.2]
    noises = [0.0, 0.05, 0.1]
    results = {}
    for drift in drifts:
        for noise in noises:
            key = f"d{drift}_n{noise}"
            sr_cl, sr_bl, sr_op = [], [], []
            for s in range(n_seeds):
                env = _env(seed=s+800, drift=drift, noise_std=noise)
                m = _model(); R = []; acts = []
                model = m; model.to(DEVICE); model.train()
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                state = env.reset(); s_buf = np.empty((1, 1), dtype=np.float32)
                for _ in range(n_steps):
                    s_buf[0, 0] = state[0]; st = torch.from_numpy(s_buf).to(DEVICE)
                    a = model(st); an = a.detach().cpu().numpy().item()
                    ns, risk, gr, _, _ = env.step(np.float32(an))
                    R.append(risk); acts.append(abs(an))
                    gr_t = torch.tensor([[gr]], dtype=torch.float32, device=DEVICE)
                    al = 0.01 * (a**2).sum()
                    opt.zero_grad(); al.backward(retain_graph=True)
                    torch.autograd.backward([a], [gr_t.view(1, 1)], retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    state = ns
                sr_cl.append(float(np.mean(R[-1000:])))
                # Blocked
                env_b = _env(seed=s+900, drift=drift, noise_std=noise)
                m_b = _model(); _train(env_b, m_b, n_steps, blocked=True)
                ro_b, _ = _rollout(env_b, m_b, 500); sr_bl.append(float(np.mean(ro_b)))
                # Open
                env_o = _env(seed=s+1000, drift=drift, noise_std=noise)
                env_o.set_seed(9999); _ = env_o.reset(); rop = []
                for _ in range(500): _, risk, _, _, _ = env_o.step(np.float32(0.0)); rop.append(risk)
                sr_op.append(float(np.mean(rop)))
            results[key] = {"closed": float(np.mean(sr_cl)), "blocked": float(np.mean(sr_bl)),
                             "open": float(np.mean(sr_op))}
            print(f"  {key}: cl={results[key]['closed']:.4f} bl={results[key]['blocked']:.2f} op={results[key]['open']:.2f}")
    path = os.path.join(OUT, "complexity_sweep.json")
    with open(path, "w") as f: json.dump(results, f, indent=2, default=_jdef)
    return results


# ===================================================================
# EXP 4: Information Deprivation (strengthened)
# ===================================================================

def run_deprivation(n_seeds=5, n_steps=3000):
    print(f"\n{'='*70}\nEXP 4: Information Deprivation\n{'='*70}")
    conds = [("clean", 0.0, 0.0, 0.0),
             ("noise_0.5", 0.5, 0.0, 0.0),
             ("noise_1.0", 1.0, 0.0, 0.0),
             ("noise_2.0", 2.0, 0.0, 0.0),
             ("null_0.3", 0.0, 0.3, 0.0),
             ("null_0.6", 0.0, 0.6, 0.0),
             ("flip_0.2", 0.0, 0.0, 0.2),
             ("flip_0.5", 0.0, 0.0, 0.5)]
    results = {}
    for label, nsig, np_null, np_flip in conds:
        sr = []
        for s in range(n_seeds):
            env = _env(seed=s+1100); m = _model(); _train(env, m, n_steps)
            env_t = _env(seed=s+1200); env_t.set_seed(9999); state = env_t.reset()
            s_buf = np.empty((1, 1), dtype=np.float32); risks = []
            for _ in range(1000):
                s_buf[0, 0] = state[0]; st = torch.from_numpy(s_buf).to(DEVICE)
                a = m(st); an = a.detach().cpu().numpy().item()
                ns, risk, gr, _, _ = env_t.step(np.float32(an))
                gr += np.random.randn() * nsig
                if np.random.random() < np_null: gr = 0.0
                if np.random.random() < np_flip: gr *= -1
                gr_t = torch.tensor([[gr]], dtype=torch.float32, device=DEVICE)
                action_loss = 0.01 * (a**2).sum()
                opt_s = torch.optim.Adam(m.parameters(), lr=1e-4)
                opt_s.zero_grad(); action_loss.backward(retain_graph=True)
                torch.autograd.backward([a], [gr_t.view(1, 1)], retain_graph=True)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt_s.step()
                risks.append(risk); state = ns
            sr.append(float(np.mean(risks)))
        results[label] = {"risk": float(np.mean(sr)), "std": float(np.std(sr))}
        print(f"  {label}: risk={results[label]['risk']:.4f}±{results[label]['std']:.4f}")
    path = os.path.join(OUT, "information_deprivation.json")
    with open(path, "w") as f: json.dump(results, f, indent=2, default=_jdef)
    return results


# ===================================================================
# EXP 5: True vs False World (kept)
# ===================================================================

class BiasedObsEnv:
    def __init__(self, base_env, bias=0.3):
        self.base = base_env; self.bias = bias
    def reset(self):
        s = self.base.reset(); return np.array([s[0] - self.bias], dtype=np.float32)
    def step(self, action):
        ns, risk, gr, _, _ = self.base.step(action)
        return np.array([ns[0] - self.bias], dtype=np.float32), risk, gr, False, {}
    def set_seed(self, seed): self.base.set_seed(seed)


def run_truefalse(n_seeds=5, n_steps=3000):
    print(f"\n{'='*70}\nEXP 5: True vs False World\n{'='*70}")
    sr_t, sr_f = [], []
    for s in range(n_seeds):
        env_t = _env(seed=s+1300); m_t = _model(); _train(env_t, m_t, n_steps)
        ro_t, _ = _rollout(env_t, m_t, 500); sr_t.append(float(np.mean(ro_t)))
        base = _env(seed=s+1400); env_f = BiasedObsEnv(base, bias=0.3)
        m_f = _model(); _train(env_f, m_f, n_steps)
        ro_f, _ = _rollout(env_f, m_f, 500); sr_f.append(float(np.mean(ro_f)))
        print(f"  seed={s}: true={sr_t[-1]:.4f} false={sr_f[-1]:.4f}")
    res = {"true_risk": float(np.mean(sr_t)), "false_risk": float(np.mean(sr_f)),
           "pass": float(np.mean(sr_f)) < float(np.mean(sr_t)) * 2.0}
    print(f"  true={res['true_risk']:.4f} false={res['false_risk']:.4f} pass={res['pass']}")
    path = os.path.join(OUT, "true_vs_false_world.json")
    with open(path, "w") as f: json.dump(res, f, indent=2, default=_jdef)
    return res


# ===================================================================
# Main
# ===================================================================



def main():
    p = argparse.ArgumentParser("Layer 4")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--skip", type=str, nargs="*", default=[])
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ns, nst = args.seeds, args.steps

    all_results = {}
    for exp, fn in [("delay", run_delay), ("minimal", run_minimal),
                     ("complexity", run_complexity), ("deprivation", run_deprivation),
                     ("truefalse", run_truefalse), ("dual_path", run_dual_path)]:
        if exp in args.skip: continue
        all_results[exp] = fn(ns, nst)

    n_pass = sum(1 for v in all_results.values() if v.get("pass", False))
    print(f"\n{'='*70}\nLayer 4: {n_pass}/{len(all_results)} passed\n{'='*70}")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=_jdef)


if __name__ == "__main__":
    main()
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


