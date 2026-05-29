"""Layer 3 Existence-Driven: loss = risk + λ||a||².

Key shift: no predictor. Gradient from risk → action injected via
env's analytic grad_risk. Models "survival pressure" directly.

Usage:
  uv run python core_mvp/layers/layer3_online.py
  uv run python core_mvp/layers/layer3_online.py --quick
"""

from __future__ import annotations

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, _root)

from core_mvp.core.core_env import QuadraticDriftEnv
from core_mvp.core.core_models import ExistenceDrivenModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
OUT = os.path.join(os.path.dirname(_here), "results", "layer3")

TOTAL = 10000
POST_BUF = 2000
INTERV = 500
RECOV = 500
WINDOW = 200
N_CHECKS = 5
ACT_TH = 0.01
RISK_TH = 0.1


def _env(seed=0):
    return QuadraticDriftEnv(drift=0.02, action_scale=0.5, dt=0.2, seed=seed)


def _model():
    return ExistenceDrivenModel(state_dim=1, hidden_dim=64, action_dim=1)


def _jdef(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


# ===================================================================
# Train with interventions (existence-driven)
# ===================================================================

def train_with_interventions(env, model):
    model.to(DEVICE); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    s = env.reset()
    s_buf = np.empty((1, 1), dtype=np.float32)
    acts, risks = [], []
    interv_m = {"grad_blocked": {"risks": [], "acts": []},
                 "grad_noise": {"risks": [], "acts": []}}
    t_stable = None
    phase = "normal"
    interv_done = False

    for t in range(TOTAL):
        s_buf[0, 0] = s[0]
        st = torch.from_numpy(s_buf).to(DEVICE)
        a = model(st)
        an = a.detach().cpu().numpy().item()
        ns, risk, gr, _, _ = env.step(np.float32(an))
        acts.append(abs(an)); risks.append(risk)

        risk_t = torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)
        gr_t = torch.tensor([[gr]], dtype=torch.float32, device=DEVICE)

        if phase == "grad_blocked":
            # Blocked: only action penalty, NO risk gradient injected
            action_loss = 0.01 * (a**2).sum()
            opt.zero_grad(); action_loss.backward()
            interv_m["grad_blocked"]["risks"].append(risk)
            interv_m["grad_blocked"]["acts"].append(abs(an))
        elif phase == "grad_noise":
            action_loss = 0.01 * (a**2).sum()
            opt.zero_grad(); action_loss.backward(retain_graph=True)
            torch.autograd.backward([a], [gr_t.view(1, 1) + torch.randn(1, 1, device=DEVICE) * 0.5], retain_graph=True)
            interv_m["grad_noise"]["risks"].append(risk)
            interv_m["grad_noise"]["acts"].append(abs(an))
        else:
            action_loss = 0.01 * (a**2).sum()
            opt.zero_grad(); action_loss.backward(retain_graph=True)
            torch.autograd.backward([a], [gr_t.view(1, 1)], retain_graph=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Phase transitions
        if t_stable is not None and not interv_done and t >= t_stable + POST_BUF:
            if t < t_stable + POST_BUF + INTERV: phase = "grad_blocked"
            elif t < t_stable + POST_BUF + INTERV + RECOV: phase = "normal"
            elif t < t_stable + POST_BUF + INTERV + RECOV + INTERV: phase = "grad_noise"
            else: phase = "normal"; interv_done = True

        # Detect t_stable
        if t_stable is None and t >= WINDOW * N_CHECKS:
            aa = np.array(acts); ra = np.array(risks)
            early_r = np.mean(ra[:500])
            for ti in range(WINDOW * N_CHECKS, t - WINDOW, WINDOW):
                wa = np.mean(aa[ti - WINDOW:ti])
                if wa > ACT_TH:
                    ok = all(np.mean(aa[ti + (k-1)*WINDOW : ti + k*WINDOW]) >= ACT_TH for k in range(1, N_CHECKS+1) if ti + k*WINDOW < len(aa))
                    if ok and len([k for k in range(1, N_CHECKS+1)]) == N_CHECKS and np.mean(ra[ti:ti + N_CHECKS * WINDOW]) < early_r * RISK_TH:
                        t_stable = ti; break

        s = ns

    model.eval()
    return {"actions": np.array(acts), "risks": np.array(risks), "t_stable": t_stable, "interv": interv_m}


# ===================================================================
# Evaluation
# ===================================================================

def _rollout(env, model, steps):
    env.set_seed(9999); s = env.reset()
    s_buf = np.empty((1, 1), dtype=np.float32); R = []
    for _ in range(steps):
        s_buf[0, 0] = s[0]; st = torch.from_numpy(s_buf).to(DEVICE)
        with torch.no_grad(): a = model(st); an = a.cpu().numpy().item()
        ns, risk, _, _, _ = env.step(np.float32(an)); R.append(risk); s = ns
    return np.array(R)


def _rollout_open(env, steps):
    env.set_seed(9999); _ = env.reset(); R = []
    for _ in range(steps): _, risk, _, _, _ = env.step(np.float32(0.0)); R.append(risk)
    return np.array(R)


def _rollout_noisy(env, model, steps, noise=0.2):
    env.set_seed(9999); s = env.reset()
    s_buf = np.empty((1, 1), dtype=np.float32); R = []
    for _ in range(steps):
        sn = s[0] + np.random.randn() * noise; s_buf[0, 0] = sn
        st = torch.from_numpy(s_buf).to(DEVICE)
        with torch.no_grad(): a = model(st); an = a.cpu().numpy().item()
        ns, risk, _, _, _ = env.step(np.float32(an)); R.append(risk); s = ns
    return np.array(R)


def _rollout_shuffle(env, model, steps):
    env.set_seed(9999); s = env.reset()
    s_buf = np.empty((1, 1), dtype=np.float32); R, buf = [], []
    for _ in range(steps):
        s_buf[0, 0] = s[0]; st = torch.from_numpy(s_buf).to(DEVICE)
        with torch.no_grad(): a = model(st); an = a.cpu().numpy().item()
        ns, risk, _, _, _ = env.step(np.float32(an)); R.append(risk); buf.append(s.copy())
        s = buf[np.random.randint(0, len(buf))].copy() if len(buf) > 10 and np.random.random() < 0.1 else ns
    return np.array(R)


# ===================================================================
# Sub-experiments
# ===================================================================

def run_l3_2(t_stars):
    valid = [t for t in t_stars if t is not None]
    if len(valid) < 2: return {"pass": False}
    m = float(np.mean(valid)); s = float(np.std(valid))
    return {"pass": len(valid) == len(t_stars) and s < m * 0.3, "mean": m, "std": s}


def run_l3_3_7(interv):
    gb = interv["grad_blocked"]
    if len(gb["risks"]) < 10: return {"pass": False}
    baseline = float(np.mean(gb["risks"][:50])) if len(gb["risks"]) >= 50 else float(np.mean(gb["risks"]))
    mid = float(np.mean(gb["risks"][len(gb["risks"])//2:]))
    return {"pass": mid > baseline * 1.3, "baseline": baseline, "mid": mid}


def run_l3_4(interv):
    gn = interv["grad_noise"]
    if len(gn["risks"]) < 10: return {"pass": False}
    baseline = float(np.mean(gn["risks"][:50])) if len(gn["risks"]) >= 50 else float(np.mean(gn["risks"]))
    mid = float(np.mean(gn["risks"][len(gn["risks"])//2:]))
    return {"pass": mid > baseline * 1.3, "baseline": baseline, "mid": mid}


def run_l3_5(env_fn, steps=2000):
    env = env_fn(9997)
    w = torch.tensor([0.0], device=DEVICE, requires_grad=True)
    b = torch.tensor([0.0], device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.01)
    s = env.reset(); log = []
    for _ in range(steps):
        target = 0.5 - s[0]
        st = torch.tensor([target], dtype=torch.float32, device=DEVICE)
        a = w * st + b; an = a.detach().cpu().numpy().item()
        ns, risk, _, _, _ = env.step(np.float32(an))
        loss = torch.tensor(risk, dtype=torch.float32, device=DEVICE) + 0.01 * (a**2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        log.append({"risk": risk, "w": w.item(), "b": b.item()}); s = ns
    env2 = env_fn(9998); ro = _rollout_open(env2, 500)
    return {"minimal_risk": float(np.mean([l["risk"] for l in log[-500:]])),
            "open_risk": float(np.mean(ro)), "w": w.item(), "b": b.item(),
            "pass": float(np.mean([l["risk"] for l in log[-500:]])) < float(np.mean(ro)) * 0.5}


def run_l3_6(model, env_fn):
    env = env_fn(9995); Rn = _rollout(env, model, 500)
    env2 = env_fn(9994); Rs = _rollout_shuffle(env2, model, 500)
    nr = float(np.mean(Rn)); sr = float(np.mean(Rs))
    return {"normal": nr, "shuffled": sr, "pass": sr > nr * 1.5}


def run_l3_8(model):
    model.to(DEVICE); model.eval()
    env = _env(9996); s = env.reset()
    s_buf = np.empty((1, 1), dtype=np.float32); gains = []
    for _ in range(500):
        s_buf[0, 0] = s[0]; st = torch.from_numpy(s_buf).to(DEVICE)
        a = model(st)
        ns, risk, gr, _, _ = env.step(a.detach().cpu().numpy().item())
        if abs(gr) > 1e-8: gains.append(abs(gr))
        s = ns
    if len(gains) < 100: return {"pass": False}
    return {"pass": float(np.std(gains[-250:])) < 0.2, "late_std": float(np.std(gains[-250:]))}


def run_l3_9(env_fn, n_seeds):
    found = sum(1 for s in range(n_seeds)
                if train_with_interventions(env_fn(s), _model())["t_stable"] is not None)
    return {"pass": found / n_seeds >= 0.8, "rate": found / n_seeds}


# ===================================================================
# Main
# ===================================================================

def run_layer3(n_seeds=8):
    os.makedirs(OUT, exist_ok=True)
    env_fn = _env
    print(f"\n{'='*70}")
    print(f"Layer 3 Existence-Driven: {n_seeds}×{TOTAL} steps")
    print(f"{'='*70}\n")

    scores = {}; entries = []; t_stars = []
    for s in range(n_seeds):
        m = _model()
        traj = train_with_interventions(env_fn(s), m)
        entries.append({"model": m, "traj": traj, "seed": s}); t_stars.append(traj["t_stable"])
        aa = traj["actions"]
        print(f"  seed={s}: t*={traj['t_stable']}  |a|={np.mean(aa[-500:]):.4f}  r={np.mean(traj['risks'][-500:]):.4f}")

    stable = [e for e in entries if e["traj"]["t_stable"] is not None]
    print(f"  stable={len(stable)}/{n_seeds}")

    l3_2 = run_l3_2(t_stars)
    print(f"\nL3-2 t*: mean={l3_2.get('mean')} std={l3_2.get('std')} pass={l3_2['pass']}")
    scores["L3-2"] = l3_2["pass"]

    l3_9 = run_l3_9(env_fn, n_seeds)
    print(f"L3-9 self-org: rate={l3_9['rate']:.2f} pass={l3_9['pass']}")
    scores["L3-9"] = l3_9["pass"]

    if not stable: print("No stable seeds — aborting"); return

    l3_8 = run_l3_8(stable[0]["model"])
    print(f"L3-8 gain: std={l3_8.get('late_std','?')} pass={l3_8['pass']}")
    scores["L3-8"] = l3_8["pass"]

    e = stable[0]; traj = e["traj"]
    l3_3 = run_l3_3_7(traj["interv"])
    l3_4 = run_l3_4(traj["interv"])
    print(f"L3-3 grad_blocked: base={l3_3.get('baseline','?')} mid={l3_3.get('mid','?')} pass={l3_3['pass']}")
    print(f"L3-4 grad_noise: base={l3_4.get('baseline','?')} mid={l3_4.get('mid','?')} pass={l3_4['pass']}")
    scores["L3-3"] = l3_3["pass"]; scores["L3-4"] = l3_4["pass"]

    l3_5 = run_l3_5(env_fn, 2000)
    print(f"L3-5 minimal: risk={l3_5['minimal_risk']:.4f} open={l3_5['open_risk']:.4f} w={l3_5['w']:.4f} pass={l3_5['pass']}")
    scores["L3-5"] = l3_5["pass"]

    l3_6 = run_l3_6(stable[0]["model"], env_fn)
    print(f"L3-6 memory: normal={l3_6['normal']:.4f} shuff={l3_6['shuffled']:.4f} pass={l3_6['pass']}")
    scores["L3-6"] = l3_6["pass"]

    # L3-7: blocked model from scratch
    cr = []
    for s in range(min(4, n_seeds)):
        env_b = env_fn(s + 10000); m_b = _model()
        _train_blocked(env_b, m_b, TOTAL)
        Rb = _rollout(env_fn(s + 20000), m_b, 500); cr.append(float(np.mean(Rb)))
    Rn = _rollout(env_fn(30000), stable[0]["model"], 500)
    nr = float(np.mean(Rn)); br = float(np.mean(cr))
    l3_7 = {"pass": br > nr * 2.0, "normal_risk": nr, "blocked_risk": br}
    print(f"L3-7 blocked: normal={nr:.4f} blocked={br:.4f} pass={l3_7['pass']}")
    scores["L3-7"] = l3_7["pass"]

    n_pass = sum(scores.values()); n_total = len(scores)
    print(f"\n{'='*70}")
    print(f"Layer 3 Existence: {n_pass}/{n_total} passed")
    for k, v in sorted(scores.items()): print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  {'ALL_PASS' if n_pass >= 6 else 'FAIL'} (need ≥6/8)")
    print(f"{'='*70}")

    out = {"scores": {k: bool(v) for k, v in scores.items()}, "n_pass": n_pass, "n_total": n_total}
    with open(os.path.join(OUT, "existence_driven.json"), "w") as f:
        json.dump(out, f, indent=2, default=_jdef)
    return out


def _train_blocked(env, model, steps):
    model.to(DEVICE); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    s = env.reset(); s_buf = np.empty((1, 1), dtype=np.float32)
    for _ in range(steps):
        s_buf[0, 0] = s[0]; st = torch.from_numpy(s_buf).to(DEVICE)
        a = model(st)
        ns, risk, _, _, _ = env.step(a.detach().cpu().numpy().item())
        loss = 0.01 * (a**2).sum()  # only action penalty
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); s = ns
    model.eval()


def main():
    p = argparse.ArgumentParser("Layer 3 Existence")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    ns = args.seeds if not args.quick else 2
    if args.quick: global TOTAL, POST_BUF, INTERV, RECOV; TOTAL = 1500; POST_BUF = 500; INTERV = 200; RECOV = 200
    run_layer3(n_seeds=ns)


if __name__ == "__main__":
    main()
