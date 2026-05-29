"""Layer 3 Final: Emergence verification on ContinuousDriftLevelEnv.

Environment forces control — drift pushes water down, agent must pump up.
Interventions run on stabilized policy (after t_stable detection).

Usage:
  uv run python core_mvp/layers/layer3_final.py
  uv run python core_mvp/layers/layer3_final.py --quick
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

from core_mvp.core.core_env import ContinuousDriftLevelEnv
from core_mvp.core.core_models import ClosedLoopModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
OUT = os.path.join(os.path.dirname(_here), "results", "layer3")

TOTAL_STEPS = 10000
STABLE_WIN = 200
STABLE_CHECKS = 5
POST_BUFFER = 2000
ACT_TH = 0.01


def _make_env(seed=0):
    return ContinuousDriftLevelEnv(drift=0.02, action_scale=1.0, noise_std=0.01, seed=seed)


def _make_model():
    return ClosedLoopModel(state_dim=1, hidden_dim=128, action_dim=1)


def _jdef(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


# ===================================================================
# Phase 1: Train + t_stable detection
# ===================================================================

def train_and_detect(env, model, steps=TOTAL_STEPS):
    model.to(DEVICE); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    s = env.reset()
    s_buf = np.empty((1, 1), dtype=np.float32)
    actions, risks = [], []
    for t in range(steps):
        s_buf[0, 0] = s[0]
        st = torch.from_numpy(s_buf).to(DEVICE)
        a, h, rp = model(st)
        an = a.detach().cpu().numpy().item()
        ns, risk, done, _ = env.step(np.float32(an))
        loss = F.mse_loss(rp, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + 0.05 * (a**2).sum()
        if torch.norm(a) < 0.005: loss = loss + 0.5 * (0.005 - torch.norm(a))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        actions.append(abs(an)); risks.append(risk)
        s = ns
        if done:
            s = env.reset()
    model.eval()
    aa = np.array(actions); ra = np.array(risks)
    early_r = np.mean(ra[:500])
    t_stable = None
    for t in range(STABLE_WIN * STABLE_CHECKS, steps - STABLE_WIN, STABLE_WIN):
        w_act = np.mean(aa[t - STABLE_WIN:t])
        if w_act > ACT_TH:
            ok = True
            for k in range(1, STABLE_CHECKS + 1):
                if t + k * STABLE_WIN >= steps: ok = False; break
                if np.mean(aa[t + (k-1)*STABLE_WIN : t + k*STABLE_WIN]) < ACT_TH: ok = False; break
            if ok and np.mean(ra[t:t + STABLE_CHECKS * STABLE_WIN]) < early_r * 0.7:
                t_stable = t; break
    return {"actions": aa, "risks": ra, "t_stable": t_stable}


# ===================================================================
# Rollout (3 modes)
# ===================================================================

def _rollout(env, model, steps, mode="normal"):
    env.set_seed(9999); s = env.reset()
    s_buf = np.empty((1, 1), dtype=np.float32)
    R, A = [], []
    for _ in range(steps):
        s_buf[0, 0] = s[0]; st = torch.from_numpy(s_buf).to(DEVICE)
        with torch.no_grad():
            h = model.encoder(st)
            if mode == "grad_disabled":
                a = model.actor(h)
            elif mode == "grad_masked":
                a = model.actor(h).detach()
            else:
                a, _, _ = model(st)
            an = a.cpu().numpy().item()
        ns, risk, _, _ = env.step(np.float32(an))
        R.append(risk); A.append(an); s = ns
    return np.array(R), np.array(A)


def _rollout_noisy(env, model, steps, noise_std=0.2):
    env.set_seed(9999); s = env.reset()
    s_buf = np.empty((1, 1), dtype=np.float32)
    R = []
    for _ in range(steps):
        s_noisy = np.clip(s[0] + np.random.randn() * noise_std, 0.0, 1.0)
        s_buf[0, 0] = s_noisy; st = torch.from_numpy(s_buf).to(DEVICE)
        with torch.no_grad():
            a, _, _ = model(st); an = a.cpu().numpy().item()
        ns, risk, _, _ = env.step(np.float32(an))
        R.append(risk); s = ns
    return np.array(R)


def _rollout_shuffle(env, model, steps, jump_prob=0.1, buf_size=10):
    env.set_seed(9999); s = env.reset()
    s_buf = np.empty((1, 1), dtype=np.float32)
    R, buf = [], []
    for _ in range(steps):
        s_buf[0, 0] = s[0]; st = torch.from_numpy(s_buf).to(DEVICE)
        with torch.no_grad():
            a, _, _ = model(st); an = a.cpu().numpy().item()
        ns, risk, _, _ = env.step(np.float32(an))
        R.append(risk); buf.append(s.copy())
        if len(buf) > buf_size and np.random.random() < jump_prob:
            s = buf[np.random.randint(0, len(buf))].copy()
        else:
            s = ns
    return np.array(R)


def _rollout_open(env, steps):
    env.set_seed(9999); _ = env.reset(); R = []
    for _ in range(steps): _, risk, _, _ = env.step(np.float32(0.0)); R.append(risk)
    return np.array(R)


# ===================================================================
# L3-2: t* consistency
# ===================================================================

def run_l3_2(t_stars):
    valid = [t for t in t_stars if t is not None]
    if not valid: return {"pass": False, "mean": None, "std": None}
    m = float(np.mean(valid)); s = float(np.std(valid)) if len(valid) >= 2 else 0.0
    return {"pass": len(valid) == len(t_stars) and s < m * 0.3, "mean": m, "std": s}


# ===================================================================
# Interventions
# ===================================================================

def run_l3_3(env, model):
    """Gradient dependency: compare 3 inference modes."""
    R_n, _ = _rollout(env, model, 500, "normal")
    R_m, _ = _rollout(env, model, 500, "grad_masked")
    R_d, _ = _rollout(env, model, 500, "grad_disabled")
    nr = float(np.mean(R_n)); mr = float(np.mean(R_m)); dr = float(np.mean(R_d))
    return {"normal": nr, "masked": mr, "disabled": dr,
            "pass": mr > nr * 1.5 or dr > nr * 1.5}


def run_l3_4(env, model):
    """Information: clean vs noisy observation."""
    R_n, _ = _rollout(env, model, 500, "normal")
    R_ny = _rollout_noisy(env, model, 500, 0.2)
    return {"clean": float(np.mean(R_n)), "noisy": float(np.mean(R_ny)),
            "pass": float(np.mean(R_ny)) > float(np.mean(R_n)) * 1.5}


def run_l3_6(env, model):
    """Memory vs loop: temporal shuffle."""
    R_n, _ = _rollout(env, model, 500, "normal")
    R_s = _rollout_shuffle(env, model, 500)
    return {"normal": float(np.mean(R_n)), "shuffled": float(np.mean(R_s)),
            "pass": float(np.mean(R_s)) > float(np.mean(R_n)) * 1.5}


def run_l3_7(env, model):
    """Counterfactual: same as L3-3 with emphasis on masked mode."""
    R_n, _ = _rollout(env, model, 500, "normal")
    R_m, _ = _rollout(env, model, 500, "grad_masked")
    return {"normal": float(np.mean(R_n)), "masked": float(np.mean(R_m)),
            "pass": float(np.mean(R_m)) > float(np.mean(R_n)) * 1.5}


def run_l3_5(env, steps=2000):
    """Minimal linear model: a = w*s + b."""
    w = torch.tensor([0.0], device=DEVICE, requires_grad=True)
    b = torch.tensor([0.0], device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.01)
    s = env.reset(); log = []
    for _ in range(steps):
        st = torch.tensor([s[0]], dtype=torch.float32, device=DEVICE)
        a = w * st + b; an = a.detach().cpu().numpy().item()
        ns, risk, _, _ = env.step(np.float32(an))
        loss = torch.tensor(risk, dtype=torch.float32, device=DEVICE) + 0.05 * (a**2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        log.append({"action": an, "risk": risk, "w": w.item(), "b": b.item()})
        s = ns
    ro = _rollout_open(env, 500)
    open_r = float(np.mean(ro)); min_r = float(np.mean([l["risk"] for l in log[-500:]]))
    return {"minimal_risk": min_r, "open_risk": open_r, "final_w": w.item(), "final_b": b.item(),
            "pass": min_r < open_r * 0.8}


def run_l3_8(model, t_stable):
    """Gain: compute ||∂risk_pred/∂action|| variance in stable phase."""
    if t_stable is None: return {"pass": False}
    model.to(DEVICE); model.eval()
    start = t_stable + POST_BUFFER
    gains = []
    env = _make_env(seed=9998); s = env.reset()
    s_buf = np.empty((1, 1), dtype=np.float32)
    for _ in range(1000):
        s_buf[0, 0] = s[0]; st = torch.from_numpy(s_buf).to(DEVICE)
        h = model.encoder(st); a = model.actor(h).detach().requires_grad_(True)
        rp = model.predictor(torch.cat([h, a], dim=-1))
        model.zero_grad(); rp.sum().backward(retain_graph=True)
        g = a.grad
        if g is not None: gains.append(float(torch.norm(g).item()))
        an = a.detach().cpu().numpy().item()
        ns, risk, _, _ = env.step(np.float32(an)); s = ns
        model.zero_grad()
    if len(gains) < 100: return {"pass": False}
    late = gains[-500:]; lstd = float(np.std(late))
    return {"late_mean": float(np.mean(late)), "late_std": lstd, "pass": lstd < 0.2}


def run_l3_9(env_fn, n_seeds):
    t_found = 0
    for s in range(n_seeds):
        env = env_fn(s); m = _make_model()
        traj = train_and_detect(env, m, TOTAL_STEPS)
        if traj["t_stable"] is not None: t_found += 1
    er = t_found / n_seeds
    return {"emergence_rate": er, "pass": er > 0.8}


# ===================================================================
# Main
# ===================================================================

def run_layer3(n_seeds=8):
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Layer 3 Final: ContinuousDriftLevelEnv  {n_seeds}×{TOTAL_STEPS}")
    print(f"{'='*70}\n")

    env_fn = _make_env
    scores = {}

    # Phase 1: train all seeds
    print("=== Phase 1: Train + t_stable ===")
    entries = []
    t_stars = []
    for s in range(n_seeds):
        env = env_fn(s); m = _make_model()
        traj = train_and_detect(env, m, TOTAL_STEPS)
        entries.append({"model": m, "env": env_fn(s + 10000), "traj": traj, "seed": s})
        t_stars.append(traj["t_stable"])
        print(f"  seed={s}: t*={traj['t_stable']}  |a|={np.mean(traj['actions'][-500:]):.4f}  r={np.mean(traj['risks'][-500:]):.4f}")

    stable = [e for e in entries if e["traj"]["t_stable"] is not None]
    print(f"  stable={len(stable)}/{n_seeds}")

    # L3-2
    l3_2 = run_l3_2(t_stars)
    print(f"\n=== L3-2: t* Consistency ===")
    print(f"  mean={l3_2['mean']} std={l3_2['std']} pass={l3_2['pass']}")
    scores["L3-2"] = l3_2["pass"]

    # L3-9
    l3_9 = run_l3_9(env_fn, n_seeds)
    print(f"\n=== L3-9: Self-Organization ===")
    print(f"  emergence_rate={l3_9['emergence_rate']:.2f} pass={l3_9['pass']}")
    scores["L3-9"] = l3_9["pass"]

    if not stable:
        print("No stable seeds — aborting interventions")
        return

    # L3-8 on first stable
    l3_8 = run_l3_8(stable[0]["model"], stable[0]["traj"]["t_stable"])
    print(f"\n=== L3-8: Gain Analysis ===")
    print(f"  gain_std={l3_8.get('late_std', '?')} pass={l3_8['pass']}")
    scores["L3-8"] = l3_8["pass"]

    # Interventions on stable seeds
    for e in stable[:3]:
        seed = e["seed"]; m = e["model"]; env = e["env"]
        l3_3 = run_l3_3(env, m)
        l3_4 = run_l3_4(env, m)
        l3_6 = run_l3_6(env, m)
        l3_7 = run_l3_7(env, m)
        print(f"\n=== Interventions seed={seed} ===")
        print(f"  L3-3: n={l3_3['normal']:.4f} m={l3_3['masked']:.4f} d={l3_3['disabled']:.4f} → {l3_3['pass']}")
        print(f"  L3-4: clean={l3_4['clean']:.4f} noisy={l3_4['noisy']:.4f} → {l3_4['pass']}")
        print(f"  L3-6: normal={l3_6['normal']:.4f} shuffled={l3_6['shuffled']:.4f} → {l3_6['pass']}")
        print(f"  L3-7: normal={l3_7['normal']:.4f} masked={l3_7['masked']:.4f} → {l3_7['pass']}")

    scores["L3-3"] = l3_3["pass"]
    scores["L3-4"] = l3_4["pass"]
    scores["L3-6"] = l3_6["pass"]
    scores["L3-7"] = l3_7["pass"]

    # L3-5
    print(f"\n=== L3-5: Minimal Linear Model ===")
    env5 = env_fn(9999)
    l3_5 = run_l3_5(env5, 2000)
    print(f"  min_risk={l3_5['minimal_risk']:.4f} open={l3_5['open_risk']:.4f} w={l3_5['final_w']:.4f} b={l3_5['final_b']:.4f} pass={l3_5['pass']}")
    scores["L3-5"] = l3_5["pass"]

    n_pass = sum(scores.values()); n_total = len(scores)
    print(f"\n{'='*70}")
    print(f"Layer 3 Final: {n_pass}/{n_total} passed")
    for k, v in sorted(scores.items()): print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  {'ALL_PASS' if n_pass >= 7 else 'FAIL'} (need ≥7/8)")
    print(f"{'='*70}")

    out = {"scores": {k: bool(v) for k, v in scores.items()},
           "n_pass": n_pass, "n_total": n_total, "stable": len(stable)}
    with open(os.path.join(OUT, "emergence_final.json"), "w") as f:
        json.dump(out, f, indent=2, default=_jdef)
    return out


def main():
    p = argparse.ArgumentParser("Layer 3 Final")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    ns = args.seeds if not args.quick else 3
    if args.quick:
        global TOTAL_STEPS, POST_BUFFER, STABLE_WIN, STABLE_CHECKS
        TOTAL_STEPS = 2000; POST_BUFFER = 500; STABLE_WIN = 100; STABLE_CHECKS = 3
    run_layer3(n_seeds=ns)


if __name__ == "__main__":
    main()
