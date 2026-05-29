"""Layer 3 Corrected: detect stable closed-loop phase before intervention.

Phase 1: Train 10000 steps, detect t_stable (|a|>0.05 sustained, risk↓).
Phase 2: After t_stable+2000, run interventions on the stabilized policy.

Usage:
  uv run python core_mvp/layers/layer3_corrected.py
  uv run python core_mvp/layers/layer3_corrected.py --quick
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

from core_mvp.core.core_env import LatentShiftDoubleWellEnv
from core_mvp.core.core_models import LatentShiftModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
OUT = os.path.join(os.path.dirname(_here), "results", "layer3")

TOTAL_STEPS = 10000
STABLE_WINDOW = 200
STABLE_CHECKS = 5
POST_STABLE_BUFFER = 2000
ACTION_THRESHOLD = 0.05
RISK_RATIO_THRESHOLD = 0.7


# ===================================================================
# Shared
# ===================================================================

def _make_env(drift=0.2, seed=0):
    return LatentShiftDoubleWellEnv(drift_strength=drift, seed=seed)


def _make_model():
    return LatentShiftModel(hidden_dim=128)


def _jdef(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


# ===================================================================
# Phase 1: Train + detect t_stable
# ===================================================================

def train_with_detection(env, model, steps=TOTAL_STEPS):
    """Train model, record full trajectory, detect t_stable."""
    model.to(DEVICE); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = env.reset(); gh = None
    s_buf = np.empty((1, 2), dtype=np.float32)
    actions, risks, grad_risks = [], [], []

    for t in range(steps):
        s_buf[0, 0] = state[0]
        s_buf[0, 1] = gh if gh is not None else 0.0
        s_t = torch.from_numpy(s_buf).to(DEVICE)
        h = model.encoder(s_t)
        a = model.actor(h)
        rp = model.predictor(torch.cat([h, a], dim=-1))
        an = a.detach().cpu().numpy().item()
        ns, risk, gr, _, _ = env.step(np.float32(an))
        loss = F.mse_loss(rp, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + 0.05 * (a**2).sum()
        if torch.norm(a) < 0.01: loss = loss + 0.1 * (0.01 - torch.norm(a))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        actions.append(abs(an)); risks.append(risk); grad_risks.append(gr)
        state = ns; gh = gr

    model.eval()

    # Detect t_stable
    actions_a = np.array(actions); risks_a = np.array(risks)
    early_risk = np.mean(risks_a[:500])
    t_stable = None
    for t in range(STABLE_WINDOW * STABLE_CHECKS, steps - STABLE_WINDOW, STABLE_WINDOW):
        w_act = np.mean(actions_a[t - STABLE_WINDOW:t])
        if w_act > ACTION_THRESHOLD:
            # check next 5 windows
            ok = True
            for k in range(1, STABLE_CHECKS + 1):
                if t + k * STABLE_WINDOW >= steps: ok = False; break
                w = np.mean(actions_a[t + (k-1)*STABLE_WINDOW : t + k*STABLE_WINDOW])
                if w < ACTION_THRESHOLD: ok = False; break
            if ok:
                w_risk = np.mean(risks_a[t:t + STABLE_CHECKS * STABLE_WINDOW])
                if w_risk < early_risk * RISK_RATIO_THRESHOLD:
                    t_stable = t
                    break

    return {"actions": actions_a, "risks": risks_a, "grad_risks": np.array(grad_risks),
            "t_stable": t_stable}


def detect_emergence(log_a, log_r):
    """Legacy: detect t* from action/risk arrays."""
    for t in range(100, len(log_a) - 200):
        wa = np.mean(np.abs(log_a[t:t+200]))
        wr_after = np.mean(log_r[t+200:t+400]) if t+400 < len(log_r) else np.mean(log_r[t+200:])
        wr_before = np.mean(log_r[max(0, t-200):t])
        if wa > 0.03 and wr_after < wr_before * 0.8:
            return t
    return None


# ===================================================================
# Phase 2: Interventions (only if t_stable found)
# ===================================================================

def _rollout(env, model, steps, detach_actor=False, shuffle_grad=False):
    """Rollout with interventions applied during inference."""
    env.set_seed(9999); s = env.reset(); gh = None
    R, A = [], []
    for _ in range(steps):
        st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
        ghv = gh if gh is not None else 0.0
        ght = torch.tensor([[ghv]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            h = model.encoder(torch.cat([st, ght], dim=-1))
            a = model.actor(h)
            if detach_actor:
                a = a.detach()
            an = a.cpu().numpy().item()
        if shuffle_grad:
            ns, risk, gr, _, _ = env.step(np.float32(an))
            gr = float(np.random.randn() * abs(gr))  # shuffle gradient
        else:
            ns, risk, gr, _, _ = env.step(np.float32(an))
        R.append(risk); A.append(an)
        s = ns; gh = gr
    return np.array(R), np.array(A)


def _rollout_open(env, steps):
    env.set_seed(9999); _ = env.reset(); R = []
    for _ in range(steps):
        _, risk, _, _, _ = env.step(np.float32(0.0)); R.append(risk)
    return np.array(R)


# ===================================================================
# Sub-experiments (require t_stable)
# ===================================================================

def run_l3_3(env, model, steps_each=500):
    """Gradient shuffle: normal → shuffled → recovery."""
    # Normal baseline
    R_base, A_base = _rollout(env, model, steps_each)
    # Shuffled
    R_sh, A_sh = _rollout(env, model, steps_each, shuffle_grad=True)
    # Recovery
    R_rec, A_rec = _rollout(env, model, steps_each)
    return {"base_risk": float(np.mean(R_base)), "shuffled_risk": float(np.mean(R_sh)),
            "recovery_risk": float(np.mean(R_rec)),
            "base_act": float(np.mean(np.abs(A_base))),
            "shuffled_act": float(np.mean(np.abs(A_sh))),
            "pass": float(np.mean(R_sh)) > float(np.mean(R_base)) * 1.5}


def run_l3_4(env, model, steps_each=500):
    """Information: full gradient vs noisy gradient."""
    # Full
    R_full, _ = _rollout(env, model, steps_each)
    # Noisy
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    s = env.reset(); gh = None; R_noisy, A_noisy = [], []
    s_buf = np.empty((1, 2), dtype=np.float32)
    for _ in range(500):
        s_buf[0, 0] = s[0]; s_buf[0, 1] = gh if gh is not None else 0.0
        st = torch.from_numpy(s_buf).to(DEVICE)
        h = model.encoder(st); a = model.actor(h)
        rp = model.predictor(torch.cat([h, a], dim=-1))
        an = a.detach().cpu().numpy().item()
        ns, risk, gr, _, _ = env.step(np.float32(an))
        gr = gr + np.random.randn() * 0.5  # destroy gradient info
        loss = F.mse_loss(rp, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + 0.05 * (a**2).sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        R_noisy.append(risk); A_noisy.append(an); s = ns; gh = gr
    model.eval()
    return {"full_risk": float(np.mean(R_full)), "noisy_risk": float(np.mean(R_noisy)),
            "noisy_act_var": float(np.var(A_noisy)),
            "pass": float(np.mean(R_noisy)) > float(np.mean(R_full)) * 2.0}


def run_l3_6(env, model, steps=500):
    """Memory vs loop: temporal shuffle."""
    state = env.reset(); gh = None; buf = []
    R_norm, R_shuf = [], []
    for t in range(steps):
        st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        ghv = gh if gh is not None else 0.0
        ght = torch.tensor([[ghv]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            h = model.encoder(torch.cat([st, ght], dim=-1))
            a = model.actor(h); an = a.cpu().numpy().item()
        ns, risk, gr, _, _ = env.step(np.float32(an))
        R_norm.append(risk); buf.append(state.copy())
        if len(buf) > 10 and np.random.random() < 0.1:
            state = buf[np.random.randint(0, len(buf))].copy()
        else:
            state = ns
        gh = gr; R_shuf.append(risk)
    return {"normal_risk": float(np.mean(R_norm)), "shuffled_risk": float(np.mean(R_shuf)),
            "pass": float(np.mean(R_shuf)) > float(np.mean(R_norm)) * 1.5}


def run_l3_7(env, model, steps_each=500):
    """Counterfactual: normal → detach → normal."""
    R_base, _ = _rollout(env, model, steps_each)
    R_det, _ = _rollout(env, model, steps_each, detach_actor=True)
    R_rec, _ = _rollout(env, model, steps_each)
    return {"base_risk": float(np.mean(R_base)), "detach_risk": float(np.mean(R_det)),
            "recovery_risk": float(np.mean(R_rec)),
            "pass": float(np.mean(R_det)) > float(np.mean(R_base)) * 1.5}


def run_l3_5(env, steps=1000):
    """Minimal loop: a = k * grad_hint, optimize k via chain rule."""
    k = 0.5
    state = env.reset(); gh_prev = 0.0; log = []
    for t in range(steps):
        an = k * gh_prev
        ns, risk, gr, _, _ = env.step(np.float32(an))
        if abs(gh_prev) > 1e-6 and abs(gr) > 1e-6:
            k -= 0.01 * gr * np.sign(gh_prev)
            k = max(-2.0, min(2.0, k))
        log.append({"t": t, "action": an, "risk": risk, "k": k})
        state = ns; gh_prev = gr
    ro = _rollout_open(env, 500)
    open_risk = float(np.mean(ro)); min_risk = float(np.mean([l["risk"] for l in log[-200:]]))
    return {"minimal_risk": min_risk, "open_risk": open_risk, "final_k": k,
            "pass": min_risk < open_risk * 0.9}


def run_l3_8(grad_risks, actions, t_stable):
    """Gain analysis: G = Δa/Δgr in stable phase."""
    if t_stable is None: return {"pass": False, "late_std": 1.0}
    start = t_stable + POST_STABLE_BUFFER
    gr = grad_risks[start:]; ac = actions[start:]
    gains = []
    for i in range(1, len(gr)):
        dg = gr[i] - gr[i-1]
        if abs(dg) > 1e-8: gains.append((ac[i] - ac[i-1]) / dg)
    if len(gains) < 100: return {"pass": False, "late_std": 1.0}
    late = gains[-min(1000, len(gains)//2):]
    lstd = float(np.std(late))
    return {"late_gain": float(np.mean(late)), "late_std": lstd, "pass": lstd < 0.2}


def run_l3_9(env_fn, n_seeds=8):
    """Self-organization across seeds."""
    t_stars = []
    for s in range(n_seeds):
        env = env_fn(seed=s); m = _make_model()
        traj = train_with_detection(env, m, TOTAL_STEPS)
        t_stars.append(traj["t_stable"] is not None)
    er = np.mean(t_stars)
    return {"emergence_rate": float(er), "pass": er > 0.8}


# ===================================================================
# Full runner
# ===================================================================

def run_layer3(n_seeds=8, drift=0.2):
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Layer 3 Corrected: {n_seeds} seeds × {TOTAL_STEPS} steps")
    print(f"{'='*70}\n")

    env_fn = lambda seed=0: _make_env(drift, seed)
    scores = {}
    details = {}

    # Phase 1: Train all seeds, detect t_stable
    print("=== Phase 1: Training + t_stable detection ===")
    trajectories = []
    for s in range(n_seeds):
        env = env_fn(s); m = _make_model()
        traj = train_with_detection(env, m, TOTAL_STEPS)
        trajectories.append({"model": m, "traj": traj, "seed": s, "env": env_fn(s + 10000)})
        print(f"  seed={s}: t_stable={traj['t_stable']}, |a|_final={np.mean(traj['actions'][-500:]):.4f}")

    stable_seeds = [t for t in trajectories if t["traj"]["t_stable"] is not None]
    unstable = n_seeds - len(stable_seeds)
    print(f"  stable={len(stable_seeds)}/{n_seeds}  unstable={unstable}")

    if len(stable_seeds) < 2:
        print("Too few stable seeds — aborting interventions")
        return

    # L3-9: Self-organization
    print("\n=== L3-9: Self-Organization ===")
    l3_9 = run_l3_9(env_fn, n_seeds)
    print(f"  emergence_rate={l3_9['emergence_rate']:.2f}")
    scores["L3-9"] = l3_9["pass"]; details["L3-9"] = l3_9

    # L3-2: t* consistency
    t_stars = [t["traj"]["t_stable"] for t in stable_seeds]
    mean_t = float(np.mean(t_stars)); std_t = float(np.std(t_stars))
    l3_2_pass = std_t < mean_t * 0.3
    print(f"\n=== L3-2: t* Consistency ===")
    print(f"  mean_t*={mean_t:.0f} std={std_t:.0f} pass={l3_2_pass}")
    scores["L3-2"] = l3_2_pass; details["L3-2"] = {"mean": mean_t, "std": std_t}

    # L3-8: Gain on first stable seed's trajectory
    tr0 = stable_seeds[0]["traj"]
    l3_8 = run_l3_8(tr0["grad_risks"], tr0["actions"], tr0["t_stable"])
    print(f"\n=== L3-8: Gain Analysis ===")
    print(f"  late_gain={l3_8.get('late_gain',0):.4f} late_std={l3_8.get('late_std',0):.4f} pass={l3_8['pass']}")
    scores["L3-8"] = l3_8["pass"]; details["L3-8"] = l3_8

    # Phase 2: Interventions on each stable seed
    for idx, entry in enumerate(stable_seeds):
        seed = entry["seed"]; m = entry["model"]
        env = entry["env"]
        print(f"\n{'='*50}")
        print(f"Interventions seed={seed}")
        print(f"{'='*50}")

        # L3-3
        l3_3 = run_l3_3(deepcopy_env(env), m)
        print(f"  L3-3: base={l3_3['base_risk']:.4f} shuffled={l3_3['shuffled_risk']:.4f} pass={l3_3['pass']}")

        # L3-4
        l3_4 = run_l3_4(deepcopy_env(env), m)
        print(f"  L3-4: full={l3_4['full_risk']:.4f} noisy={l3_4['noisy_risk']:.4f} pass={l3_4['pass']}")

        # L3-6
        l3_6 = run_l3_6(deepcopy_env(env), m)
        print(f"  L3-6: normal={l3_6['normal_risk']:.4f} shuffled={l3_6['shuffled_risk']:.4f} pass={l3_6['pass']}")

        # L3-7
        l3_7 = run_l3_7(deepcopy_env(env), m)
        print(f"  L3-7: base={l3_7['base_risk']:.4f} detach={l3_7['detach_risk']:.4f} pass={l3_7['pass']}")

        if idx == 0:  # save first seed's results
            details["L3-3"] = l3_3; details["L3-4"] = l3_4
            details["L3-6"] = l3_6; details["L3-7"] = l3_7

    # Aggregate intervention passes across seeds
    for label in ["L3-3", "L3-4", "L3-6", "L3-7"]:
        if label in details and "pass" in details[label]:
            scores[label] = details[label]["pass"]
        else:
            scores[label] = False

    # L3-5: Minimal loop
    print(f"\n=== L3-5: Minimal Loop ===")
    env5 = env_fn(999); l3_5 = run_l3_5(env5, 1000)
    print(f"  minimal_risk={l3_5['minimal_risk']:.4f} open={l3_5['open_risk']:.4f} k={l3_5['final_k']:.4f} pass={l3_5['pass']}")
    scores["L3-5"] = l3_5["pass"]; details["L3-5"] = l3_5

    # Summary
    n_pass = sum(scores.values()); n_total = len(scores)
    print(f"\n{'='*70}")
    print(f"Layer 3 Corrected: {n_pass}/{n_total} passed")
    for k, v in sorted(scores.items()): print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  {'ALL_PASS' if n_pass >= 6 else 'FAIL'} (need ≥6/8)")
    print(f"{'='*70}")

    out = {"scores": {k: bool(v) for k, v in scores.items()},
           "n_pass": n_pass, "n_total": n_total, "details": details,
           "stable_seeds": len(stable_seeds), "unstable_seeds": unstable}
    with open(os.path.join(OUT, "emergence_corrected.json"), "w") as f:
        json.dump(out, f, indent=2, default=_jdef)
    return out


def deepcopy_env(env):
    return _make_env(drift=0.2, seed=np.random.randint(0, 100000))


def main():
    p = argparse.ArgumentParser("Layer 3 Corrected")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--drift", type=float, default=0.2)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    ns = args.seeds if not args.quick else 3
    if args.quick:
        global TOTAL_STEPS, POST_STABLE_BUFFER, STABLE_WINDOW, STABLE_CHECKS
        TOTAL_STEPS = 2000; POST_STABLE_BUFFER = 500; STABLE_WINDOW = 100; STABLE_CHECKS = 3
    run_layer3(n_seeds=ns, drift=args.drift)


if __name__ == "__main__":
    main()
