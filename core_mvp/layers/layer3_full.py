"""Layer 3: Emergence Verification — 9 sub-experiments.

Red lines (checked each step):
  ❗1 闭环是训练中涌现的  ❗2 依赖信息通路  ❗3 可被干预破坏
  ❗4 优于所有非闭环     ❗5 有稳定性贡献   ❗6 形成可定位(t*)
  ❗7 可压缩(线性模型)   ❗8 跨种子一致

Usage:
  uv run python core_mvp/layers/layer3_full.py
  uv run python core_mvp/layers/layer3_full.py --quick
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


# ===================================================================
# Shared (GPU-optimized)
# ===================================================================

def _train_one(env, model, steps, lr=1e-3, lam=0.05, record=False):
    """Train single model on single env — minimal overhead."""
    model.to(DEVICE); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    state = env.reset(); gh = None
    s_buf = np.empty((1, 2), dtype=np.float32)
    log = []
    for t in range(steps):
        s_buf[0, 0] = state[0]
        s_buf[0, 1] = gh if gh is not None else 0.0
        s_t = torch.from_numpy(s_buf).to(DEVICE)
        h = model.encoder(s_t)
        a = model.actor(h)
        rp = model.predictor(torch.cat([h, a], dim=-1))
        an = a.detach().cpu().numpy().item()
        ns, risk, gr, _, _ = env.step(np.float32(an))
        loss = F.mse_loss(rp, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + lam * (a**2).sum()
        if torch.norm(a) < 0.01: loss = loss + 0.1 * (0.01 - torch.norm(a))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if record:
            log.append({"t": t, "state": float(state[0]), "action": float(an),
                         "risk": float(risk), "grad_risk": float(gr),
                         "h_norm": float(torch.norm(h).item()),
                         "loss": float(loss.item())})
        state = ns; gh = gr
    model.eval()
    return log if record else None


def _make_env(drift=0.2, seed=0):
    return LatentShiftDoubleWellEnv(drift_strength=drift, seed=seed)


def _make_model():
    return LatentShiftModel(hidden_dim=128)


def _jdef(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


def _train_batch_shared(env_fn, n_models, steps, lr=1e-3, lam=0.05, blocked=False):
    """Train n_models on n_models envs — batched forward across all envs."""
    envs = [env_fn(s) for s in range(n_models)]
    models = [_make_model() for _ in range(n_models)]
    for m in models: m.to(DEVICE)
    opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]
    states_np = [e.reset() for e in envs]
    gh = [None] * n_models
    n = n_models
    s_batch = np.empty((n, 2), dtype=np.float32)
    for _ in range(steps):
        for i in range(n):
            s_batch[i, 0] = states_np[i][0]
            s_batch[i, 1] = gh[i] if gh[i] is not None else 0.0
        s_t = torch.from_numpy(s_batch).to(DEVICE)
        for i in range(n):
            h = models[i].encoder(s_t[i:i+1])
            a = models[i].actor(h)
            rp = models[i].predictor(torch.cat([h, a.detach() if blocked else a], dim=-1))
            an = a.detach().cpu().numpy().item()
            ns, risk, gr, done, _ = envs[i].step(np.float32(an))
            loss = F.mse_loss(rp, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + lam * (a**2).sum()
            if torch.norm(a) < 0.01: loss = loss + 0.1 * (0.01 - torch.norm(a))
            opts[i].zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(models[i].parameters(), 1.0); opts[i].step()
            states_np[i] = ns; gh[i] = gr
            if done: states_np[i] = envs[i].reset(); gh[i] = None
    for m in models: m.eval()
    return models


# ===================================================================
# L3-1: Training trajectory recording
# ===================================================================

def record_training_trajectory(env, model, steps=5000):
    model.to(DEVICE); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = env.reset(); gh = None; log = []
    for t in range(steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        ghv = gh if gh is not None else 0.0
        ght = torch.tensor([[ghv]], dtype=torch.float32, device=DEVICE)
        h = model.encoder(torch.cat([s_t, ght], dim=-1))
        action = model.actor(h)
        risk_pred = model.predictor(torch.cat([h, action], dim=-1))
        an = float(action.detach().cpu().numpy().squeeze(0).item())
        ns, risk, gr, _, _ = env.step(an)
        loss = F.mse_loss(risk_pred, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + 0.05 * (action**2).sum()
        if torch.norm(action) < 0.01: loss = loss + 0.1 * (0.01 - torch.norm(action))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        log.append({"t": t, "state": float(state[0]), "action": float(an),
                     "risk": float(risk), "grad_risk": float(gr),
                     "h_norm": float(torch.norm(h).item()),
                     "loss": float(loss.item())})
        state = ns; gh = gr
    model.eval()
    return log


# ===================================================================
# L3-2: Emergence detection (t*)
# ===================================================================

def detect_emergence(log, epsilon=0.01, K=50):
    actions = np.array([e["action"] for e in log])
    risks = np.array([e["risk"] for e in log])
    for t in range(len(actions) - 2*K):
        wa = np.abs(actions[t:t+K])
        if np.all(wa > epsilon):
            rb = np.mean(risks[max(0, t-50):t])
            ra = np.mean(risks[t+K:t+2*K])
            if ra < rb: return t
    return None


def run_l3_2(env_factory, n_seeds=8, steps=5000):
    t_stars = []
    for s in range(n_seeds):
        env = env_factory(seed=s); m = _make_model()
        log = record_training_trajectory(env, m, steps)
        t_star = detect_emergence(log)
        t_stars.append(t_star)
    valid = [t for t in t_stars if t is not None]
    er = len(valid) / n_seeds
    mean_t = float(np.mean(valid)) if valid else None
    std_t = float(np.std(valid)) if len(valid) >= 2 else 0.0
    return {"emergence_rate": er, "mean_t_star": mean_t, "std_t_star": std_t,
            "t_stars": [float(t) if t else None for t in t_stars],
            "pass": er > 0.8 and (std_t < mean_t * 0.3 if mean_t and std_t else True)}


# ===================================================================
# L3-3: Gradient causal chain (normal → shuffled → recovery)
# ===================================================================

def gradient_causal_chain_test(env, model, steps=3000):
    model.to(DEVICE); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = env.reset(); gh = None; log = []; phase = "normal"
    for t in range(steps):
        if t == 1000: phase = "shuffled"
        elif t == 2000: phase = "recovery"
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        ghv = gh if gh is not None else 0.0
        ght = torch.tensor([[ghv]], dtype=torch.float32, device=DEVICE)
        h = model.encoder(torch.cat([s_t, ght], dim=-1))
        action = model.actor(h)
        risk_pred = model.predictor(torch.cat([h, action], dim=-1))
        an = float(action.detach().cpu().numpy().squeeze(0).item())
        ns, risk, gr, _, _ = env.step(an)
        loss = F.mse_loss(risk_pred, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + 0.05 * (action**2).sum()
        opt.zero_grad(); loss.backward()
        if phase == "shuffled" and action.grad is not None:
            noise = torch.randn_like(action.grad)
            action.grad = noise / (torch.norm(noise) + 1e-8) * torch.norm(action.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        log.append({"t": t, "phase": phase, "action": float(an), "risk": float(risk)})
        state = ns; gh = gr
    model.eval()
    p1_act = np.mean(np.abs([l["action"] for l in log if l["phase"] == "normal"][-200:]))
    p2_act = np.mean(np.abs([l["action"] for l in log if l["phase"] == "shuffled"]))
    p1_risk = np.mean([l["risk"] for l in log if l["phase"] == "normal"][-200:])
    p2_risk = np.mean([l["risk"] for l in log if l["phase"] == "shuffled"])
    p3_risk = np.mean([l["risk"] for l in log if l["phase"] == "recovery"][-200:])
    return {"phase1_action": float(p1_act), "phase2_action": float(p2_act),
            "phase1_risk": float(p1_risk), "phase2_risk": float(p2_risk),
            "phase3_risk": float(p3_risk),
            "pass": bool(p2_act < p1_act * 0.5 and p2_risk > p1_risk * 1.3)}


# ===================================================================
# L3-4: Information utilization (full vs noisy gradient)
# ===================================================================

def information_test(env, model, steps=3000):
    results = {}
    for condition, noise_scale in [("full", 0.0), ("noisy", 0.5)]:
        m = LatentShiftModel(hidden_dim=128); m.to(DEVICE); m.train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        state = env.reset(); gh = None; alog, rlog = [], []
        for _ in range(steps):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
            ghv = gh if gh is not None else 0.0
            ght = torch.tensor([[ghv]], dtype=torch.float32, device=DEVICE)
            h = m.encoder(torch.cat([s_t, ght], dim=-1))
            action = m.actor(h)
            risk_pred = m.predictor(torch.cat([h, action], dim=-1))
            an = float(action.detach().cpu().numpy().squeeze(0).item())
            ns, risk, gr, _, _ = env.step(an)
            if condition == "noisy": gr += np.random.randn() * noise_scale
            loss = F.mse_loss(risk_pred, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + 0.05 * (action**2).sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            alog.append(float(an)); rlog.append(float(risk))
            state = ns; gh = gr
        results[condition] = {"mean_action": float(np.mean(np.abs(alog))),
                               "mean_risk": float(np.mean(rlog))}
    m.eval()
    aa = results["full"]["mean_action"]; ab = results["noisy"]["mean_action"]
    ra = results["full"]["mean_risk"]; rb = results["noisy"]["mean_risk"]
    return {"full": results["full"], "noisy": results["noisy"],
            "pass": bool(ab > 0 and aa/ab > 2.0 and rb > ra * 1.3)}


# ===================================================================
# L3-5: Minimal loop model (a = k * grad_hint)
# ===================================================================

class MinimalLoopModel(nn.Module):
    def __init__(self, k=0.5):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(k))

    def forward(self, state, grad_hint):
        return self.k * grad_hint, torch.tensor(0.0)


def minimal_loop_test(env, steps=3000):
    m = MinimalLoopModel(); m.to(DEVICE)
    opt = torch.optim.Adam([m.k], lr=0.01)
    state = env.reset(); gh_prev = 0.0; log = []
    for t in range(steps):
        action, _ = m(state, gh_prev)
        an = float(action.detach().cpu().numpy().item()) if isinstance(action, torch.Tensor) else float(action)
        ns, risk, gr, _, _ = env.step(np.float32(an))
        # loss: if grad_hint ≠ 0, push k to produce action that reduces risk
        # ∂risk/∂k = ∂risk/∂a * ∂a/∂k = gr * gh_prev
        if abs(gh_prev) > 1e-6 and abs(gr) > 1e-6:
            grad_k = gr * gh_prev  # chain rule approximation
            m.k.grad = torch.tensor(grad_k, dtype=torch.float32, device=DEVICE).reshape_as(m.k)
            opt.step()
        opt.zero_grad()
        log.append({"t": t, "action": an, "risk": risk, "k": float(m.k.item())})
        state = ns; gh_prev = gr
    # Compare vs open
    env_o = LatentShiftDoubleWellEnv(drift_strength=0.2, seed=9999)
    ro = []
    for _ in range(500): _, risk, _, _, _ = env_o.step(np.float32(0.0)); ro.append(risk)
    open_risk = float(np.mean(ro))
    min_risk = float(np.mean([l["risk"] for l in log[-500:]]))
    return {"minimal_mean_risk": min_risk, "open_risk": open_risk,
            "final_k": float(m.k.item()),
            "pass": min_risk < open_risk * 0.8}


# ===================================================================
# L3-6: Memory vs loop (temporal shuffle)
# ===================================================================

def memory_vs_loop_test(env, model, steps=2000):
    state = env.reset(); gh = None
    buf, nlog, slog = [], [], []
    for t in range(steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        ghv = gh if gh is not None else 0.0
        ght = torch.tensor([[ghv]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            h = model.encoder(torch.cat([s_t, ght], dim=-1))
            a = model.actor(h); an = a.cpu().numpy().squeeze(0)
        ns, risk, gr, _, _ = env.step(an)
        nlog.append(risk)
        buf.append(state.copy())
        if len(buf) > 10 and np.random.random() < 0.1:
            state = buf[np.random.randint(0, len(buf))].copy()
        else:
            state = ns
        gh = gr
        slog.append(risk)
    return {"normal_risk": float(np.mean(nlog)),
            "shuffled_risk": float(np.mean(slog)),
            "pass": float(np.mean(slog)) > float(np.mean(nlog)) * 1.5}


# ===================================================================
# L3-7: Counterfactual break (normal → detach → normal)
# ===================================================================

def counterfactual_break(env, model, steps=3000):
    state = env.reset(); gh = None; p1, p2, p3 = [], [], []
    for t in range(steps):
        if t == 1000: phase = "phase2"
        elif t == 2000: phase = "phase3"
        else: phase = "phase1"
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        ghv = gh if gh is not None else 0.0
        ght = torch.tensor([[ghv]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            h = model.encoder(torch.cat([s_t, ght], dim=-1))
            if phase == "phase2":
                a = model.actor(h).detach()
            else:
                a = model.actor(h)
            an = float(a.cpu().numpy().squeeze(0).item())
        ns, risk, gr, _, _ = env.step(an)
        (p1 if phase == "phase1" else p2 if phase == "phase2" else p3).append(risk)
        state = ns; gh = gr
    return {"p1_risk": float(np.mean(p1[-200:])),
            "p2_risk": float(np.mean(p2)),
            "p3_risk": float(np.mean(p3[-200:])),
            "pass": float(np.mean(p2)) > float(np.mean(p1[-200:])) * 1.5}


# ===================================================================
# L3-8: Loop gain analysis
# ===================================================================

def loop_gain_analysis(log):
    gains = []
    for t in range(1, len(log)):
        da = log[t]["action"] - log[t-1]["action"]
        dg = log[t]["grad_risk"] - log[t-1]["grad_risk"]
        if abs(dg) > 1e-8: gains.append(da / dg)
    if len(gains) < 3: return {"pass": False, "early_gain": 0, "late_gain": 0, "late_std": 1}
    n = len(gains); eg = float(np.mean(gains[:n//3]))
    lg = float(np.mean(gains[2*n//3:])); lstd = float(np.std(gains[2*n//3:]))
    return {"early_gain": eg, "late_gain": lg, "late_std": lstd,
            "pass": lstd < 0.2}


# ===================================================================
# L3-9: Self-organization (random init, zero prior)
# ===================================================================

def self_organization_test(env_factory, n_seeds=8, steps=5000):
    results = []
    for s in range(n_seeds):
        env = env_factory(seed=s); m = _make_model()
        log = record_training_trajectory(env, m, steps)
        t_star = detect_emergence(log)
        final_risk = float(np.mean([l["risk"] for l in log[-500:]]))
        results.append({"seed": s, "t_star": t_star, "final_risk": final_risk})
    valid = [r for r in results if r["t_star"] is not None]
    er = len(valid) / n_seeds
    return {"emergence_rate": er, "results": results,
            "pass": er > 0.8}


# ===================================================================
# Full runner
# ===================================================================

def run_layer3(n_seeds=8, n_steps=5000, drift=0.2, quick=False):
    os.makedirs(OUT, exist_ok=True)
    ns = n_seeds if not quick else 2
    nst = n_steps if not quick else 500

    print(f"\n{'='*70}")
    print(f"Layer 3: Emergence Verification")
    print(f"  seeds={ns}  steps={nst}  drift={drift}")
    print(f"{'='*70}\n")

    env_fn = lambda seed=0: _make_env(drift, seed)
    scores = {}

    # L3-1 + L3-2 + L3-8: share trajectory log
    print("=== L3-1/2/8: Trajectory + Emergence + Gain ===")
    env = env_fn(0); m = _make_model()
    log = record_training_trajectory(env, m, nst)
    t_star = detect_emergence(log)
    l3_8 = loop_gain_analysis(log)
    print(f"  t*={t_star}  gain: early={l3_8['early_gain']:.3f} late={l3_8['late_gain']:.3f} σ={l3_8['late_std']:.3f}")
    scores["L3-2"] = True if t_star is not None else False
    scores["L3-8"] = l3_8["pass"]

    # L3-2 multi-seed
    l3_2 = run_l3_2(env_fn, ns, nst)
    print(f"  emergence_rate={l3_2['emergence_rate']:.2f} t*_mean={l3_2['mean_t_star']} t*_std={l3_2['std_t_star']}")
    scores["L3-2"] = l3_2["pass"]

    # L3-3
    print("\n=== L3-3: Gradient Causal Chain ===")
    env3 = env_fn(3); m3 = _make_model()
    l3_3 = gradient_causal_chain_test(env3, m3, nst)
    print(f"  p1_act={l3_3['phase1_action']:.4f} p2_act={l3_3['phase2_action']:.4f} p1_risk={l3_3['phase1_risk']:.4f} p2_risk={l3_3['phase2_risk']:.4f}")
    scores["L3-3"] = l3_3["pass"]

    # L3-4
    print("\n=== L3-4: Information Utilization ===")
    env4 = env_fn(4); m4 = _make_model()
    l3_4 = information_test(env4, m4, nst)
    print(f"  full: |a|={l3_4['full']['mean_action']:.4f} risk={l3_4['full']['mean_risk']:.4f}")
    print(f"  noisy: |a|={l3_4['noisy']['mean_action']:.4f} risk={l3_4['noisy']['mean_risk']:.4f}")
    scores["L3-4"] = l3_4["pass"]

    # L3-5
    print("\n=== L3-5: Minimal Loop ===")
    env5 = env_fn(5)
    l3_5 = minimal_loop_test(env5, nst)
    print(f"  minimal_risk={l3_5['minimal_mean_risk']:.4f} open_risk={l3_5['open_risk']:.4f} k={l3_5['final_k']:.4f}")
    scores["L3-5"] = l3_5["pass"]

    # L3-6
    print("\n=== L3-6: Memory vs Loop ===")
    env6 = env_fn(6); m6 = _make_model()
    _ = record_training_trajectory(env6, m6, nst)
    env6b = env_fn(7)
    l3_6 = memory_vs_loop_test(env6b, m6, min(nst, 2000))
    print(f"  normal_risk={l3_6['normal_risk']:.4f} shuffled_risk={l3_6['shuffled_risk']:.4f}")
    scores["L3-6"] = l3_6["pass"]

    # L3-7
    print("\n=== L3-7: Counterfactual Break ===")
    env7 = env_fn(8); m7 = _make_model()
    _ = record_training_trajectory(env7, m7, nst)
    env7b = env_fn(9)
    l3_7 = counterfactual_break(env7b, m7, min(nst, 3000))
    print(f"  p1={l3_7['p1_risk']:.4f} p2={l3_7['p2_risk']:.4f} p3={l3_7['p3_risk']:.4f}")
    scores["L3-7"] = l3_7["pass"]

    # L3-9
    print("\n=== L3-9: Self-Organization ===")
    l3_9 = self_organization_test(lambda seed: _make_env(drift, seed), ns, nst)
    print(f"  emergence_rate={l3_9['emergence_rate']:.2f}")
    scores["L3-9"] = l3_9["pass"]

    # Summary
    n_pass = sum(scores.values())
    n_total = len(scores)
    print(f"\n{'='*70}")
    print(f"Layer 3: {n_pass}/{n_total} sub-experiments passed")
    for k, v in scores.items(): print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  {'ALL_PASS' if n_pass >= 6 else 'FAIL'} (need ≥6/8)")
    print(f"{'='*70}")

    out = {"scores": {k: bool(v) for k, v in scores.items()},
           "n_pass": n_pass, "n_total": n_total,
           "details": {"L3-2": l3_2, "L3-3": l3_3, "L3-4": l3_4,
                       "L3-5": l3_5, "L3-6": l3_6, "L3-7": l3_7,
                       "L3-8": l3_8, "L3-9": l3_9}}
    with open(os.path.join(OUT, "emergence_full.json"), "w") as f:
        json.dump(out, f, indent=2, default=_jdef)
    return out


def main():
    p = argparse.ArgumentParser("Layer 3 Emergence")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--drift", type=float, default=0.2)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    run_layer3(n_seeds=args.seeds, n_steps=args.steps, drift=args.drift, quick=args.quick)


if __name__ == "__main__":
    main()
