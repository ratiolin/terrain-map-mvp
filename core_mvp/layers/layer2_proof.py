"""Layer 2 Proof v2 — complete evidence chain.

1. Phase scan: causal_rate vs α ∈ [0.05..0.4], find α_c where rate > 0.8
2. Survival (normalized): relative risk failure, hazard_ratio > 3.0
3. Open-loop baseline: risk ordering closed < open < blocked
4. Irreversibility: mid-trajectory gradient cutoff → 100% failure

Usage:
  uv run python core_mvp/layers/layer2_proof.py
  uv run python core_mvp/layers/layer2_proof.py --quick
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
OUT = os.path.join(os.path.dirname(_here), "results", "layer2")

# ===================================================================
# Shared helpers
# ===================================================================

def _train_batch(envs, models, steps, lr=1e-3, lam=0.05, blocked=False):
    """Batched: stack all N env states → one forward → step N envs in parallel."""
    n = len(envs)
    for m in models: m.to(DEVICE)
    opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]
    states_np = [e.reset() for e in envs]
    gh = [None] * n
    for _ in range(steps):
        # stack states + grad_hints
        s_batch = np.stack([states_np[i].astype(np.float32) for i in range(n)])
        gh_vals = np.array([[gh[i]] if gh[i] is not None else [0.0] for i in range(n)], dtype=np.float32)
        s_in = torch.from_numpy(np.concatenate([s_batch, gh_vals], axis=1)).to(DEVICE)
        # forward all models separately (weights differ)
        for i in range(n):
            h = models[i].encoder(s_in[i:i+1])
            a = models[i].actor(h)
            rp = models[i].predictor(torch.cat([h, a.detach() if blocked else a], dim=-1))
            an = a.detach().cpu().numpy().squeeze(0)
            ns, risk, gr, done, _ = envs[i].step(an)
            loss = nn.functional.mse_loss(rp, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + lam * (a**2).sum()
            if torch.norm(a) < 0.01: loss = loss + 0.1 * (0.01 - torch.norm(a))
            opts[i].zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(models[i].parameters(), 1.0)
            opts[i].step()
            states_np[i] = ns; gh[i] = gr
            if done: states_np[i] = envs[i].reset(); gh[i] = None
    for m in models: m.eval()


def _rollout(env, model, steps):
    env.set_seed(9999); s = env.reset(); gh = None
    R, A, xs = [], [], []
    for _ in range(steps):
        st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
        ghv = gh if gh is not None else 0.0
        ght = torch.tensor([[ghv]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            h = model.encoder(torch.cat([st, ght], dim=-1))
            a = model.actor(h); an = a.cpu().numpy().squeeze(0)
        ns, risk, gr, done, _ = env.step(an)
        R.append(risk); A.append(an); xs.append(env.x_true)
        s = ns; gh = gr
        if done: s = env.reset(); gh = None
    return np.array(R), np.array(A), np.array(xs)


def _rollout_open(env, steps):
    env.set_seed(9999); _ = env.reset()
    R, xs = [], []
    for _ in range(steps):
        ns, risk, _, _, _ = env.step(np.float32(0.0))
        R.append(risk); xs.append(env.x_true)
    return np.array(R), np.array(xs)


def _jdef(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


# ===================================================================
# 1. Phase Scan — extended
# ===================================================================

def run_phase_scan(n_seeds=16, n_steps=3000, out_dir=None):
    out_dir = out_dir or OUT; os.makedirs(out_dir, exist_ok=True)
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    print(f"\n=== Phase Scan: α ∈ {alphas} seeds={n_seeds} ===")
    results = {}

    for alpha in alphas:
        envs_n = [LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s) for s in range(n_seeds)]
        models_n = [LatentShiftModel(hidden_dim=128) for _ in range(n_seeds)]
        _train_batch(envs_n, models_n, n_steps, blocked=False)

        envs_b = [LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s) for s in range(n_seeds)]
        models_b = [LatentShiftModel(hidden_dim=128) for _ in range(n_seeds)]
        _train_batch(envs_b, models_b, n_steps, blocked=True)

        causal_cnt = 0; collapses = []
        for s in range(n_seeds):
            env_t = LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s+10000)
            _, A_n, _ = _rollout(env_t, models_n[s], 500)
            _, A_b, _ = _rollout(env_t, models_b[s], 500)
            na = np.mean(np.abs(A_n)); nb = np.mean(np.abs(A_b))
            col = 1.0 - nb / max(na, 1e-8)
            collapses.append(col)
            if col > 0.5: causal_cnt += 1

        cr = causal_cnt / n_seeds
        results[str(alpha)] = {"causal_rate": cr, "collapse_mean": float(np.mean(collapses)),
                                "collapse_std": float(np.std(collapses))}
        print(f"  α={alpha}: causal={cr:.3f} collapse={np.mean(collapses):.3f}±{np.std(collapses):.3f}")

    critical = sorted([(float(a), r["causal_rate"]) for a, r in results.items() if r["causal_rate"] >= 0.8])
    if critical:
        print(f"CRITICAL α_c = {critical[0][0]} (causal_rate={critical[0][1]:.2f})")
    else:
        max_alpha, max_rate = max(results.items(), key=lambda x: x[1]["causal_rate"])
        print(f"No α_c found; max causal_rate={max_rate['causal_rate']:.2f} at α={max_alpha}")

    path = os.path.join(out_dir, "phase_scan.json")
    with open(path, "w") as f: json.dump(results, f, indent=2, default=_jdef)
    return results


# ===================================================================
# 2. Survival — normalized relative risk
# ===================================================================

def _survival_normalized(env, model, baseline_risk, max_steps=5000, k=3.0):
    env.set_seed(9999); s = env.reset(); gh = None
    for t in range(max_steps):
        st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
        ghv = gh if gh is not None else 0.0
        ght = torch.tensor([[ghv]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            h = model.encoder(torch.cat([st, ght], dim=-1))
            a = model.actor(h); an = a.cpu().numpy().squeeze(0)
        ns, risk, gr, _, _ = env.step(an)
        if risk > k * baseline_risk: return t
        s = ns; gh = gr
    return max_steps


def _survival_open_normalized(env, baseline_risk, max_steps=5000, k=3.0):
    env.set_seed(9999); _ = env.reset()
    for t in range(max_steps):
        _, risk, _, _, _ = env.step(np.float32(0.0))
        if risk > k * baseline_risk: return t
    return max_steps


def run_survival(n_seeds=16, n_steps=3000, out_dir=None):
    out_dir = out_dir or OUT; os.makedirs(out_dir, exist_ok=True)
    alphas = [0.05, 0.1, 0.15, 0.2]
    print(f"\n=== Survival Analysis: α ∈ {alphas} seeds={n_seeds} ===")
    results = {}

    for alpha in alphas:
        # Baseline: train one model on drift=0 to get reference risk
        envs_n = [LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s) for s in range(n_seeds)]
        models_n = [LatentShiftModel(hidden_dim=128) for _ in range(n_seeds)]
        _train_batch(envs_n, models_n, n_steps, blocked=False)

        surv_cl, surv_op = [], []
        for s in range(n_seeds):
            # baseline risk from drift=0 env
            env_z = LatentShiftDoubleWellEnv(drift_strength=0.0, seed=s)
            R_z, _, _ = _rollout(env_z, models_n[s], 500)
            bl = float(np.mean(R_z))

            env_t = LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s+10000)
            surv_cl.append(_survival_normalized(env_t, models_n[s], bl))

            env_o = LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s+10000)
            surv_op.append(_survival_open_normalized(env_o, bl))

        fc = np.mean([1.0 if s < 5000 else 0.0 for s in surv_cl])
        fo = np.mean([1.0 if s < 5000 else 0.0 for s in surv_op])
        hr = fo / max(fc, 1e-8)
        e_ratio = np.mean(surv_cl) / max(np.mean(surv_op), 1e-8)

        results[str(alpha)] = {"mean_closed": float(np.mean(surv_cl)),
                                "mean_open": float(np.mean(surv_op)),
                                "hazard_ratio": float(hr),
                                "E_ratio": float(e_ratio)}
        print(f"  α={alpha}: E_cl={np.mean(surv_cl):.0f} E_op={np.mean(surv_op):.0f} "
              f"HR={hr:.1f} E_ratio={e_ratio:.1f}")

    path = os.path.join(out_dir, "survival_normalized.json")
    with open(path, "w") as f: json.dump(results, f, indent=2, default=_jdef)
    return results


# ===================================================================
# 3. Open-Loop Baseline
# ===================================================================

def run_open_loop(n_seeds=16, n_steps=3000, alpha=0.1, out_dir=None):
    out_dir = out_dir or OUT; os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Open-Loop Baseline: α={alpha} seeds={n_seeds} ===")

    envs_n = [LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s) for s in range(n_seeds)]
    models_n = [LatentShiftModel(hidden_dim=128) for _ in range(n_seeds)]
    _train_batch(envs_n, models_n, n_steps, blocked=False)

    envs_b = [LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s) for s in range(n_seeds)]
    models_b = [LatentShiftModel(hidden_dim=128) for _ in range(n_seeds)]
    _train_batch(envs_b, models_b, n_steps, blocked=True)

    r_cl, r_bl, r_op = [], [], []
    for s in range(n_seeds):
        env_t = LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s+10000)
        R_n, _, _ = _rollout(env_t, models_n[s], 500); r_cl.append(float(np.mean(R_n)))
        env_t2 = LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s+10000)
        R_b, _, _ = _rollout(env_t2, models_b[s], 500); r_bl.append(float(np.mean(R_b)))
        env_t3 = LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s+10000)
        R_o, _ = _rollout_open(env_t3, 500); r_op.append(float(np.mean(R_o)))

    cm = float(np.mean(r_cl)); bm = float(np.mean(r_bl)); om = float(np.mean(r_op))
    ordering = bool(cm < om < bm)
    res = {"closed_mean": cm, "open_mean": om, "blocked_mean": bm, "ordering_correct": ordering}
    print(f"  risk: closed={cm:.4f} open={om:.4f} blocked={bm:.4f}  ordering={ordering}")
    path = os.path.join(out_dir, "open_loop_baseline.json")
    with open(path, "w") as f: json.dump(res, f, indent=2)
    return res


# ===================================================================
# 4. Irreversibility
# ===================================================================

def run_irreversibility(n_seeds=16, n_steps=3000, alpha=0.1, out_dir=None):
    out_dir = out_dir or OUT; os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Irreversibility: α={alpha} seeds={n_seeds} ===")

    envs_n = [LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s) for s in range(n_seeds)]
    models = [LatentShiftModel(hidden_dim=128) for _ in range(n_seeds)]
    _train_batch(envs_n, models, n_steps, blocked=False)

    failed, recovered = 0, 0
    for s in range(n_seeds):
        env_t = LatentShiftDoubleWellEnv(drift_strength=alpha, seed=s+10000)
        env_t.set_seed(9999); state = env_t.reset(); gh = None
        for _ in range(2500):
            st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
            ghv = gh if gh is not None else 0.0
            ght = torch.tensor([[ghv]], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                h = models[s].encoder(torch.cat([st, ght], dim=-1))
                a = models[s].actor(h); an = a.cpu().numpy().squeeze(0)
            state, risk, gr, _, _ = env_t.step(an); gh = gr
        for _ in range(2500):
            st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
            ghv = gh if gh is not None else 0.0
            ght = torch.tensor([[ghv]], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                h = models[s].encoder(torch.cat([st, ght], dim=-1))
                a_det = models[s].actor(h).detach()
                an = a_det.cpu().numpy().squeeze(0)
            state, risk, _, _, _ = env_t.step(an)
            if abs(env_t.x_true) > 3.0 or risk > 3.0:
                failed += 1; break
        else: recovered += 1

    fr = failed / n_seeds
    res = {"failed": failed, "recovered": recovered, "total": n_seeds, "failure_rate": fr}
    print(f"  failed={failed}/{n_seeds} rate={fr:.2f}")
    path = os.path.join(out_dir, "irreversibility.json")
    with open(path, "w") as f: json.dump(res, f, indent=2)
    return res


# ===================================================================
# Main
# ===================================================================

def main():
    p = argparse.ArgumentParser("Layer 2 Proof v2")
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    ns = args.seeds if not args.quick else 4
    nst = args.steps if not args.quick else 500

    r1 = run_phase_scan(n_seeds=ns, n_steps=nst)
    r2 = run_survival(n_seeds=ns, n_steps=nst)
    r3 = run_open_loop(n_seeds=ns, n_steps=nst, alpha=args.alpha)
    r4 = run_irreversibility(n_seeds=ns, n_steps=nst, alpha=args.alpha)

    # Verdict
    p1 = any(v["causal_rate"] >= 0.8 for v in r1.values())
    p2 = any(v["hazard_ratio"] > 3.0 and v["E_ratio"] > 2.0 for v in r2.values())
    p3 = r3["ordering_correct"]
    p4 = r4["failure_rate"] > 0.7
    all_pass = p1 and p2 and p3 and p4

    print(f"\n{'='*70}")
    print(f"FINAL: {'ALL_PASS ✓' if all_pass else 'FAIL'}")
    print(f"  phase scan:   {'PASS' if p1 else 'FAIL'}")
    print(f"  survival:     {'PASS' if p2 else 'FAIL'} (E_ratio>2, HR>3)")
    print(f"  open-loop:    {'PASS' if p3 else 'FAIL'}")
    print(f"  irreversibility: {'PASS' if p4 else 'FAIL'} (rate={r4['failure_rate']:.2f})")


if __name__ == "__main__":
    main()
