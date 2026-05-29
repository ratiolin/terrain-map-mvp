"""Layer 1: Closed-Loop Dominance & Directional Quality.

Phase A — Existence: train ClosedLoopModel, verify genuine control via
  cross-mode rollout (closed/open/pseudo), compute W2 / grad_norm / gain /
  eff_net / action PCA / Jacobian, then render a pass/fail verdict.

Phase B — Direction: sweep 4 loss variants (baseline, directional, one-sided,
  baseline-sign) across α/β, measure cos_sim sign consistency and
  proj_efficiency, recommend the best variant.

Usage:
  uv run python core_mvp/layers/layer1_core.py --phase A
  uv run python core_mvp/layers/layer1_core.py --phase B
  uv run python core_mvp/layers/layer1_core.py --phase all --quick
"""

from __future__ import annotations

import os, sys, json, time, argparse
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.dirname(_root))

from core_mvp.core.core_env import MultiModeEnv, PseudoActionGenerator
from core_mvp.core.core_models import ClosedLoopModel, get_designed_hidden_dim
from core_mvp.core.core_metrics import (
    compute_wasserstein, compute_w2_gaussian, compute_spearman_correlation,
    compute_k80, effective_rank as _eff_rank_fn,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
K = 2
D_DIMS = [2, 5, 10, 20, 50, 100]
N_SEEDS = 5
EPISODES = 5
STEPS = 2000
ROLLOUT = 1000
LR = 1e-3
LAMBDA = 0.1
DRIFT = 0.5
ALPHAS = [0.01, 0.1, 0.5]
BETAS = [0.01, 0.1]

OUT = os.path.join(os.path.dirname(_here), "results", "layer1")


# ===================================================================
# Shared helpers
# ===================================================================

def _make_env(d, seed, mode="closed"):
    env = MultiModeEnv(d_total=d, k_controlled=K, mode=mode, theta=0.5,
                       noise_std=0.05, coupling=0.05, drift=DRIFT,
                       force_scale=0.1, action_scale=0.1, seed=seed,
                       pseudo_affects_noise=False)
    env.calibrate_noise_scales()
    return env


def _make_model(d, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    return ClosedLoopModel(state_dim=d, hidden_dim=get_designed_hidden_dim(d), action_dim=K)


def _mean_std(vals):
    a = np.array(vals, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)), "n": int(len(a))}


def _json_default(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


# ===================================================================
# Phase A — closed-loop dominance verification
# ===================================================================

def _train_closed(model, env, eps, steps, lr, lam, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    env.set_seed(seed); state = env.reset()
    anorms = []
    for _ in range(eps * steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        a, h, p = model(s_t)
        an = a.squeeze(0).detach().cpu().numpy()
        anorms.append(float(np.linalg.norm(an)))
        ns, risk, _, _ = env.step(an + np.random.randn(*an.shape) * 0.03)
        loss = nn.functional.mse_loss(p, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + lam * torch.mean(a**2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); state = ns
    return np.array(anorms)


def _rollout(env, model, steps, seed, zero_action=False, pseudo_gen=None):
    env.set_seed(seed); state = env.reset()
    if pseudo_gen: pseudo_gen.reset(state)
    S, A, NS, R = [], [], [], []
    for _ in range(steps):
        if zero_action: an = np.zeros(K, dtype=np.float32)
        elif pseudo_gen: an = pseudo_gen.step(state)
        else: an = model.act_numpy(state)
        ns, risk, _, _ = env.step(an)
        S.append(state.copy()); A.append(an.copy()); NS.append(ns.copy()); R.append(risk)
        state = ns
    return {"states": np.array(S, dtype=np.float32), "actions": np.array(A, dtype=np.float32),
            "next_states": np.array(NS, dtype=np.float32), "risks": np.array(R, dtype=np.float32)}


def _r_metrics(actions, ns, k=K):
    if actions.shape[0] < 2: return {"r_max": 0.0, "r_mean": 0.0}
    sc = ns[:, :k]; m = np.zeros((actions.shape[1], sc.shape[1]))
    for i in range(actions.shape[1]):
        for j in range(sc.shape[1]):
            rho, _ = compute_spearman_correlation(actions[:, i], sc[:, j])
            m[i, j] = rho if not np.isnan(rho) else 0.0
    a = np.abs(m)
    return {"r_max": float(np.max(a)), "r_mean": float(np.mean(a))}


def _dist_metrics(tcl, top, tps, k=K):
    scl = tcl["next_states"][:, :k]; sop = top["next_states"][:, :k]; sps = tps["next_states"][:, :k]
    w2co = compute_w2_gaussian(scl, sop); w2po = compute_w2_gaussian(sps, sop)
    return {"w2_cl_op": float(w2co), "w2_ps_op": float(w2po),
            "pr": float(w2po / max(w2co, 1e-12))}


def _gain(tcl, top, k=K):
    return float(np.mean(np.linalg.norm(tcl["next_states"][:, :k] - top["next_states"][:, :k], axis=1)))


def _grad_norm(model, states, n=200):
    ii = np.linspace(0, len(states)-1, min(n, len(states)), dtype=int)
    return model.compute_grad_norm(states[ii])


def _eff_pca(actions, ns_ctrl):
    a = actions.astype(np.float64)
    if a.shape[0] < 3: return {"eff_rank": 1.0, "k90": a.shape[1]}
    cov = (a - a.mean(0)).T @ (a - a.mean(0)) / (a.shape[0] - 1)
    try: _, S, _ = np.linalg.svd(cov, full_matrices=False)
    except: S = np.ones(a.shape[1])
    S = np.maximum(S, 0); p = S / (S.sum() + 1e-12); p = p[p > 0]
    er = float(np.exp(-np.sum(p * np.log(p)))) if len(p) > 0 else 1.0
    cs = np.cumsum(S) / (S.sum() + 1e-12)
    k90 = int(np.searchsorted(cs, 0.90) + 1)
    return {"eff_rank": er, "k90": min(k90, len(S))}


def _env_jac(env, model, states, n=200):
    eps = 1e-3; fv = []
    ii = np.linspace(0, len(states)-1, min(n, len(states)), dtype=int)
    for s in states[ii]:
        ar = model.act_numpy(s); sv = env.save_state()
        env.state = s.copy(); env._rng.set_state(sv["rng_state"])
        J = np.zeros((K, K), dtype=np.float64)
        for j in range(K):
            e = np.zeros(K, dtype=np.float32); e[j] = eps
            env.restore_state(sv); np_, _, _, _ = env.step(ar + e)
            env.restore_state(sv); nm_, _, _, _ = env.step(ar - e)
            J[:, j] = (np_[:K].astype(np.float64) - nm_[:K].astype(np.float64)) / (2 * eps)
        env.restore_state(sv); fv.append(np.linalg.norm(J, 'fro'))
    return float(np.mean(fv))


def _verdict_a(rcl, rps, w2, pr, gn, gain, an, d):
    c = [w2 > 1e-4, (d == K) or (pr < 0.3),
         (rcl > 0.05 and (rcl > rps * 1.1 or rps < 0.01)),
         gn > 1e-6, gain > 1e-4, an > 1e-3]
    return {"pass": all(c), "criteria": {
        "w2": bool(c[0]), "pr": bool(c[1]), "r": bool(c[2]),
        "gn": bool(c[3]), "gain": bool(c[4]), "an": bool(c[5])}}


def run_phase_a(d_dims=None, seeds=None, eps=EPISODES, steps=STEPS, rsteps=ROLLOUT, out_dir=None):
    d_dims = d_dims or D_DIMS; seeds = seeds or list(range(N_SEEDS))
    out_dir = out_dir or OUT; os.makedirs(out_dir, exist_ok=True)
    agg = {}
    for d in d_dims:
        sr = {}
        for seed in seeds:
            env = _make_env(d, seed, "closed"); m = _make_model(d, seed)
            an = _train_closed(m, env, eps, steps, LR, LAMBDA, seed); m.eval()
            fa = float(np.mean(an[-500:])) if len(an) >= 500 else float(np.mean(an))
            if fa < 1e-3:
                sr[seed] = {"aborted": True, "action_norm": fa}; continue
            base = _make_env(d, seed + 10000, "closed")
            tcl = _rollout(deepcopy(base), m, rsteps, seed + 20000)
            base.set_mode("closed")
            be = deepcopy(base); be.set_mode("open")
            top = _rollout(be, m, rsteps, seed + 20000, zero_action=True)
            be2 = deepcopy(base); be2.set_mode("pseudo")
            pg = PseudoActionGenerator(d, K, seed=seed + 30000, scale=0.1)
            tps = _rollout(be2, m, rsteps, seed + 20000, pseudo_gen=pg)
            rc = _r_metrics(tcl["actions"], tcl["next_states"])
            rp = _r_metrics(tps["actions"], tps["next_states"])
            dm = _dist_metrics(tcl, top, tps)
            gv = _gain(tcl, top); gn = _grad_norm(m, tcl["states"])
            pca = _eff_pca(tcl["actions"], tcl["next_states"][:, :K])
            jf = _env_jac(deepcopy(base), m, tcl["states"])
            anm = float(np.mean(np.linalg.norm(tcl["actions"], axis=1)))
            v = _verdict_a(rc["r_max"], rp["r_max"], dm["w2_cl_op"], dm["pr"], gn, gv, anm, d)
            sr[seed] = {"r_cl": rc, "r_ps": rp, "dist": dm, "gain": gv,
                        "grad_norm": gn, "action_norm": anm, "pca": pca,
                        "jac_frob": jf, "verdict": v}
        agg_d = {}
        active = [s for s in sr.values() if not s.get("aborted")]
        for k in ["r_cl", "gain", "grad_norm", "action_norm", "jac_frob"]:
            vals = [s.get(k, {}).get("r_max", s.get(k, 0)) if isinstance(s.get(k), dict) else s.get(k, 0) for s in active]
            if vals: agg_d[k] = _mean_std(vals)
        for k in ["dist", "pca"]:
            pass
        agg_d["w2"] = _mean_std([s.get("dist", {}).get("w2_cl_op", 0) for s in active])
        agg_d["pr"] = _mean_std([s.get("dist", {}).get("pr", 0) for s in active])
        agg_d["eff_rank"] = _mean_std([s.get("pca", {}).get("eff_rank", 0) for s in active])
        agg_d["pass_rate"] = _mean_std([1.0 if s.get("verdict", {}).get("pass") else 0.0 for s in active])
        agg_d["n_aborted"] = len(sr) - len(active)
        agg[f"d{d}"] = agg_d
        with open(os.path.join(out_dir, f"phase_a_d{d}.json"), "w") as f:
            json.dump({"per_seed": {str(k): v for k, v in sr.items()}, "agg": agg_d}, f, indent=2, default=_json_default)
        print(f"  d={d}: pass={agg_d['pass_rate']['mean']:.2f} w2={agg_d['w2']['mean']:.4f} gain={agg_d['gain']['mean']:.4f} gn={agg_d['grad_norm']['mean']:.4f} |a|={agg_d['action_norm']['mean']:.4f}")
    with open(os.path.join(out_dir, "phase_a_summary.json"), "w") as f:
        json.dump(agg, f, indent=2, default=_json_default)
    print("Phase A complete.")
    return agg


# ===================================================================
# Phase B — directional fix
# ===================================================================

def _train_baseline(m, env, eps, steps, lr, lam, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr); env.set_seed(seed)
    state = env.reset(); anorms, risks = [], []
    for _ in range(eps * steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        a, h, p = m(s_t);         an = a.squeeze(0).detach().cpu().numpy()
        anorms.append(float(np.linalg.norm(an)))
        ns, risk, _, _ = env.step(an + np.random.randn(*an.shape) * 0.03)
        risks.append(risk)
        loss = nn.functional.mse_loss(p, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + lam * torch.mean(a**2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); state = ns
    return np.array(anorms), np.array(risks)


def _train_directional(m, env, eps, steps, lr, lam, alpha, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr); env.set_seed(seed)
    state = env.reset(); anorms, risks = [], []
    for _ in range(eps * steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        a, h, p = m(s_t);         an = a.squeeze(0).detach().cpu().numpy()
        anorms.append(float(np.linalg.norm(an)))
        ns, risk, _, _ = env.step(an + np.random.randn(*an.shape) * 0.03)
        risks.append(risk)
        loss = (nn.functional.mse_loss(p, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE))
                + alpha * p.mean() + lam * torch.mean(a**2))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); state = ns
    return np.array(anorms), np.array(risks)


def _train_onesided(m, env, eps, steps, lr, lam, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr); env.set_seed(seed)
    state = env.reset(); anorms, risks = [], []
    for _ in range(eps * steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        a, h, p = m(s_t);         an = a.squeeze(0).detach().cpu().numpy()
        anorms.append(float(np.linalg.norm(an)))
        ns, risk, _, _ = env.step(an + np.random.randn(*an.shape) * 0.03)
        risks.append(risk)
        loss = torch.mean(torch.relu(p - torch.tensor([[risk]], dtype=torch.float32, device=DEVICE))**2) + lam * torch.mean(a**2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); state = ns
    return np.array(anorms), np.array(risks)


def _train_bsign(m, env, eps, steps, lr, lam, beta, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr); env.set_seed(seed)
    state = env.reset(); anorms, risks = [], []
    ema = None; decay = 0.99
    for _ in range(eps * steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        a, h, p = m(s_t);         an = a.squeeze(0).detach().cpu().numpy()
        anorms.append(float(np.linalg.norm(an)))
        ns, risk, _, _ = env.step(an + np.random.randn(*an.shape) * 0.03)
        risks.append(risk)
        if ema is None: ema = p.detach().clone()
        else: ema = decay * ema + (1 - decay) * p.detach()
        adv = p - ema
        loss = torch.mean(adv**2) + beta * torch.mean(torch.sign(adv)) + lam * torch.mean(a**2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); state = ns
    return np.array(anorms), np.array(risks)


def _eval_align(m, traj, n=500):
    st = traj["states"]; ii = np.linspace(0, len(st)-1, min(n, len(st)), dtype=int)
    cv, pv = [], []
    for s in st[ii]:
        st_ = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
        h = m.encoder(st_); a = m.actor(h)
        ad = a.detach().requires_grad_(True)
        rp = m.predictor(torch.cat([h, ad], -1)); m.zero_grad()
        rp.sum().backward(retain_graph=True); g = ad.grad
        if g is None: m.zero_grad(); continue
        an = a.detach().squeeze(0).cpu().numpy().astype(np.float64)
        gn = g.squeeze(0).cpu().numpy().astype(np.float64)
        na, ng = np.linalg.norm(an), np.linalg.norm(gn)
        if na > 1e-12 and ng > 1e-12:
            cv.append(float(np.dot(an, gn) / (na * ng)))
            pv.append(float(abs(np.dot(an, gn) / ng) / na))
        else: cv.append(0.0); pv.append(0.0)
        m.zero_grad()
    ca = np.array(cv); pa = np.array(pv)
    return {"cos_mean": float(np.mean(ca)), "cos_std": float(np.std(ca)),
            "proj_mean": float(np.mean(pa)), "total_risk": float(np.mean(traj["risks"])),
            "sign_pos": int(np.sum(ca > 0.01)), "sign_neg": int(np.sum(ca < -0.01))}


def run_phase_b(d_dims=None, seeds=None, eps=EPISODES, steps=STEPS, rsteps=ROLLOUT, out_dir=None):
    d_dims = d_dims or D_DIMS; seeds = seeds or list(range(N_SEEDS))
    out_dir = out_dir or OUT; os.makedirs(out_dir, exist_ok=True)
    methods = [("baseline", _train_baseline, {})]
    for a in ALPHAS: methods.append((f"dir_a{a}", _train_directional, {"alpha": a}))
    methods.append(("onesided", _train_onesided, {}))
    for b in BETAS: methods.append((f"bsign_b{b}", _train_bsign, {"beta": b}))
    all_r = {}
    for d in d_dims:
        for label, fn, xargs in methods:
            sm = []
            for seed in seeds:
                env = _make_env(d, seed, "closed"); m = _make_model(d, seed)
                an, _ = fn(m, env, eps, steps, LR, LAMBDA, seed=seed, **xargs); m.eval()
                fa = float(np.mean(an[-500:])) if len(an) >= 500 else float(np.mean(an))
                if fa < 1e-3: sm.append({"aborted": True, "an": fa}); continue
                te = _make_env(d, seed + 10000, "closed")
                tr = _rollout(te, m, rsteps, seed + 20000)
                ev = _eval_align(m, tr); ev["seed"] = seed; ev["an"] = fa
                sm.append(ev)
            act = [s for s in sm if not s.get("aborted")]
            if len(act) < 2: continue
            cv = [s["cos_mean"] for s in act]
            signs = [1 if c > 0.01 else (-1 if c < -0.01 else 0) for c in cv]
            pos, neg = sum(1 for s in signs if s == 1), sum(1 for s in signs if s == -1)
            con = max(pos, neg) / len(signs) if signs else 0
            key = f"d{d}_{label}"
            all_r[key] = {"cos_mean": float(np.mean(cv)), "cos_std": float(np.std(cv)),
                          "proj_mean": float(np.mean([s["proj_mean"] for s in act])),
                          "risk": float(np.mean([s["total_risk"] for s in act])),
                          "sign_consistency": con, "sign_majority": "pos" if pos > neg else "neg"}
            print(f"  {key}: cos={all_r[key]['cos_mean']:+.3f} con={con:.2f} proj={all_r[key]['proj_mean']:.3f} risk={all_r[key]['risk']:.4f}")
    with open(os.path.join(out_dir, "phase_b_summary.json"), "w") as f:
        json.dump(all_r, f, indent=2, default=_json_default)
    print("Phase B complete.")
    return all_r


# ===================================================================
# CLI
# ===================================================================

def main():
    p = argparse.ArgumentParser(description="Layer 1: Dominance + Direction")
    p.add_argument("--phase", choices=["A", "B", "all"], default="all")
    p.add_argument("--d-dims", type=int, nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--rollout", type=int, default=ROLLOUT)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    dd = args.d_dims or D_DIMS
    sd = list(range(args.seeds))
    od = args.out or OUT
    if args.quick:
        dd = [2, 10]; sd = [0, 1, 2]; eps = 2; stp = 500; rst = 500
    else:
        eps, stp, rst = args.episodes, args.steps, args.rollout

    if args.phase in ("A", "all"):
        print("\n=== Phase A: Closed-Loop Dominance ===")
        run_phase_a(dd, sd, eps, stp, rst, od)
    if args.phase in ("B", "all"):
        print("\n=== Phase B: Directional Fix ===")
        run_phase_b(dd, sd, eps, stp, rst, od)


if __name__ == "__main__":
    main()
