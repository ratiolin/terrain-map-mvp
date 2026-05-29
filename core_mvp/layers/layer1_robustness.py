"""Layer 1 Robustness — four supplementary tests on trained ClosedLoopModel.

① Open-Loop Comparison — prove performance gap is from real-time feedback.
② Perturbation Recovery — verify re-stabilisation after environment shift.
③ Long-Term Stability   — confirm no divergence/oscillation over 5000 steps.
④ Control Utilisation   — show action + gradient stay active, respond to g.

All tests use the existing core/ modules.  No architecture or loss changes.

Usage:
  uv run python core_mvp/layers/layer1_robustness.py
  uv run python core_mvp/layers/layer1_robustness.py --quick
"""

from __future__ import annotations

import os, sys, json, time, argparse
from copy import deepcopy
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, _root)

from core_mvp.core.core_env import MultiModeEnv
from core_mvp.core.core_models import ClosedLoopModel, get_designed_hidden_dim
from core_mvp.core.core_metrics import compute_spearman_correlation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
K = 2
D_DIMS = [2, 10, 50, 100]
N_SEEDS = 5
EPISODES = 5
STEPS = 2000
ROLLOUT = 2000
LONG_STEPS = 5000
LR = 1e-3
LAMBDA = 0.1
DRIFT_VALS = [-0.5, 0.0, 0.5, 1.0]   # g values for perturbation + utilisation
OUT = os.path.join(os.path.dirname(_here), "results", "layer1")


# ===================================================================
# Helpers
# ===================================================================

def _env(d, seed, mode="closed", g=0.5):
    e = MultiModeEnv(d_total=d, k_controlled=K, mode=mode, theta=0.5,
                     noise_std=0.05, coupling=0.05, drift=g,
                     force_scale=0.1, action_scale=0.1, seed=seed,
                     pseudo_affects_noise=False)
    e.calibrate_noise_scales()
    return e


def _model(d, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    return ClosedLoopModel(state_dim=d, hidden_dim=get_designed_hidden_dim(d), action_dim=K)


def _train(m, env, eps, stp, lr, lam, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    env.set_seed(seed); s = env.reset(); anorms, risks, preds = [], [], []
    for _ in range(eps * stp):
        st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
        a, h, p = m(st); an = a.squeeze(0).detach().cpu().numpy()
        anorms.append(float(np.linalg.norm(an)))
        ns, risk, _, _ = env.step(an + np.random.randn(*an.shape) * 0.03)
        risks.append(risk); preds.append(float(p.item()))
        loss = nn.functional.mse_loss(p, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + lam * torch.mean(a**2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); s = ns
    return np.array(anorms), np.array(risks), np.array(preds)


def _rollout(env, m, steps, seed, zero_action=False):
    env.set_seed(seed); s = env.reset()
    states, actions, risks, preds = [], [], [], []
    for _ in range(steps):
        if zero_action: an = np.zeros(K, dtype=np.float32)
        else:
            st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                a_out, _, p_out = m(st)
            an = a_out.squeeze(0).cpu().numpy()
            preds.append(float(p_out.item()))
        ns, risk, _, _ = env.step(an)
        states.append(s.copy()); actions.append(an.copy()); risks.append(risk)
        s = ns
    return {"states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "risks": np.array(risks, dtype=np.float32),
            "preds": np.array(preds, dtype=np.float32) if preds else np.array([])}


def _grad_norm_at(m, states, n_sample=200):
    ii = np.linspace(0, len(states)-1, min(n_sample, len(states)), dtype=int)
    return m.compute_grad_norm(states[ii])


def _linreg_slope(y):
    x = np.arange(len(y), dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 2: return 0.0, 1.0
    A = np.stack([x, np.ones_like(x)], axis=1)
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ beta; ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return float(beta[0]), float(r2)


def _mean_std(a):
    a = np.array(a, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)), "n": int(len(a))}


def _json_default(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)


# ===================================================================
# ① Open-Loop Comparison
# ===================================================================

def run_open_closed_comparison(d_dims=None, seeds=None, eps=EPISODES, stp=STEPS,
                               rsteps=ROLLOUT, out_dir=None):
    d_dims = d_dims or D_DIMS; seeds = seeds or list(range(N_SEEDS))
    out_dir = out_dir or OUT; os.makedirs(out_dir, exist_ok=True)
    print("\n=== ① Open vs Closed Loop Comparison ===")
    results = {}
    for d in d_dims:
        sr = {}
        for seed in seeds:
            env_cl = _env(d, seed, "closed")
            m_cl = _model(d, seed)
            _, risks_cl, preds_cl = _train(m_cl, env_cl, eps, stp, LR, LAMBDA, seed)
            m_cl.eval()
            tr_cl = _rollout(_env(d, seed + 10000, "closed"), m_cl, rsteps, seed + 20000)
            # Open-loop: same env, zero action, only predictor runs
            env_op = _env(d, seed + 10000, "open")
            m_op = _model(d, seed)  # fresh model trained on open-loop data
            # Train open-loop: predict risk from state only (action always 0)
            torch.manual_seed(seed); np.random.seed(seed)
            m_op.to(DEVICE)
            opt = torch.optim.Adam(m_op.parameters(), lr=LR)
            env_op.set_seed(seed); s = env_op.reset()
            risks_op, preds_op = [], []
            for _ in range(eps * stp):
                st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
                a, h, p = m_op(st)
                # In open mode, env ignores action → zero-action prediction
                ns, risk, _, _ = env_op.step(np.zeros(K, dtype=np.float32))
                risks_op.append(risk); preds_op.append(float(p.item()))
                loss = nn.functional.mse_loss(p, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m_op.parameters(), 1.0)
                opt.step(); s = ns
            m_op.eval()
            tr_op = _rollout(_env(d, seed + 10000, "open"), m_op, rsteps, seed + 20000,
                             zero_action=True)
            var_cl = float(np.var(np.linalg.norm(tr_cl["states"][:, :K], axis=1)))
            var_op = float(np.var(np.linalg.norm(tr_op["states"][:, :K], axis=1)))
            sr[seed] = {"risk_cl_mean": float(np.mean(tr_cl["risks"])),
                        "risk_op_mean": float(np.mean(tr_op["risks"])),
                        "var_cl": var_cl, "var_op": var_op,
                        "pred_mse_cl": float(np.mean((tr_cl["preds"] - tr_cl["risks"])**2)),
                        "pred_mse_op": float(np.mean(
                            (np.zeros(rsteps) - tr_op["risks"])**2))}
        agg = {k: _mean_std([s[k] for s in sr.values()]) for k in sr[list(sr.keys())[0]]}
        results[f"d{d}"] = agg
        print(f"  d={d}: risk_cl={agg['risk_cl_mean']['mean']:.4f}±{agg['risk_cl_mean']['std']:.4f}  "
              f"risk_op={agg['risk_op_mean']['mean']:.4f}±{agg['risk_op_mean']['std']:.4f}  "
              f"var_ratio={agg['var_cl']['mean']/max(agg['var_op']['mean'],1e-8):.2f}")
    path = os.path.join(out_dir, "open_closed_comparison.json")
    with open(path, "w") as f: json.dump(results, f, indent=2, default=_json_default)
    print(f"Saved → {path}")
    return results


# ===================================================================
# ② Perturbation Recovery
# ===================================================================

def _rollout_with_shift(env, m, steps, seed, shift_at=None, g_before=0.5, g_after=-0.5,
                        zero_action=False):
    """Rollout with drift shift at step `shift_at`. Default: T/2."""
    if shift_at is None: shift_at = steps // 2
    env.set_seed(seed); s = env.reset()
    env.drift = g_before
    risks, actions, states = [], [], []
    for t in range(steps):
        if t == shift_at:
            env.drift = g_after
        if zero_action: an = np.zeros(K, dtype=np.float32)
        else: an = m.act_numpy(s)
        ns, risk, _, _ = env.step(an)
        risks.append(risk); actions.append(np.linalg.norm(an)); states.append(s.copy())
        s = ns
    return {"risks": np.array(risks, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "states": np.array(states, dtype=np.float32)}


def _recovery_time(risks, shift_at, baseline_window=50, threshold=1.2):
    """Steps after shift until risk returns to ≤ threshold × pre-shift mean."""
    if shift_at >= len(risks): return len(risks)
    pre_mean = np.mean(risks[max(0, shift_at - baseline_window):shift_at])
    target = max(pre_mean * threshold, pre_mean + 0.01)
    for i in range(shift_at, len(risks)):
        if risks[i] <= target:
            return i - shift_at
    return len(risks) - shift_at


def run_perturbation_recovery(d_dims=None, seeds=None, eps=EPISODES, stp=STEPS,
                              rsteps=ROLLOUT, out_dir=None):
    d_dims = d_dims or D_DIMS; seeds = seeds or list(range(N_SEEDS))
    out_dir = out_dir or OUT; os.makedirs(out_dir, exist_ok=True)
    shifts = [(-0.5, 1.0), (1.0, -0.5), (0.5, -0.5)]  # (g_before, g_after)
    print("\n=== ② Perturbation Recovery ===")
    results = {}
    for d in d_dims:
        sr_all = {}
        for (gb, ga) in shifts:
            key = f"g{gb}_to_g{ga}"
            sr = {}
            for seed in seeds:
                env = _env(d, seed, "closed", g=gb)
                m = _model(d, seed)
                _train(m, env, eps, stp, LR, LAMBDA, seed); m.eval()
                shift_at = rsteps // 2
                te = _env(d, seed + 10000, "closed", g=gb)
                tr = _rollout_with_shift(te, m, rsteps, seed + 20000,
                                         shift_at=shift_at, g_before=gb, g_after=ga)
                te2 = _env(d, seed + 10000, "open", g=gb)
                tr_op = _rollout_with_shift(te2, m, rsteps, seed + 20000,
                                            shift_at=shift_at, g_before=gb, g_after=ga,
                                            zero_action=True)
                rec = _recovery_time(tr["risks"], shift_at)
                peak = float(np.max(tr["risks"][shift_at:]))
                peak_op = float(np.max(tr_op["risks"][shift_at:]))
                post_mean = float(np.mean(tr["risks"][shift_at + min(rec + 50, len(tr["risks"]) - shift_at - 1):]))
                sr[seed] = {"recovery_steps": rec, "peak_risk": peak,
                            "peak_risk_open": peak_op, "post_risk_mean": post_mean}
            agg = {k: _mean_std([s[k] for s in sr.values()]) for k in sr[list(sr.keys())[0]]}
            sr_all[key] = agg
            print(f"  d={d} {key}: rec={agg['recovery_steps']['mean']:.0f}±{agg['recovery_steps']['std']:.0f}  "
                  f"peak={agg['peak_risk']['mean']:.3f}  peak_op={agg['peak_risk_open']['mean']:.3f}")
        results[f"d{d}"] = sr_all
    path = os.path.join(out_dir, "perturbation_recovery.json")
    with open(path, "w") as f: json.dump(results, f, indent=2, default=_json_default)
    print(f"Saved → {path}")
    return results


# ===================================================================
# ③ Long-Term Stability
# ===================================================================

def run_long_term_stability(d_dims=None, seeds=None, eps=EPISODES, stp=STEPS,
                            long_steps=LONG_STEPS, out_dir=None):
    d_dims = d_dims or D_DIMS; seeds = seeds or list(range(N_SEEDS))
    out_dir = out_dir or OUT; os.makedirs(out_dir, exist_ok=True)
    print("\n=== ③ Long-Term Stability ===")
    results = {}
    for d in d_dims:
        sr = {}
        for seed in seeds:
            env = _env(d, seed, "closed")
            m = _model(d, seed)
            anorms, _, _ = _train(m, env, eps, stp, LR, LAMBDA, seed); m.eval()
            te = _env(d, seed + 10000, "closed")
            tr = _rollout(te, m, long_steps, seed + 20000)
            r = tr["risks"]; a = np.linalg.norm(tr["actions"], axis=1)
            half = len(r) // 2
            r_first = r[:half]; r_second = r[half:]
            a_first = a[:half]; a_second = a[half:]
            slope, r2 = _linreg_slope(r)
            a_slope, _ = _linreg_slope(a)
            sr[seed] = {
                "risk_first_mean": float(np.mean(r_first)), "risk_first_std": float(np.std(r_first)),
                "risk_second_mean": float(np.mean(r_second)), "risk_second_std": float(np.std(r_second)),
                "risk_ratio_2nd_1st": float(np.mean(r_second) / max(np.mean(r_first), 1e-8)),
                "risk_slope": slope, "risk_slope_r2": r2,
                "action_first_mean": float(np.mean(a_first)), "action_second_mean": float(np.mean(a_second)),
                "action_slope": a_slope,
            }
        agg = {k: _mean_std([s[k] for s in sr.values()]) for k in sr[list(sr.keys())[0]]}
        results[f"d{d}"] = agg
        print(f"  d={d}: r_ratio={agg['risk_ratio_2nd_1st']['mean']:.3f}  "
              f"slope={agg['risk_slope']['mean']:.5f}  "
              f"|a|_ratio={agg['action_second_mean']['mean']/max(agg['action_first_mean']['mean'],1e-8):.3f}")
    path = os.path.join(out_dir, "long_term_stability.json")
    with open(path, "w") as f: json.dump(results, f, indent=2, default=_json_default)
    print(f"Saved → {path}")
    return results


# ===================================================================
# ④ Control Utilisation
# ===================================================================

def run_control_utilisation(d_dims=None, seeds=None, eps=EPISODES, stp=STEPS,
                            rsteps=ROLLOUT, drift_vals=None, out_dir=None):
    d_dims = d_dims or D_DIMS; seeds = seeds or list(range(N_SEEDS))
    drift_vals = drift_vals or DRIFT_VALS
    out_dir = out_dir or OUT; os.makedirs(out_dir, exist_ok=True)
    print("\n=== ④ Control Utilisation ===")
    results = {}
    for d in d_dims:
        sr_all = {}
        for g in drift_vals:
            sr = {}
            for seed in seeds:
                env = _env(d, seed, "closed", g=g)
                m = _model(d, seed)
                _train(m, env, eps, stp, LR, LAMBDA, seed); m.eval()
                te = _env(d, seed + 10000, "closed", g=g)
                tr = _rollout(te, m, rsteps, seed + 20000)
                an_mean = float(np.mean(np.linalg.norm(tr["actions"], axis=1)))
                gn = _grad_norm_at(m, tr["states"], n_sample=200)
                sr[seed] = {"action_norm_mean": an_mean, "grad_norm": gn}
            agg = {k: _mean_std([s[k] for s in sr.values()]) for k in sr[list(sr.keys())[0]]}
            sr_all[f"g{g}"] = agg
        # Correlation: ||action|| vs g, ||grad|| vs g
        g_list = np.array(drift_vals, dtype=np.float64)
        an_means = np.array([sr_all[f"g{g}"]["action_norm_mean"]["mean"] for g in drift_vals])
        gn_means = np.array([sr_all[f"g{g}"]["grad_norm"]["mean"] for g in drift_vals])
        rho_an, p_an = compute_spearman_correlation(g_list, an_means)
        rho_gn, p_gn = compute_spearman_correlation(g_list, gn_means)
        sr_all["_correlation"] = {"rho_action_vs_g": float(rho_an) if not np.isnan(rho_an) else 0.0,
                                  "p_action_vs_g": float(p_an) if not np.isnan(p_an) else 1.0,
                                  "rho_grad_vs_g": float(rho_gn) if not np.isnan(rho_gn) else 0.0,
                                  "p_grad_vs_g": float(p_gn) if not np.isnan(p_gn) else 1.0}
        results[f"d{d}"] = sr_all
        print(f"  d={d}: ρ(|a|,g)={rho_an:.3f} (p={p_an:.3f})  ρ(gn,g)={rho_gn:.3f} (p={p_gn:.3f})")
    path = os.path.join(out_dir, "control_utilization.json")
    with open(path, "w") as f: json.dump(results, f, indent=2, default=_json_default)
    print(f"Saved → {path}")
    return results


# ===================================================================
# Final judgment
# ===================================================================

def compile_judgment(open_closed, perturbation, long_term, utilisation, out_dir=None):
    out_dir = out_dir or OUT; os.makedirs(out_dir, exist_ok=True)
    print("\n=== Final Judgment ===")

    verdicts = {}

    # Criterion 1: closed significantly better than open
    d_keys = list(open_closed.keys())
    if d_keys:
        d0 = d_keys[0]
        r_cl = open_closed[d0]["risk_cl_mean"]["mean"]
        r_op = open_closed[d0]["risk_op_mean"]["mean"]
        c1 = r_cl < r_op * 0.9  # closed at least 10% better
        verdicts["open_closed"] = {"pass": c1, "detail": f"risk_cl={r_cl:.3f} vs risk_op={r_op:.3f}"}
        print(f"  ① Open-Closed: {'PASS' if c1 else 'FAIL'}  ({verdicts['open_closed']['detail']})")

    # Criterion 2: recovery within reasonable time (< T/2)
    if d_keys and perturbation:
        d0k = list(perturbation[d_keys[0]].keys())
        if d0k:
            rec = perturbation[d_keys[0]][d0k[0]]["recovery_steps"]["mean"]
            c2 = rec < ROLLOUT / 4
            verdicts["perturbation"] = {"pass": c2, "detail": f"recovery={rec:.0f} steps"}
            print(f"  ② Perturbation: {'PASS' if c2 else 'FAIL'}  ({verdicts['perturbation']['detail']})")

    # Criterion 3: no long-term degradation
    if d_keys and long_term:
        ratio = long_term[d_keys[0]]["risk_ratio_2nd_1st"]["mean"]
        slope = long_term[d_keys[0]]["risk_slope"]["mean"]
        c3 = ratio < 1.1 and slope < 0.0005
        verdicts["long_term"] = {"pass": c3, "detail": f"2nd/1st={ratio:.3f} slope={slope:.5f}"}
        print(f"  ③ Long-Term: {'PASS' if c3 else 'FAIL'}  ({verdicts['long_term']['detail']})")

    # Criterion 4: action + gradient stay active
    if d_keys and utilisation:
        u = utilisation[d_keys[0]]
        rho = u["_correlation"]["rho_action_vs_g"]
        gn_list = [u[f"g{g}"]["grad_norm"]["mean"] for g in DRIFT_VALS]
        gn_min = min(gn_list)
        c4 = abs(rho) > 0.1 and gn_min > 1e-3
        verdicts["utilisation"] = {"pass": c4, "detail": f"ρ(|a|,g)={rho:.3f} min_grad={gn_min:.4f}"}
        print(f"  ④ Utilisation: {'PASS' if c4 else 'FAIL'}  ({verdicts['utilisation']['detail']})")

    all_pass = all(v["pass"] for v in verdicts.values())
    final = {"verdicts": verdicts, "all_pass": all_pass,
             "conclusion": ("Layer 1 ROBUSTNESS CONFIRMED: control is necessary, not just present."
                            if all_pass else
                            "Layer 1 ROBUSTNESS FAILED: see individual criteria.")}
    path = os.path.join(out_dir, "layer1_final_judgment.json")
    with open(path, "w") as f: json.dump(final, f, indent=2, default=_json_default)
    print(f"\n{'='*50}\nFINAL: {final['conclusion']}\n{'='*50}")
    print(f"Saved → {path}")
    return final


# ===================================================================
# CLI
# ===================================================================

def main():
    p = argparse.ArgumentParser(description="Layer 1 Robustness Experiments")
    p.add_argument("--d-dims", type=int, nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    dd = args.d_dims or D_DIMS
    sd = list(range(args.seeds))
    od = args.out or OUT
    if args.quick:
        dd = [2, 10]; sd = [0, 1, 2]; eps = 2; stp = 400; rst = 400; lst = 1000
    else:
        eps, stp, rst, lst = args.episodes, args.steps, ROLLOUT, LONG_STEPS

    r_oc = run_open_closed_comparison(dd, sd, eps, stp, rst, od)
    r_pr = run_perturbation_recovery(dd, sd, eps, stp, rst, od)
    r_lt = run_long_term_stability(dd, sd, eps, stp, lst, od)
    r_cu = run_control_utilisation(dd, sd, eps, stp, rst, None, od)

    compile_judgment(r_oc, r_pr, r_lt, r_cu, od)


if __name__ == "__main__":
    main()
