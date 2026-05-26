"""G3: Attribution-Control Closed-Loop Validation.

Verifies that attribution bias improvement in representation translates to
real control performance gains.

Uses continuous drift environment (sinusoidal drift).
d ∈ [16, 32], hd = d*12.
Architectures: baseline, split_encoder (best), minimal_attribution (best from G2).
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

from core_mvp_v4.env import ContinuousDriftEnv
from core_mvp_v4.models import (
    V4Model, SplitV4Model,
    compute_jacobian, collect_controllability_data,
)
from core_mvp_v4.metrics import compute_k80, alignment


def _measure(model, env, d, k):
    test_h, test_C = collect_controllability_data(model, env, n_samples=300)
    probe = LinearRegression().fit(test_h, test_C)
    R2 = float(probe.score(test_h, test_C))

    env.reset()
    jac_states = [env.get_state()]
    for _ in range(100):
        a = model.act_numpy(env.get_state())
        env.step(a)
        jac_states.append(env.get_state())
    J_mean = np.mean([compute_jacobian(model, s) for s in jac_states[::3]], axis=0)
    _, S_mean, Vt_mean = np.linalg.svd(J_mean, full_matrices=False)
    V = Vt_mean.T
    k80 = compute_k80(S_mean)
    U_true = np.eye(d)[:, :k]
    k_use = min(k, V.shape[1])
    align_gt = float(alignment(V[:, :k_use], U_true[:, :k_use], k=k_use))
    return {"R2": R2, "alignment_gt": align_gt, "k80": k80}


def _behavioral(model, env, d, k, n_steps=1000):
    env.reset()
    cost_sum = 0.0
    in_zone = 0
    for _ in range(n_steps):
        a = model.act_numpy(env.get_state())
        ns, risk, _, _ = env.step(a)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
    return {
        "mean_cost": cost_sum / n_steps,
        "in_zone_rate": in_zone / n_steps,
    }


def _train(model, env, n_ep, ep_len, seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            ns, risk, _, _ = env.step(a_np)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            loss = nn.functional.mse_loss(risk_pred, risk_t) + 0.1 * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def run_g3_control_loop(n_seeds=8, k=2, n_episodes=3, episode_length=2000):
    d_list = [16, 32]
    results = {}

    for d in d_list:
        hd = d * 12
        d_results = {"baseline": [], "split_encoder": [], "minimal": []}

        for seed in range(n_seeds):
            env = ContinuousDriftEnv(d=d, k=k, seed=seed)
            env2 = ContinuousDriftEnv(d=d, k=k, seed=seed + 1000)
            m = _train_measure(V4Model(state_dim=d, hidden_dim=hd, action_dim=k),
                               env, d, k, hd, n_episodes, episode_length, seed)
            bv = _behavioral(m["model"], env2, d, k)
            d_results["baseline"].append({
                "align": m["align"], "R2": m["R2"], "k80": m["k80"],
                "cost": bv["mean_cost"], "zone": bv["in_zone_rate"],
            })

            env_s = ContinuousDriftEnv(d=d, k=k, seed=seed)
            env_s2 = ContinuousDriftEnv(d=d, k=k, seed=seed + 1000)
            m = _train_measure(SplitV4Model(state_dim=d, hidden_dim=hd, action_dim=k, k=k),
                               env_s, d, k, hd, n_episodes, episode_length, seed)
            bv = _behavioral(m["model"], env_s2, d, k)
            d_results["split_encoder"].append({
                "align": m["align"], "R2": m["R2"], "k80": m["k80"],
                "cost": bv["mean_cost"], "zone": bv["in_zone_rate"],
            })

            if seed == 0:
                print(f"  d={d} seed={seed} split_encoder done...")

        results[str(d)] = _aggregate(d_results, d)

    results["correlation_analysis"] = _correlate(results, d_list)
    return results


def _train_measure(model, env, d, k, hd, n_ep, ep_len, seed):
    _train(model, env, n_ep, ep_len, seed)
    m = _measure(model, env, d, k)
    return {"model": model, "align": m["alignment_gt"], "R2": m["R2"], "k80": m["k80"]}


def _aggregate(d_results, d):
    out = {}
    for arch, items in d_results.items():
        aligns = [i["align"] for i in items]
        r2s = [i["R2"] for i in items]
        costs = [i["cost"] for i in items]
        zones = [i["zone"] for i in items]
        out[arch] = {
            "align_mean": float(np.mean(aligns)), "align_std": float(np.std(aligns)),
            "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
            "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
            "zone_mean": float(np.mean(zones)), "zone_std": float(np.std(zones)),
        }
    return out


def _correlate(results, d_list):
    all_aligns = []
    all_costs = []
    all_zones = []

    for d in d_list:
        for arch, data in results[str(d)].items():
            if data["align_mean"] is not None:
                all_aligns.append(data["align_mean"])
                all_costs.append(data["cost_mean"])
                all_zones.append(data["zone_mean"])

    if len(all_aligns) < 3:
        return {"rho_align_cost": None, "rho_align_zone": None, "conclusion": "Insufficient data"}

    rho_cost, p_cost = spearmanr(all_aligns, all_costs)
    rho_zone, p_zone = spearmanr(all_aligns, all_zones)

    cost_sig = p_cost < 0.05
    zone_sig = p_zone < 0.05

    parts = []
    if cost_sig:
        parts.append(f"alignment-cost r={rho_cost:.3f} (p={p_cost:.4f})")
    if zone_sig:
        parts.append(f"alignment-zone r={rho_zone:.3f} (p={p_zone:.4f})")

    if cost_sig or zone_sig:
        conclusion = "CLOSED LOOP VALIDATED: " + "; ".join(parts)
    else:
        conclusion = f"NO SIGNIFICANT CORRELATION: cost r={rho_cost:.3f} p={p_cost:.4f}, zone r={rho_zone:.3f} p={p_zone:.4f}"

    return {"rho_align_cost": rho_cost, "p_cost": p_cost,
            "rho_align_zone": rho_zone, "p_zone": p_zone,
            "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_g3_control_loop(n_seeds=8)
    with open("core_mvp_v4/results/g3_control_loop.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    ca = r["correlation_analysis"]
    print("G3:", ca["conclusion"])
