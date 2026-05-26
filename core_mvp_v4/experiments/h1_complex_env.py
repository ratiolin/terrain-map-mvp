"""H1: Complex Environment — decoupling validation beyond double-well.

Tests triple-well potential for re-coupling of representation and behavior.
Compares Spearman rho against double-well baseline results.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

from core_mvp_v4.env import MultiDimDoubleWell, TripleWellEnv
from core_mvp_v4.models import V4Model, SplitV4Model, compute_jacobian, collect_controllability_data
from core_mvp_v4.metrics import compute_k80, alignment


def _behavioral(model, env, d, k, n_steps=500):
    env.reset()
    cost_sum = 0.0
    in_zone = 0
    for _ in range(n_steps):
        a = model.act_numpy(env.get_state())
        ns, risk, _, _ = env.step(a)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
    return cost_sum / n_steps, in_zone / n_steps


def _train(model, env, n_ep, ep_len, seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            outputs = model(s_t)
            action, h, risk_pred = outputs[0], outputs[1], outputs[2]
            a_np = action.squeeze(0).detach().numpy()
            ns, risk, _, _ = env.step(a_np)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            loss = nn.functional.mse_loss(risk_pred, risk_t) + 0.1 * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def _eval(model, env, d, k):
    test_h, test_C = collect_controllability_data(model, env, n_samples=200)
    probe = LinearRegression().fit(test_h, test_C)
    R2_val = float(probe.score(test_h, test_C))
    env.reset()
    Js = [compute_jacobian(model, env.get_state())]
    for _ in range(50):
        a = model.act_numpy(env.get_state())
        env.step(a)
        Js.append(compute_jacobian(model, env.get_state()))
    J_mean = np.mean(Js, axis=0)
    _, S_mean, Vt_mean = np.linalg.svd(J_mean, full_matrices=False)
    V = Vt_mean.T
    k80_val = compute_k80(S_mean)
    U_true = np.eye(d)[:, :k]
    k_use = min(k, V.shape[1])
    align_gt = float(alignment(V[:, :k_use], U_true[:, :k_use], k=k_use))
    return {"align": align_gt, "R2": R2_val, "k80": k80_val}


def run_h1_complex_env(n_seeds=8, d=8, k=2, hd=96, n_episodes=3, episode_length=2000):
    results = {}

    for env_label, EnvClass in [("double_well", MultiDimDoubleWell), ("triple_well", TripleWellEnv)]:
        aligns = []; costs = []; zones = []; r2s = []
        split_aligns = []; split_costs = []; split_zones = []

        for seed in range(n_seeds):
            env = EnvClass(d=d, k=k, drift=0.5, seed=seed)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
            _train(model, env, n_episodes, episode_length, seed)
            m = _eval(model, env, d, k)
            c, z = _behavioral(model, env, d, k)
            aligns.append(m["align"]); costs.append(c); zones.append(z); r2s.append(m["R2"])

            env_s = EnvClass(d=d, k=k, drift=0.5, seed=seed)
            model_s = SplitV4Model(state_dim=d, hidden_dim=hd, action_dim=k, k=k)
            _train(model_s, env_s, n_episodes, episode_length, seed)
            m = _eval(model_s, env_s, d, k)
            cs, zs = _behavioral(model_s, env_s, d, k)
            split_aligns.append(m["align"]); split_costs.append(cs); split_zones.append(zs)

        rho_cost, p_cost = spearmanr(aligns, costs)
        rho_zone, p_zone = spearmanr(aligns, zones)

        results[env_label] = {
            "baseline": {
                "align_mean": float(np.mean(aligns)), "align_std": float(np.std(aligns)),
                "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
                "zone_mean": float(np.mean(zones)), "R2_mean": float(np.mean(r2s)),
            },
            "split_encoder": {
                "align_mean": float(np.mean(split_aligns)), "align_std": float(np.std(split_aligns)),
                "cost_mean": float(np.mean(split_costs)), "zone_mean": float(np.mean(split_zones)),
            },
            "rho_align_cost": float(rho_cost), "p_align_cost": float(p_cost),
            "rho_align_zone": float(rho_zone), "p_align_zone": float(p_zone),
        }

    dw_rho = results["double_well"]["rho_align_cost"]
    tw_rho = results["triple_well"]["rho_align_cost"]

    if abs(tw_rho) > 0.5:
        conclusion = f"RECOUPLING: triple-well re-couples behavior (rho={tw_rho:.3f}). Double-well decoupling was task-specific."
    elif abs(tw_rho) < 0.3:
        conclusion = f"CROSS-TASK DECOUPLING: both environments show |rho|<0.3 (DW: {dw_rho:.3f}, TW: {tw_rho:.3f}). Subspace-behavior decoupling is cross-task."
    else:
        conclusion = f"INCONCLUSIVE: triple-well rho={tw_rho:.3f}, double-well rho={dw_rho:.3f}"

    results["conclusion"] = conclusion
    return results


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_h1_complex_env(n_seeds=8)
    with open("core_mvp_v4/results/h1_complex_env.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("H1:", r["conclusion"])
