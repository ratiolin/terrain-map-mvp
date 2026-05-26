"""H4: Subspace-Behavior Coupling Condition Search.

Sweeps drift g, action cost weight w_action, and time discount gamma
to find parameter regions where alignment_gt and behavioral cost re-couple.

Calculates Spearman correlation: rho = corr(alignment_gt, cost) per point.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, compute_jacobian, collect_controllability_data
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


def _train_eval(seed, d, k, hd, g, w_action, gamma, n_ep, ep_len):
    env = MultiDimDoubleWell(d=d, k=k, drift=g, seed=seed, coupling=0.0)
    model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
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
            loss = nn.functional.mse_loss(risk_pred, risk_t) + w_action * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

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


def run_h4_coupling(n_seeds=5, d=16, k=2, hd=192, n_episodes=3, episode_length=1000):
    g_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    w_list = [0.01, 0.05, 0.1, 0.5, 1.0]
    gamma_list = [0.9, 0.95, 0.99, 0.999]

    grid = []
    for g in g_list:
        for w in w_list:
            aligns = []; costs = []; r2s = []
            for seed in range(n_seeds):
                m = _train_eval(seed, d, k, hd, g, w, gamma_list[0], n_episodes, episode_length)
                env2 = MultiDimDoubleWell(d=d, k=k, drift=g, seed=seed+10000, coupling=0.0)
                model2 = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                opt = torch.optim.Adam(model2.parameters(), lr=1e-3)
                for _ in range(n_episodes):
                    env2.reset()
                    for __ in range(episode_length):
                        s_t = torch.from_numpy(env2.get_state().astype(np.float32)).unsqueeze(0)
                        action, h, risk_pred = model2(s_t)
                        a_np = action.squeeze(0).detach().numpy()
                        ns, risk, _, _ = env2.step(a_np)
                        risk_t = torch.tensor([[risk]], dtype=torch.float32)
                        loss = nn.functional.mse_loss(risk_pred, risk_t) + w * torch.mean(action**2)
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
                        opt.step()
                c, _ = _behavioral(model2, env2, d, k)
                aligns.append(m["align"]); costs.append(c); r2s.append(m["R2"])

            if len(aligns) > 2:
                rho, pval = spearmanr(aligns, costs)
                rho_r2, pval_r2 = spearmanr(aligns, r2s)
            else:
                rho, pval = 0.0, 1.0
                rho_r2, pval_r2 = 0.0, 1.0

            grid.append({
                "g": g, "w_action": w,
                "align_mean": float(np.mean(aligns)), "cost_mean": float(np.mean(costs)),
                "rho_align_cost": float(rho), "p_align_cost": float(pval),
                "rho_align_R2": float(rho_r2), "p_align_R2": float(pval_r2),
            })

    best = max(grid, key=lambda x: abs(x["rho_align_cost"]))
    if abs(best["rho_align_cost"]) > 0.5 and best["p_align_cost"] < 0.05:
        conclusion = (f"COUPLING WINDOW: g={best['g']}, w={best['w_action']}, "
                      f"rho={best['rho_align_cost']:.3f} p={best['p_align_cost']:.4f}")
    else:
        conclusion = f"GLOBAL DECOUPLING: max |rho|={abs(best['rho_align_cost']):.3f} < 0.5"

    return {"grid": grid, "best_point": best, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_h4_coupling(n_seeds=5)
    with open("core_mvp_v4/results/h4_coupling_phase.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("H4:", r["conclusion"])
