"""V4-4: Gradient Field Topology-Task Alignment.

Maps critical points to task events.
Tests if attractors → low-cost zones, saddles → decision boundaries.
"""

import json
import numpy as np
import torch; import torch.nn as nn
from sklearn.decomposition import PCA

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model


def _train(model, env, n_ep, ep_len, seed):
    if seed is not None: torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            ns, risk, _, _ = env.step(action.squeeze(0).detach().numpy())
            loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def _cost_of_h(model, env, h, s0):
    a = model.actor(torch.from_numpy(h.astype(np.float32)).unsqueeze(0)).squeeze(0).detach().numpy()
    return float(np.linalg.norm(env.forward_static(s0, a)[:2]))


def _grad_h(model, env, h, s0, eps=0.01):
    d = len(h); g = np.zeros(d)
    for _ in range(10):
        delta = np.random.randn(d) * eps / np.sqrt(d)
        g += (_cost_of_h(model, env, h + delta, s0) - _cost_of_h(model, env, h - delta, s0)) * delta / (20 * eps**2 / d)
    norm = np.linalg.norm(g)
    return g / norm if norm > 1e-8 else g


def run_v4_4_topology(n_seeds=4, d=16, k=2, hd=192, n_episodes=3, episode_length=2000, n_points=300):
    ep_len = episode_length
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, ep_len, seed)

        env.reset()
        H_list, S_list, G_list, C_list, D_list = [], [], [], [], []
        for _ in range(n_points):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            g = _grad_h(model, env, h, s)
            c = _cost_of_h(model, env, h, s)
            dr = env.drift
            H_list.append(h); S_list.append(s); G_list.append(g); C_list.append(c)
            D_list.append(dr); env.step(model.act_numpy(s))

        G_arr = np.array(G_list); C_arr = np.array(C_list); S_arr = np.array(S_list)
        g_norms = np.linalg.norm(G_arr, axis=1)
        crit_mask = g_norms < 0.1
        n_crit = int(np.sum(crit_mask))

        if n_crit > 0:
            crit_costs = C_arr[crit_mask]
            crit_s0 = S_arr[crit_mask, 0]
            low_cost_idx = C_arr < np.median(C_arr)
            low_cost_s0 = S_arr[low_cost_idx, 0]

            # Null hypothesis: critical points have same cost distribution as all points
            base_median = np.median(C_arr)
            crit_below = float(np.mean(crit_costs < base_median))
            crit_s0_mean = float(np.mean(crit_s0))

            # Check if critical points cluster near s=0 (low risk) or s=±1.5 (barriers)
            near_barrier = float(np.mean((np.abs(crit_s0) > 1.0) & (np.abs(crit_s0) < 2.0)))
        else:
            crit_below = 0.5; crit_s0_mean = 0.0; near_barrier = 0.0

        results["seeds"].append({
            "seed": seed, "n_critical": n_crit,
            "crit_cost_below_median": crit_below,
            "crit_s0_mean": crit_s0_mean,
            "crit_near_barrier_frac": near_barrier,
        })

    crit_b = float(np.mean([s["crit_cost_below_median"] for s in results["seeds"]]))
    barrier_f = float(np.mean([s["crit_near_barrier_frac"] for s in results["seeds"]]))

    if crit_b > 0.7:
        conclusion = f"ATTRACTORS AT LOW-COST: {crit_b:.0%} of critical pts below median cost. Gradient field converges to task goals."
    elif barrier_f > 0.3:
        conclusion = f"SADDLES AT BARRIERS: {barrier_f:.0%} critical pts near barriers. Field reflects decision boundaries."
    else:
        conclusion = f"NO CLEAR TOPOLOGY-TASK ALIGNMENT: crit_b={crit_b:.2%}, barrier={barrier_f:.2%}."
    results["analysis"] = {"crit_below_median": crit_b, "near_barrier_frac": barrier_f, "conclusion": conclusion}
    return results


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_v4_4_topology(n_seeds=4)
    with open("core_mvp_v4/results/g4_topology.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("V4-4:", r["analysis"]["conclusion"])
