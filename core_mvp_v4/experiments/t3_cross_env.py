"""Tier 3-8: Cross-Environment Generalization.

Repeats Tier 1-1 and Tier 1-2 on triple-well environment
to verify conclusions are not toy-task specific.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import TripleWellEnv
from core_mvp_v4.models import V4Model
from core_mvp_v4.metrics import compute_k80, effective_rank, alignment


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
            loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def _da_dh(model, s, k):
    s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
    s_t.requires_grad = True
    h_t = model.encoder(s_t)
    a_t = model.actor(h_t)
    J = np.zeros((k, model.hidden_dim))
    for j in range(k):
        grad = torch.autograd.grad(a_t[0, j], h_t, retain_graph=True, allow_unused=True)[0]
        if grad is not None:
            J[j] = grad.squeeze(0).detach().numpy()
    return J


def _behavioral(model, env, n_steps=500):
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


def run_t3_cross_env(n_seeds=8, d=8, k=2, hd=96,
                     n_episodes=3, episode_length=2000):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = TripleWellEnv(d=d, k=k, drift=0.5, seed=seed)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        J_list = []
        for _ in range(300):
            s = env.get_state()
            J = _da_dh(model, s, k)
            J_list.append(J)
            a = model.act_numpy(s)
            env.step(a)

        k80s = []; eff_ranks = []
        for J in J_list:
            _, S, _ = np.linalg.svd(J, full_matrices=False)
            k80s.append(compute_k80(S))
            eff_ranks.append(effective_rank(S))

        W_original = model.actor.weight.data.clone().numpy()
        U, S, Vt = np.linalg.svd(W_original, full_matrices=False)

        base_c, base_z = _behavioral(model, env)

        S_mod = S.copy()
        S_mod[:min(2, len(S))] = 0.0
        W_mod = U @ np.diag(S_mod) @ Vt
        model.actor.weight.data = torch.from_numpy(W_mod.astype(np.float32))
        ablate_c, ablate_z = _behavioral(model, env)

        results["seeds"].append({
            "seed": seed,
            "k80_mean": float(np.mean(k80s)),
            "eff_rank_mean": float(np.mean(eff_ranks)),
            "baseline_cost": base_c,
            "ablated_cost": ablate_c,
            "cost_ratio": ablate_c / (base_c + 1e-6),
        })

    k80_m = float(np.mean([s["k80_mean"] for s in results["seeds"]]))
    ratio_m = float(np.mean([s["cost_ratio"] for s in results["seeds"]]))

    if k80_m <= 3 and ratio_m > 1.5:
        conclusion = f"CROSS-ENV VALIDATED: k80={k80_m:.1f}≤3, ablation cost×{ratio_m:.1f}. Low-rank structure + causal necessity generalize."
    elif k80_m <= 3:
        conclusion = f"PARTIAL: k80={k80_m:.1f}≤3 structure generalizes but ablation effect ×{ratio_m:.1f} (weak)."
    else:
        conclusion = f"TASK-SPECIFIC: k80={k80_m:.1f}>3 in triple-well (vs double-well)."

    results["analysis"] = {
        "k80_mean": k80_m, "cost_ratio": ratio_m, "conclusion": conclusion,
    }
    return results


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_t3_cross_env(n_seeds=8)
    with open("core_mvp_v4/results/t3_cross_env.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("T3-8:", r["analysis"]["conclusion"])
