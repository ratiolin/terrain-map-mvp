"""Tier 3-7: NTK Analysis — local linearity of control.

Computes Neural Tangent Kernel of policy head around operating points.
Tests whether the controller behaves as a locally linear system.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model


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


def _compute_ntk(model, states, k):
    """Approximate NTK: outer product of Jacobians ∂a/∂θ."""
    n = len(states)
    all_grads = []
    for s in states:
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        s_t.requires_grad = True
        model.zero_grad()
        action, h, _ = model(s_t)
        grads = []
        for j in range(k):
            grad_j = torch.autograd.grad(action[0, j], model.parameters(),
                                         retain_graph=True, allow_unused=True)
            flat = torch.cat([g.flatten() for g in grad_j if g is not None])
            grads.append(flat.detach().numpy())
        all_grads.append(np.mean(grads, axis=0))

    G = np.array(all_grads)
    ntk = G @ G.T / n
    return ntk


def run_t3_ntk(n_seeds=8, d=16, k=2, hd=192,
               n_episodes=3, episode_length=2000, n_op=50):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        op_states = []
        for _ in range(n_op):
            s = env.get_state()
            op_states.append(s)
            a = model.act_numpy(s)
            env.step(a)

        ntk = _compute_ntk(model, op_states, k)
        eigvals = np.linalg.eigvalsh(ntk)
        eigvals = eigvals[eigvals > 0]
        effective_rank_ntk = float(np.exp(-np.sum((eigvals / eigvals.sum()) * np.log(eigvals / eigvals.sum() + 1e-10))))

        results["seeds"].append({
            "seed": seed,
            "ntk_eff_rank": effective_rank_ntk,
            "ntk_rank": int(np.sum(eigvals > 1e-3 * eigvals.max())),
            "ntk_max_eig": float(eigvals.max()) if len(eigvals) > 0 else 0.0,
        })

    eff_ranks = [s["ntk_eff_rank"] for s in results["seeds"]]
    ranks = [s["ntk_rank"] for s in results["seeds"]]

    eff_mean = float(np.mean(eff_ranks))
    rank_mean = float(np.mean(ranks))

    if eff_mean < 3:
        conclusion = f"LOCALLY LINEAR: NTK eff_rank={eff_mean:.1f} → controller is low-rank linear."
    elif eff_mean < 10:
        conclusion = f"MODERATELY LINEAR: NTK eff_rank={eff_mean:.1f}, rank={rank_mean:.1f}."
    else:
        conclusion = f"NON-LINEAR: NTK eff_rank={eff_mean:.1f} → controller uses nonlinear transformations."

    results["analysis"] = {
        "ntk_eff_rank": eff_mean, "ntk_rank": rank_mean,
        "conclusion": conclusion,
    }
    return results


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_t3_ntk(n_seeds=8)
    with open("core_mvp_v4/results/t3_ntk.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("T3-7:", r["analysis"]["conclusion"])
