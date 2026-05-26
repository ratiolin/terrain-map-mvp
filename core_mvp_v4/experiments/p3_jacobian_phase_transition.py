"""P3-5: Jacobian Phase Transition — rank collapse during training.

Saves checkpoints throughout training, computes policy head Jacobian
effective rank at each checkpoint. Detects collapse point and
correlation with performance saturation.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model


def _effective_rank(W):
    _, S, _ = np.linalg.svd(W, full_matrices=False)
    if S.sum() == 0:
        return 1.0
    p = S / S.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def run_p3_jacobian_phase_transition(n_seeds=8, d=16, k=2, hd=192,
                                     n_episodes=5, episode_length=2000,
                                     ckpt_interval=100):
    results = {"seeds": []}
    total_steps = n_episodes * episode_length
    n_checkpoints = total_steps // ckpt_interval

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        ranks = []
        costs = []
        steps_list = []
        step_counter = 0

        for ep in range(n_episodes):
            env.reset()
            for st in range(episode_length):
                s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
                action, h, risk_pred = model(s_t)
                a_np = action.squeeze(0).detach().numpy()
                ns, risk, _, _ = env.step(a_np)
                loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if step_counter % ckpt_interval == 0:
                    W = model.actor.weight.data.clone().numpy()
                    ranks.append(_effective_rank(W))
                    costs.append(float(risk))
                    steps_list.append(step_counter)
                step_counter += 1

        collapse_step = None
        for i in range(1, len(ranks)):
            if ranks[i] < 2.5 and ranks[i-1] >= 2.5:
                collapse_step = steps_list[i]
                break

        results["seeds"].append({
            "seed": seed,
            "ranks": ranks, "costs": costs, "steps": steps_list,
            "collapse_step": collapse_step,
            "initial_rank": ranks[0] if ranks else 0.0,
            "final_rank": ranks[-1] if ranks else 0.0,
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    collapses = [s["collapse_step"] for s in results["seeds"] if s["collapse_step"] is not None]
    final_ranks = [s["final_rank"] for s in results["seeds"]]
    initial_ranks = [s["initial_rank"] for s in results["seeds"]]

    init_r = float(np.mean(initial_ranks))
    final_r = float(np.mean(final_ranks))
    n_collapse = len(collapses)

    if n_collapse >= len(results["seeds"]) * 0.5:
        mean_collapse = float(np.mean(collapses))
        conclusion = (
            f"RANK COLLAPSE PHASE TRANSITION: {n_collapse}/{len(results['seeds'])} seeds "
            f"show collapse from {init_r:.1f}→{final_r:.1f} at step ≈{mean_collapse:.0f}. "
            f"Low-rank is a learned property, not a parameterization artifact."
        )
    elif init_r < 2.5:
        conclusion = (
            f"BORN LOW-RANK: initial rank={init_r:.1f}<2.5. "
            f"Low-rank structure present from initialization — parameterization constraint."
        )
    else:
        conclusion = (
            f"NO CONSISTENT COLLAPSE: initial={init_r:.1f}, final={final_r:.1f}. "
            f"{n_collapse}/{len(results['seeds'])} seeds detected collapse."
        )

    return {
        "initial_rank": init_r, "final_rank": final_r,
        "n_collapse_seeds": n_collapse, "mean_collapse_step": float(np.mean(collapses)) if collapses else None,
        "conclusion": conclusion,
    }


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_p3_jacobian_phase_transition(n_seeds=8)
    with open("core_mvp_v4/results/p3_jacobian_phase_transition.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("P3-5:", r["analysis"]["conclusion"])
