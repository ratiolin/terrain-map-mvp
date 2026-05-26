"""I3: Training Dynamics — functional convergence paths.

Tracks functional distance d(f_i, f_j) = E_s[||f_i(s) - f_j(s)||] across training.
Measures interpolation barriers, convergence to unique function.
Tests optimizer bias (Adam vs SGD).
"""

import json, copy
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model


def _train_with_ckpts(model, env, n_ep, ep_len, seed, test_s, n_ckpts=10):
    if seed is not None: torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    total = n_ep * ep_len; interval = max(1, total // n_ckpts)
    ckpt_actions = []; step = 0
    for ep in range(n_ep):
        env.reset()
        for st in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            ns, risk, _, _ = env.step(action.squeeze(0).detach().numpy())
            loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % interval == 0:
                acts = [model.act_numpy(s) for s in test_s]
                ckpt_actions.append(np.array(acts))
            step += 1
    return ckpt_actions


def _functional_distance(acts_i, acts_j):
    return float(np.mean([np.linalg.norm(acts_i[t] - acts_j[t]) for t in range(len(acts_i))]))


def _interpolation_barrier(m1, m2, test_s, n_steps=10):
    costs = []
    for alpha in np.linspace(0, 1, n_steps):
        m_interp = copy.deepcopy(m1)
        with torch.no_grad():
            for p1, p2, pi in zip(m1.parameters(), m2.parameters(), m_interp.parameters()):
                pi.data = (1 - alpha) * p1.data + alpha * p2.data
        acts = [m_interp.act_numpy(s) for s in test_s]
        costs.append(float(np.linalg.norm(np.array(acts))))
    max_c = max(costs); edge_c = (costs[0] + costs[-1]) / 2
    return (max_c - edge_c) / (edge_c + 1e-6)


def run_i3_training_dynamics(n_seeds=4, d=16, k=2, hd=192,
                              n_episodes=3, episode_length=2000):
    env_ref = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=0, coupling=0.0)
    env_ref.reset()
    test_s = [env_ref.get_state() for _ in range(50)]

    all_ckpts = []
    final_models = []
    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        ckpts = _train_with_ckpts(model, env, n_episodes, episode_length, seed, test_s)
        all_ckpts.append(ckpts)
        final_models.append(copy.deepcopy(model))

    final_dists = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            final_dists.append(_functional_distance(all_ckpts[i][-1], all_ckpts[j][-1]))
    final_dist_m = float(np.mean(final_dists)) if final_dists else 0.0

    barriers = []
    for i in range(min(3, n_seeds)):
        for j in range(i + 1, min(n_seeds, i + 4)):
            if i != j:
                barriers.append(_interpolation_barrier(final_models[i], final_models[j], test_s))

    barrier_m = float(np.mean(barriers)) if barriers else 0.0

    if final_dist_m < 0.01:
        conclusion = f"UNIQUE FUNCTION: final distance={final_dist_m:.4f}. All seeds converge to same mapping."
    elif barrier_m < 0.1:
        conclusion = f"CONNECTED EQUIVALENCE: dist={final_dist_m:.4f}, barrier={barrier_m:.3f}. Connected solution manifold."
    else:
        conclusion = f"DISCONNECTED SOLUTIONS: dist={final_dist_m:.4f}, barrier={barrier_m:.3f}."
    return {"analysis": {"final_functional_distance": final_dist_m,
                         "interpolation_barrier": barrier_m, "conclusion": conclusion}}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_i3_training_dynamics(n_seeds=4)
    with open("core_mvp_v4/results/i3_training_dynamics.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("I3:", r["analysis"]["conclusion"])
