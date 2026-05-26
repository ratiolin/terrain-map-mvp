"""P2-4: Rank Origin — why rank=2?

A: Task geometry — intrinsic dimension of s_control manifold.
B: Action dim constraint — sweep action_dim, measure effective rank.
C: Optimization trajectory — rank evolution during training.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from sklearn.decomposition import PCA


def _train_full(model, env, n_ep, ep_len, seed, checkpoint_fn=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpts = []
    for ep in range(n_ep):
        env.reset()
        for step in range(ep_len):
            if checkpoint_fn and step % (ep_len // 10) == 0:
                ckpts.append(checkpoint_fn())
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            ns, risk, _, _ = env.step(a_np)
            loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return ckpts


def _effective_rank(W):
    _, S, _ = np.linalg.svd(W, full_matrices=False)
    if S.sum() == 0:
        return 1.0
    p = S / S.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def run_p2_rank_origin(n_seeds=8, d=16, k=2, hd=192,
                       n_episodes=3, episode_length=2000):
    results = {}

    # A: Task geometry — ID of s_control
    env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=0, coupling=0.0)
    env.reset()
    trajectories = []
    for _ in range(2000):
        trajectories.append(env.state[:k].copy())
        env.step(np.zeros(k))
    traj = np.array(trajectories)
    pca = PCA().fit(traj)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    id_90 = int(np.argmax(cumvar >= 0.90)) + 1
    id_95 = int(np.argmax(cumvar >= 0.95)) + 1
    results["A_task_geometry"] = {
        "ID_90pct": id_90, "ID_95pct": id_95,
        "explained_var_ratio": pca.explained_variance_ratio_.tolist(),
        "note": f"s_control intrinsic dimension ≈ {id_90}-{id_95}",
    }

    # B: Action dim sweep
    from core_mvp_v4.models import V4Model
    action_dims = [1, 2, 4, 8]
    b_results = {}
    for ad in action_dims:
        eff_ranks = []
        for seed in range(min(n_seeds, 4)):
            env_b = MultiDimDoubleWell(d=d, k=min(ad, k), drift=0.5, seed=seed, coupling=0.0)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=ad)
            _train_full(model, env_b, n_episodes, episode_length, seed)
            W = model.actor.weight.data.numpy()
            eff_ranks.append(_effective_rank(W))
        b_results[f"action_dim_{ad}"] = {
            "eff_rank_mean": float(np.mean(eff_ranks)),
            "eff_rank_std": float(np.std(eff_ranks)),
        }
    results["B_action_dim_sweep"] = b_results

    # C: Optimization trajectory (rank collapse)
    seed = 0
    env_c = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
    model_c = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)

    def ckpt_fn():
        W = model_c.actor.weight.data.clone().numpy()
        return _effective_rank(W)

    ckpts = _train_full(model_c, env_c, n_episodes, episode_length, seed, checkpoint_fn=ckpt_fn)
    results["C_optimization_trajectory"] = {
        "effective_ranks": ckpts,
        "initial_rank": ckpts[0] if ckpts else 0,
        "final_rank": ckpts[-1] if ckpts else 0,
    }

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    id_val = results["A_task_geometry"]["ID_90pct"]
    b_vals = results["B_action_dim_sweep"]
    c_ranks = results["C_optimization_trajectory"]["effective_ranks"]

    parts = [f"Task manifold ID≈{id_val}"]
    if id_val == 2 or id_val == 1:
        parts.append("→ rank=2 matches s_control intrinsic dimension.")

    follows_ad = all(
        abs(v["eff_rank_mean"] - float(ad)) < 1.0
        for ad_str, v in b_vals.items()
        for ad in [int(ad_str.split("_")[-1])]
    )
    if follows_ad:
        parts.append("Rank follows action_dim → parameterization constraint.")

    if c_ranks and c_ranks[0] > c_ranks[-1] * 1.3:
        parts.append(f"Rank collapse: {c_ranks[0]:.1f}→{c_ranks[-1]:.1f} → optimization selects low-rank.")

    return {"id_90": id_val, "conclusion": " ".join(parts)}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_p2_rank_origin(n_seeds=8)
    with open("core_mvp_v4/results/p2_rank_origin.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("P2-4:", r["analysis"]["conclusion"])
