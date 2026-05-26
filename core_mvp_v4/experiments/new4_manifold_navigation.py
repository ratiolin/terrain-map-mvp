"""New4: Manifold Navigation — hidden state low-dimensional structure.

UMAP of h-space, constructs control paths along manifold directions.
Maps manifold position → behavioral output.
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


def _action_change(model, h_orig, h_new):
    a_orig = model.actor(torch.from_numpy(h_orig.astype(np.float32)).unsqueeze(0))
    a_orig = a_orig.squeeze(0).detach().numpy()
    a_new = model.actor(torch.from_numpy(h_new.astype(np.float32)).unsqueeze(0))
    a_new = a_new.squeeze(0).detach().numpy()
    return float(np.linalg.norm(a_new - a_orig))


def run_new4_manifold_navigation(n_seeds=8, d=16, k=2, hd=192,
                                 n_episodes=3, episode_length=2000, n_samples=1000):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        H_all = []; S_all = []; A_all = []
        for _ in range(n_samples):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            a = model.act_numpy(s)
            H_all.append(h)
            S_all.append(s)
            A_all.append(a)
            env.step(a)

        H_arr = np.array(H_all)
        S_arr = np.array(S_all)
        A_arr = np.array(A_all)

        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed)
            H_2d = reducer.fit_transform(H_arr)
            has_umap = True
        except ImportError:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            H_2d = pca.fit_transform(H_arr)
            has_umap = False

        pca_full = PCA(n_components=min(10, H_arr.shape[1]))
        pca_full.fit(H_arr)
        direction_effects = {}

        for i in range(min(5, pca_full.components_.shape[0])):
            pc_dir = pca_full.components_[i]
            pc_dir = pc_dir / (np.linalg.norm(pc_dir) + 1e-8)

            changes = []
            for h in H_arr[::5]:
                h_pert = h + 0.1 * pc_dir
                changes.append(_action_change(model, h, h_pert))
            direction_effects[f"PC{i+1}"] = {
                "mean_action_change": float(np.mean(changes)),
                "std_action_change": float(np.std(changes)),
            }

        random_changes = []
        for h in H_arr[::5]:
            r_dir = np.random.randn(hd)
            r_dir /= np.linalg.norm(r_dir)
            h_pert = h + 0.1 * r_dir
            random_changes.append(_action_change(model, h, h_pert))
        random_effect = float(np.mean(random_changes))

        max_pc_key = max(direction_effects, key=lambda k: direction_effects[k]["mean_action_change"])
        max_pc_effect = direction_effects[max_pc_key]["mean_action_change"]

        behavior_map = {}
        for i, h2d in enumerate(H_2d):
            grid_x = int(np.clip(h2d[0] / 0.5, -20, 20))
            grid_y = int(np.clip(h2d[1] / 0.5, -20, 20))
            key = f"{grid_x},{grid_y}"
            if key not in behavior_map:
                behavior_map[key] = []
            behavior_map[key].append(float(np.linalg.norm(A_arr[i])))

        behavior_grid = {}
        for key, vals in behavior_map.items():
            behavior_grid[key] = float(np.mean(vals))

        results["seeds"].append({
            "seed": seed,
            "has_umap": has_umap,
            "direction_effects": direction_effects,
            "random_effect": random_effect,
            "max_pc_key": max_pc_key,
            "max_pc_effect": max_pc_effect,
            "pc_vs_random_ratio": max_pc_effect / (random_effect + 1e-6),
            "n_behavior_grid_cells": len(behavior_grid),
            "behavior_grid_mean": float(np.mean(list(behavior_grid.values()))),
            "behavior_grid_std": float(np.std(list(behavior_grid.values()))),
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    ratios = [s["pc_vs_random_ratio"] for s in results["seeds"]]
    ratio_m = float(np.mean(ratios))
    grid_stds = [s["behavior_grid_std"] for s in results["seeds"]]
    grid_means = [s["behavior_grid_mean"] for s in results["seeds"]]
    cv = float(np.mean(grid_stds)) / (float(np.mean(grid_means)) + 1e-6)

    if ratio_m > 3.0:
        conclusion = f"MANIFOLD CONTROL PATHS: PC directions {ratio_m:.1f}x more effective than random."
    elif ratio_m > 1.5 and cv > 0.2:
        conclusion = f"SPATIAL STRUCTURE: manifold has behavioral variation (CV={cv:.2f})."
    else:
        conclusion = f"UNIFORM MANIFOLD: PC={ratio_m:.1f}x random, CV={cv:.2f}. Behavior is spatially uniform."

    return {"pc_random_ratio": ratio_m, "behavior_cv": cv, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_new4_manifold_navigation(n_seeds=8)
    with open("core_mvp_v4/results/new4_manifold_navigation.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("New4:", r["analysis"]["conclusion"])
