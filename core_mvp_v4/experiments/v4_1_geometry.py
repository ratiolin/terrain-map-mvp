"""V4-1: Gradient Field Geometry.

Quantifies: smoothness, Lipschitz constant, Hessian spectrum, critical points.
PCA visualization of vector field with cost contours.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
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
    ns = env.forward_static(s0, a)
    return float(np.linalg.norm(ns[:2]))


def _grad_h(model, env, h, s0, eps=0.01):
    d = len(h)
    g = np.zeros(d)
    for _ in range(10):
        delta = np.random.randn(d) * eps / np.sqrt(d)
        cp = _cost_of_h(model, env, h + delta, s0)
        cm = _cost_of_h(model, env, h - delta, s0)
        g += (cp - cm) * delta / (20 * eps**2 / d)
    norm = np.linalg.norm(g)
    if norm > 1e-8: g = g / norm
    return g


def _finite_diff_hessian(model, env, h, s0, eps=0.05, n_dir=5):
    d = len(h)
    hess = np.zeros((d, d))
    for _ in range(n_dir):
        v = np.random.randn(d)
        v /= np.linalg.norm(v) + 1e-8
        gp = _grad_h(model, env, h + eps * v, s0)
        gm = _grad_h(model, env, h - eps * v, s0)
        hess += np.outer(gp - gm, v) / (2 * eps * n_dir)
    eigs = np.linalg.eigvalsh(hess + hess.T)
    return hess, eigs


def run_v4_1_geometry(n_seeds=4, d=16, k=2, hd=192,
                      n_episodes=3, episode_length=2000, n_points=200):
    ep_len = episode_length
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, ep_len, seed)

        env.reset()
        H_list = []; S_list = []; G_list = []; C_list = []
        for _ in range(n_points):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            g = _grad_h(model, env, h, s)
            c = _cost_of_h(model, env, h, s)
            H_list.append(h); S_list.append(s); G_list.append(g); C_list.append(c)
            env.step(model.act_numpy(s))

        H_arr = np.array(H_list); G_arr = np.array(G_list); C_arr = np.array(C_list)

        nn_model = NearestNeighbors(n_neighbors=20).fit(H_arr)
        smoothness = []
        lip_vals = []
        for i in range(n_points):
            _, nbrs = nn_model.kneighbors([H_arr[i]], n_neighbors=20)
            for j in nbrs[0][1:]:
                cos = np.dot(G_arr[i], G_arr[j]) / (np.linalg.norm(G_arr[i]) * np.linalg.norm(G_arr[j]) + 1e-8)
                smoothness.append(float(cos))
                dist = np.linalg.norm(H_arr[i] - H_arr[j])
                if dist > 1e-6:
                    lip_vals.append(np.linalg.norm(G_arr[i] - G_arr[j]) / dist)

        critical = []
        hess_ranks = []
        hess_top_eigs = []
        for i in range(0, n_points, max(1, n_points // 20)):
            g_norm = np.linalg.norm(G_arr[i])
            if g_norm < 0.1:
                _, eigs = _finite_diff_hessian(model, env, H_arr[i], S_list[i])
                n_pos = np.sum(eigs > 0.01)
                n_neg = np.sum(eigs < -0.01)
                if n_pos == 0 and n_neg > 0:
                    tp = "attractor"
                elif n_pos > 0 and n_neg == 0:
                    tp = "repeller"
                else:
                    tp = "saddle"
                critical.append({"type": tp, "cost": float(C_arr[i]),
                                 "pos_eigs": int(n_pos), "neg_eigs": int(n_neg)})
                hess_ranks.append(len(eigs) - n_pos - n_neg)
                if len(eigs) > 0:
                    hess_top_eigs.append(float(np.max(np.abs(eigs))))

        smooth_m = float(np.mean(smoothness)) if smoothness else 0.0
        lip_m = float(np.mean(lip_vals)) if lip_vals else 0.0
        lip_max = float(np.max(lip_vals)) if lip_vals else 0.0

        pca = PCA(n_components=2).fit(H_arr)
        var_ratio = pca.explained_variance_ratio_

        results["seeds"].append({
            "seed": seed,
            "smoothness_mean": smooth_m, "lip_mean": lip_m, "lip_max": lip_max,
            "n_critical": len(critical), "critical_types": critical,
            "hess_eff_rank": float(np.mean(hess_ranks)) if hess_ranks else 0.0,
            "hess_top_eig": float(np.mean(hess_top_eigs)) if hess_top_eigs else 0.0,
            "pca_var_ratio": var_ratio.tolist(),
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    smooth = float(np.mean([s["smoothness_mean"] for s in results["seeds"]]))
    lip = float(np.mean([s["lip_mean"] for s in results["seeds"]]))
    n_crit = float(np.mean([s["n_critical"] for s in results["seeds"]]))

    if smooth < 0.1 and n_crit > 0:
        conclusion = f"ROUGH FIELD: smoothness={smooth:.3f}, {n_crit:.0f} critical pts. Lip={lip:.2f}. Field is locally valid but globally rough."
    elif smooth > 0.3:
        conclusion = f"SMOOTH FIELD: smoothness={smooth:.3f}. Gradient directions coherent in h-space."
    else:
        conclusion = f"MIXED: smoothness={smooth:.3f}, {n_crit:.0f} critical points found."
    return {"smoothness": smooth, "lipschitz": lip, "n_critical": n_crit,
            "conclusion": conclusion}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_v4_1_geometry(n_seeds=4)
    with open("core_mvp_v4/results/g1_geometry.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("V4-1:", r["analysis"]["conclusion"])
