"""V4-2: Gradient Field Regularization.

Trains models with smoothness/spectral constraints.
Compares gradient field quality vs baseline.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from core_mvp_v4.env import MultiDimDoubleWell


def _cost_of_h(model, env, h, s0):
    a = model.actor(torch.from_numpy(h.astype(np.float32)).unsqueeze(0)).squeeze(0).detach().numpy()
    ns = env.forward_static(s0, a)
    return float(np.linalg.norm(ns[:2]))


def _grad_h(model, env, h, s0, eps=0.01):
    d = len(h); g = np.zeros(d)
    for _ in range(10):
        delta = np.random.randn(d) * eps / np.sqrt(d)
        g += (_cost_of_h(model, env, h + delta, s0) - _cost_of_h(model, env, h - delta, s0)) * delta / (20 * eps**2 / d)
    norm = np.linalg.norm(g)
    return g / norm if norm > 1e-8 else g


def _train_regularized(model_class, env, n_ep, ep_len, seed, reg_type="baseline", reg_lambda=0.01):
    if seed is not None: torch.manual_seed(seed); np.random.seed(seed)
    from core_mvp_v4.models import V4Model
    model = V4Model(state_dim=env.d, hidden_dim=192, action_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            ns, risk, _, _ = env.step(a_np)
            task_loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
            reg_loss = 0.0
            if reg_type == "spectral":
                for p in model.encoder.parameters():
                    if p.ndim > 1:
                        reg_loss += torch.norm(p, p=2)
            loss = task_loss + reg_lambda * reg_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


def _field_quality(model, env, d, hd, n_points=80):
    env.reset()
    H, G, C = [], [], []
    for _ in range(n_points):
        s = env.get_state()
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h = model.encoder(s_t).squeeze(0).detach().numpy()
        g = _grad_h(model, env, h, s)
        c = _cost_of_h(model, env, h, s)
        H.append(h); G.append(g); C.append(c)
        env.step(model.act_numpy(s))
    H_arr = np.array(H); G_arr = np.array(G)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10).fit(H_arr)
    smooth = []
    for i in range(len(H_arr)):
        _, nbrs = nn.kneighbors([H_arr[i]], 10)
        for j in nbrs[0][1:]:
            smooth.append(float(np.dot(G[i], G[j]) / (np.linalg.norm(G[i]) * np.linalg.norm(G[j]) + 1e-8)))
    return float(np.mean(smooth)) if smooth else 0.0


def run_v4_2_regularized(n_seeds=4, d=16, k=2, hd=192, n_episodes=3, episode_length=2000):
    ep_len = episode_length
    reg_types = ["baseline", "spectral"]
    results = {}
    for reg in reg_types:
        smooths = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = _train_regularized(None, env, n_episodes, ep_len, seed, reg_type=reg)
            sm = _field_quality(model, env, d, hd)
            smooths.append(sm)
        results[reg] = {"smoothness_mean": float(np.mean(smooths)), "smoothness_std": float(np.std(smooths))}
    base_s = results["baseline"]["smoothness_mean"]
    best = max(results, key=lambda r: results[r]["smoothness_mean"])
    best_s = results[best]["smoothness_mean"]
    improvement = (best_s / (base_s + 1e-6) - 1) * 100
    conclusion = f"{best} improves smoothness by {improvement:.0f}% vs baseline ({base_s:.3f}→{best_s:.3f})."
    results["analysis"] = {"improvement_pct": improvement, "conclusion": conclusion}
    return results


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_v4_2_regularized(n_seeds=4)
    with open("core_mvp_v4/results/g2_regularized.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("V4-2:", r["analysis"]["conclusion"])
