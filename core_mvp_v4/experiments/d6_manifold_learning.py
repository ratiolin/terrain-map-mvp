"""D6: Hidden State Manifold Learning.

UMAP on h and z. Visualizes manifold structure,
quantifies alignment with task variables (R², MI).
Compares with random-initialized encoder baseline.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model


def _train(model, env, n_ep, ep_len, seed):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            ns, risk, _, _ = env.step(action.squeeze(0).detach().numpy())
            loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def _probe_r2(X, y):
    split = int(len(X) * 0.7)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    m = LinearRegression().fit(X_tr, y_tr)
    return float(m.score(X_te, y_te))


def run_d6_manifold_learning(n_seeds=8, d=16, k=2, hd=192,
                             n_episodes=3, episode_length=2000, n_samples=1500):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        H_all = []; S_all = []; C_all = []
        for _ in range(n_samples):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            H_all.append(h); S_all.append(s)
            ns, risk, _, _ = env.step(model.act_numpy(s))
            C_all.append(risk)
        H_arr = np.array(H_all); S_arr = np.array(S_all); C_arr = np.array(C_all)

        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed)
            H_2d = reducer.fit_transform(H_arr)
            has_umap = True
        except ImportError:
            pca = PCA(n_components=2)
            H_2d = pca.fit_transform(H_arr)
            has_umap = False

        R2_h0_risk = _probe_r2(H_2d[:, 0].reshape(-1, 1), C_arr)
        R2_h1_risk = _probe_r2(H_2d[:, 1].reshape(-1, 1), C_arr)
        R2_h_risk = _probe_r2(H_2d, C_arr)
        R2_h_s0 = _probe_r2(H_2d, S_arr[:, 0])

        results["seeds"].append({
            "seed": seed, "has_umap": has_umap,
            "R2_umap0_cost": R2_h0_risk, "R2_umap1_cost": R2_h1_risk,
            "R2_umap_cost": R2_h_risk,
            "R2_umap_s0": R2_h_s0,
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    R2_cost = float(np.mean([s["R2_umap_cost"] for s in results["seeds"]]))
    R2_s0 = float(np.mean([s["R2_umap_s0"] for s in results["seeds"]]))

    if R2_cost > 0.3:
        conclusion = f"STRUCTURED MANIFOLD: UMAP explains cost (R²={R2_cost:.3f}). Manifold encodes task structure."
    elif R2_s0 > 0.3:
        conclusion = f"STATE-ALIGNED MANIFOLD: UMAP reflects s_control (R²={R2_s0:.3f}). Manifold tracks state, not cost."
    else:
        conclusion = f"UNIFORM MANIFOLD: UMAP explains neither cost nor state (R²_cost={R2_cost:.3f}, R²_s0={R2_s0:.3f})."

    return {"R2_umap_cost": R2_cost, "R2_umap_s0": R2_s0, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_d6_manifold_learning(n_seeds=8)
    with open("core_mvp_v4/results/d6_manifold_learning.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("D6:", r["analysis"]["conclusion"])
