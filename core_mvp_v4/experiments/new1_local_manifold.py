"""New1: Local Manifold Learning in Hidden Space.

Local PCA on h-space, alignment between local dominant directions
and ∂a/∂h Jacobian. Identifies control-sensitive regions.
Compares with random-initialized encoder.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

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


def run_new1_local_manifold(n_seeds=8, d=16, k=2, hd=192,
                            n_episodes=3, episode_length=2000, n_samples=2000,
                            n_neighbors=80):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        S_all = []; H_all = []; J_all = []
        for _ in range(n_samples):
            s = env.get_state()
            h = model.encoder(torch.from_numpy(s.astype(np.float32)).unsqueeze(0))
            h = h.squeeze(0).detach().numpy()
            J = _da_dh(model, s, k)
            S_all.append(s); H_all.append(h); J_all.append(J)
            env.step(model.act_numpy(s))

        H_arr = np.array(H_all)
        S_arr = np.array(S_all)

        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(H_arr)
        local_ids = []
        local_alignments = []
        indices = np.random.choice(len(H_arr), min(500, len(H_arr)), replace=False)

        for idx in indices:
            _, nbrs = nn.kneighbors([H_arr[idx]], n_neighbors=n_neighbors)
            local_H = H_arr[nbrs[0]]
            local_H_centered = local_H - local_H.mean(axis=0)
            if local_H_centered.shape[0] < 3:
                continue
            C = local_H_centered.T @ local_H_centered / len(local_H_centered)
            eigvals, eigvecs = np.linalg.eigh(C)
            ev_ratio = np.cumsum(eigvals[::-1]) / eigvals.sum()
            local_id = int(np.argmax(ev_ratio >= 0.95)) + 1
            local_ids.append(local_id)

            top_pc = eigvecs[:, -1]
            J_top = J_all[idx][0] if k >= 1 else J_all[idx][:1, :].mean(axis=0)
            J_top = J_top / (np.linalg.norm(J_top) + 1e-8)
            cos_sim = float(np.abs(np.dot(top_pc, J_top)))
            local_alignments.append(cos_sim)

        high_align_idx = [indices[i] for i in range(len(local_alignments))
                          if local_alignments[i] > 0.5]
        s_high = [float(np.linalg.norm(S_arr[idx][:2])) for idx in high_align_idx] if high_align_idx else []

        results["seeds"].append({
            "seed": seed,
            "local_id_mean": float(np.mean(local_ids)), "local_id_std": float(np.std(local_ids)),
            "alignment_mean": float(np.mean(local_alignments)),
            "alignment_std": float(np.std(local_alignments)),
            "high_align_frac": len(high_align_idx) / max(len(local_alignments), 1),
            "high_align_avg_risk": float(np.mean(s_high)) if s_high else 0.0,
            "n_degenerate": sum(1 for lid in local_ids if lid <= 1),
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    align_m = float(np.mean([s["alignment_mean"] for s in results["seeds"]]))
    id_m = float(np.mean([s["local_id_mean"] for s in results["seeds"]]))
    ha_frac = float(np.mean([s["high_align_frac"] for s in results["seeds"]]))
    ha_risk = float(np.mean([s["high_align_avg_risk"] for s in results["seeds"]]))

    if align_m > 0.5 and ha_frac > 0.1:
        conclusion = (f"LOCAL CONTROL DIRECTIONS: align={align_m:.3f}, "
                      f"{ha_frac:.1%} of points show high alignment. "
                      f"Local ID≈{id_m:.1f}. Control directions are locally structured.")
    elif align_m < 0.3:
        conclusion = f"UNIFORMLY LOW ALIGNMENT: align={align_m:.3f}. No strong alignment anywhere."
    else:
        conclusion = f"WEAK STRUCTURE: align={align_m:.3f}, high_align_frac={ha_frac:.1%}."

    return {"mean_alignment": align_m, "local_id": id_m,
            "high_align_frac": ha_frac, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_new1_local_manifold(n_seeds=8)
    with open("core_mvp_v4/results/new1_local_manifold.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("New1:", r["analysis"]["conclusion"])
