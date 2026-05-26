"""Tier 2-4: Trajectory Jacobian Dynamics — state-dependent structure.

Sliding window SVD on J=∂a/∂h along trajectory.
Measures window-to-window alignment and grouping by environmental conditions.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model
from core_mvp_v4.metrics import compute_k80, alignment


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


def run_t2_dynamic_jacobian(n_seeds=8, d=16, k=2, hd=192,
                            n_episodes=3, episode_length=2000, traj_len=5000):
    window = 500
    stride = 250
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        J_history = []
        s_history = []
        drift_history = []
        for t in range(traj_len):
            s = env.get_state()
            J = _da_dh(model, s, k)
            J_history.append(J)
            s_history.append(s.copy())
            drift_history.append(env.drift)
            a = model.act_numpy(s)
            env.step(a)

        window_alignments = []
        prev_V = None
        for i in range(0, traj_len - window, stride):
            J_win = np.mean(J_history[i:i + window], axis=0)
            _, S, Vt = np.linalg.svd(J_win, full_matrices=False)
            k80 = compute_k80(S)
            V_now = Vt.T[:, :k80] if k80 > 0 else Vt.T
            if prev_V is not None:
                k_use = min(k80, prev_V.shape[1])
                if k_use > 0:
                    window_alignments.append(float(alignment(V_now[:, :k_use], prev_V[:, :k_use], k=k_use)))
            prev_V = V_now.copy()

        center_positions = []
        for i in range(0, traj_len - window, stride):
            s_win = np.mean(s_history[i:i + window], axis=0)
            center_positions.append(s_win[:k])

        low_risk_idx = [i for i, s in enumerate(center_positions) if np.linalg.norm(s) < 1.0]
        high_risk_idx = [i for i, s in enumerate(center_positions) if np.linalg.norm(s) >= 1.0]

        low_aligns = [window_alignments[i] for i in low_risk_idx if i < len(window_alignments)]
        high_aligns = [window_alignments[i] for i in high_risk_idx if i < len(window_alignments)]

        results["seeds"].append({
            "seed": seed,
            "mean_alignment": float(np.mean(window_alignments)) if window_alignments else 0.0,
            "min_alignment": float(np.min(window_alignments)) if window_alignments else 0.0,
            "low_risk_alignment": float(np.mean(low_aligns)) if low_aligns else 0.0,
            "high_risk_alignment": float(np.mean(high_aligns)) if high_aligns else 0.0,
            "n_windows": len(window_alignments),
        })

    align_means = [s["mean_alignment"] for s in results["seeds"]]
    low_means = [s.get("low_risk_alignment", 0) for s in results["seeds"]]
    high_means = [s.get("high_risk_alignment", 0) for s in results["seeds"]]

    mean_al = float(np.mean(align_means))
    low_al = float(np.mean(low_means))
    high_al = float(np.mean(high_means))

    if mean_al > 0.9:
        conclusion = f"GLOBALLY STABLE: mean alignment={mean_al:.3f}>0.9. J structure is fixed."
    elif abs(low_al - high_al) > 0.1:
        conclusion = f"STATE-DEPENDENT: low_risk align={low_al:.3f}, high_risk={high_al:.3f}. Condition-dependent structure."
    else:
        conclusion = f"MODERATELY STABLE: mean align={mean_al:.3f}. Low={low_al:.3f} vs high={high_al:.3f}."

    results["analysis"] = {
        "mean_alignment": mean_al, "low_risk_alignment": low_al,
        "high_risk_alignment": high_al, "conclusion": conclusion,
    }
    return results


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_t2_dynamic_jacobian(n_seeds=8)
    with open("core_mvp_v4/results/t2_dynamic_jacobian.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("T2-4:", r["analysis"]["conclusion"])
