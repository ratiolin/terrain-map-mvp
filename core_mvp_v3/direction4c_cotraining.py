"""Direction 4c: Co-Training.

Trains two agents from scratch simultaneously in a shared environment.
Both agents share the same reward (prediction loss) and the same
environment. Tracks CKA, action correlation, and subspace angles
over the course of training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork, PredictionNetwork
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA


def linear_CKA(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K = X @ X.T
    L = Y @ Y.T
    hsic = np.sum(K * L)
    norm = np.linalg.norm(K) * np.linalg.norm(L)
    return float(hsic / (norm + 1e-8))


def main():
    print("=" * 60)
    print("  DIRECTION 4c: CO-TRAINING")
    print("=" * 60)

    torch.manual_seed(99)
    np.random.seed(99)

    schedule = [
        (2000, (0.1, 0.3)),
        (2000, (1.0, 2.0)),
        (2000, (0.1, 0.3)),
        (2000, (1.0, 2.0)),
    ]
    env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)

    pred_A = PredictionNetwork(hidden_dim=32)
    pred_B = PredictionNetwork(hidden_dim=32)
    policy_A = PolicyNetwork(hidden_dim=32)
    policy_B = PolicyNetwork(hidden_dim=32)

    params = (
        list(pred_A.parameters()) + list(policy_A.parameters())
        + list(pred_B.parameters()) + list(policy_B.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=1e-3)

    T = 8000
    log_every = 200
    n_logs = T // log_every

    logs = {
        "step": [], "cka": [], "action_corr": [],
        "subspace_angle": [], "loss_A": [], "loss_B": [],
    }
    h_buffer_A, h_buffer_B = [], []
    a_buffer_A, a_buffer_B = [], []

    env.reset()
    print(f"  training {T} steps, logging every {log_every}")

    for step in range(T):
        state = env.state.copy()
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        as_A, aa_A, h_A = policy_A(state_t)
        as_B, aa_B, h_B = policy_B(state_t)

        a_A_val = float(as_A.item())
        a_B_val = float(as_B.item())

        next_state = env.step(a_A_val, a_B_val)
        risk_next = abs(float(next_state[0]))
        risk_tensor = torch.tensor([[risk_next]], dtype=torch.float32)

        pred_risk_A = pred_A(state_t, torch.tensor([[a_A_val]], dtype=torch.float32))
        pred_risk_B = pred_B(state_t, torch.tensor([[a_B_val]], dtype=torch.float32))

        loss_A = F.mse_loss(pred_risk_A, risk_tensor)
        loss_B = F.mse_loss(pred_risk_B, risk_tensor)
        loss = loss_A + loss_B

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        h_buffer_A.append(h_A.detach().cpu().numpy().flatten())
        h_buffer_B.append(h_B.detach().cpu().numpy().flatten())
        a_buffer_A.append(a_A_val)
        a_buffer_B.append(a_B_val)

        if (step + 1) % log_every == 0:
            H_A = np.array(h_buffer_A)
            H_B = np.array(h_buffer_B)
            cka = linear_CKA(H_A, H_B)
            a_A_arr = np.array(a_buffer_A)
            a_B_arr = np.array(a_buffer_B)
            action_corr = float(np.corrcoef(a_A_arr, a_B_arr)[0, 1])

            k = min(5, min(H_A.shape[1], H_A.shape[0]))
            pca_A = PCA(n_components=k).fit(H_A)
            pca_B = PCA(n_components=k).fit(H_B)
            angle = float(np.degrees(subspace_angles(
                pca_A.components_.T, pca_B.components_.T)).max())

            logs["step"].append(step + 1)
            logs["cka"].append(round(cka, 4))
            logs["action_corr"].append(round(action_corr, 4))
            logs["subspace_angle"].append(round(angle, 1))
            logs["loss_A"].append(round(float(loss_A.item()), 6))
            logs["loss_B"].append(round(float(loss_B.item()), 6))
            h_buffer_A, h_buffer_B = [], []
            a_buffer_A, a_buffer_B = [], []

            if (step + 1) % 2000 == 0:
                print(f"  step {step+1:5d}: CKA={cka:.4f}  "
                      f"action_corr={action_corr:.4f}  "
                      f"pca_angle={angle:.1f}°  "
                      f"loss_A={loss_A.item():.4f}  loss_B={loss_B.item():.4f}")

    print(f"\n  final CKA: {logs['cka'][-1]:.4f}")
    print(f"  final action_corr: {logs['action_corr'][-1]:.4f}")
    print(f"  final subspace_angle: {logs['subspace_angle'][-1]:.1f}°")

    early_cka = np.mean(logs["cka"][:len(logs["cka"])//4])
    late_cka = np.mean(logs["cka"][3*len(logs["cka"])//4:])
    early_corr = np.mean(logs["action_corr"][:len(logs["action_corr"])//4])
    late_corr = np.mean(logs["action_corr"][3*len(logs["action_corr"])//4:])
    print(f"  CKA early→late: {early_cka:.4f} → {late_cka:.4f} "
          f"({'↑ converging' if late_cka > early_cka else '↓ diverging'})")
    print(f"  corr early→late: {early_corr:.4f} → {late_corr:.4f} "
          f"({'↑ converging' if late_corr > early_corr else '↓ diverging'})")

    results = {
        "logs": logs,
        "summary": {
            "early_cka": float(early_cka),
            "late_cka": float(late_cka),
            "early_corr": float(early_corr),
            "late_corr": float(late_corr),
            "trend_cka": "converging" if late_cka > early_cka else "diverging",
            "trend_corr": "converging" if late_corr > early_corr else "diverging",
        },
    }

    out_path = Path("results_final/direction4c_cotraining.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out_path}")

    torch.save(policy_A.state_dict(), Path("results_final/direction4c_policy_A.pt"))
    torch.save(policy_B.state_dict(), Path("results_final/direction4c_policy_B.pt"))
    torch.save(pred_A.state_dict(), Path("results_final/direction4c_pred_A.pt"))
    torch.save(pred_B.state_dict(), Path("results_final/direction4c_pred_B.pt"))
    print("  saved co-trained models")


if __name__ == "__main__":
    main()
