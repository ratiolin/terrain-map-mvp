"""Direction 3: Multi-Agent Interaction Consistency.

Runs two independently trained agents (seed=42, seed=1) synchronously
in a shared environment. Measures:
  L1 — behavioral alignment (action diff, controllability corr)
  L2 — representational alignment (CKA, subspace angles)
  L3 — temporal dynamics (sliding-window CKA)
  L4 — intervention (perturb agent B, test coordination collapse)
"""
import json
import sys
import copy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork

from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles


def load_policy(tag=""):
    suffix = f"_{tag}" if tag else ""
    net = PolicyNetwork(hidden_dim=32)
    state_dict = torch.load(Path(f"results_final/phase0_policy_net{suffix}.pt"),
                            map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def encode(net, state):
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action_shape, action_adapt, h = net(x)
    return h.detach().cpu().numpy(), float(action_shape.item()), float(action_adapt.item())


def linear_CKA(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K = X @ X.T
    L = Y @ Y.T
    hsic = np.sum(K * L)
    norm = np.linalg.norm(K) * np.linalg.norm(L)
    return float(hsic / (norm + 1e-8))


def top_subspace(H, k=5):
    pca = PCA(n_components=k)
    pca.fit(H)
    return pca.components_.T


def main():
    print("=" * 60)
    print("  DIRECTION 3: MULTI-AGENT INTERACTION")
    print("=" * 60)

    print("\n--- loading agents ---")
    net_A = load_policy()
    net_B = load_policy("seed1")

    schedule = [
        (2000, (0.1, 0.3)),
        (2000, (1.0, 2.0)),
        (2000, (0.1, 0.3)),
        (2000, (1.0, 2.0)),
    ]
    env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)

    T = 8000
    print(f"  rollout length: {T}")

    print("\n--- STEP 2: synchronized rollout ---")
    env.reset()
    trajectory = []
    ctrl_A_list = []
    ctrl_B_list = []

    T_ctrl = 10
    for t in range(T):
        state = env.state.copy()
        h_np_A, a_shape_A, a_adapt_A = encode(net_A, state)
        h_np_B, a_shape_B, a_adapt_B = encode(net_B, state)

        action_A = a_shape_A
        action_B = a_shape_B

        next_state = env.step(action_A, action_B)

        if t % T_ctrl == 0:
            hA_flat = h_np_A.flatten().tolist()
            hB_flat = h_np_B.flatten().tolist()
        else:
            hA_flat = []
            hB_flat = []

        trajectory.append({
            "state": state.tolist(),
            "h_A": hA_flat,
            "h_B": hB_flat,
            "a_A": action_A,
            "a_B": action_B,
            "drift": env.current_drift,
        })

    print(f"  collected {len(trajectory)} steps")

    a_A = np.array([t["a_A"] for t in trajectory])
    a_B = np.array([t["a_B"] for t in trajectory])
    H_A_raw = np.array([t["h_A"] for t in trajectory if len(t["h_A"]) > 0])
    H_B_raw = np.array([t["h_B"] for t in trajectory if len(t["h_B"]) > 0])

    print("\n--- STEP 3: behavioral alignment (L1) ---")
    action_diff = float(np.mean(np.abs(a_A - a_B)))
    action_corr = float(np.corrcoef(a_A, a_B)[0, 1])
    print(f"  mean |a_A - a_B|: {action_diff:.4f}")
    print(f"  corr(a_A, a_B):  {action_corr:.4f}")

    print("\n--- STEP 4: representational alignment (L2) ---")
    cka_score = linear_CKA(H_A_raw, H_B_raw)
    print(f"  CKA: {cka_score:.4f}")

    k_top = min(5, min(H_A_raw.shape[1], H_A_raw.shape[0]))
    Q_A = top_subspace(H_A_raw, k=k_top)
    Q_B = top_subspace(H_B_raw, k=k_top)
    angles = np.degrees(subspace_angles(Q_A, Q_B))
    print(f"  PCA-{k_top} principal angles: {np.round(angles, 1)}")
    print(f"  min angle: {angles.min():.1f}°  max angle: {angles.max():.1f}°")

    print("\n--- STEP 5: temporal dynamics (L3) ---")
    window = 200
    cka_time = []
    for i in range(0, len(H_A_raw) - window, max(1, (len(H_A_raw) - window) // 50)):
        cka_time.append(linear_CKA(H_A_raw[i:i+window], H_B_raw[i:i+window]))
    cka_time = np.array(cka_time)
    print(f"  CKA over time: min={cka_time.min():.4f} max={cka_time.max():.4f} "
          f"mean={cka_time.mean():.4f} trend={'↑' if cka_time[-1] > cka_time[0] else '↓'}")

    print("\n--- STEP 6: intervention experiment ---")
    noise_std = 0.1
    a_B_noisy = a_B + np.random.normal(0, noise_std, size=a_B.shape)
    action_diff_noisy = float(np.mean(np.abs(a_A - a_B_noisy)))
    action_corr_noisy = float(np.corrcoef(a_A, a_B_noisy)[0, 1])

    print(f"  original  |a_A - a_B|: {action_diff:.4f}  corr: {action_corr:.4f}")
    print(f"  perturbed |a_A - a_B|: {action_diff_noisy:.4f}  corr: {action_corr_noisy:.4f}")
    print(f"  Δdiff: {action_diff_noisy - action_diff:+.4f}  "
          f"Δcorr: {action_corr_noisy - action_corr:+.4f}")

    coordination_collapse = action_corr_noisy < action_corr * 0.5
    print(f"  coordination collapse: {'YES' if coordination_collapse else 'no'}")

    results = {
        "behavioral": {
            "action_diff": action_diff,
            "action_corr": action_corr,
            "action_diff_noisy": action_diff_noisy,
            "action_corr_noisy": action_corr_noisy,
            "coordination_collapse": coordination_collapse,
        },
        "representational": {
            "cka": cka_score,
            "cka_time_mean": float(cka_time.mean()),
            "cka_time_min": float(cka_time.min()),
            "cka_time_max": float(cka_time.max()),
            "cka_trend": "up" if cka_time[-1] > cka_time[0] else "down",
            "pca_k": k_top,
            "subspace_angles": angles.tolist(),
            "min_angle": float(angles.min()),
        },
        "setup": {
            "T": T,
            "agent_A_seed": 42,
            "agent_B_seed": 1,
            "action_mix": 0.5,
            "intervention_noise": noise_std,
        },
    }

    out_path = Path("results_final/direction4_interaction.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
