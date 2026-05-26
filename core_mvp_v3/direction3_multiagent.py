"""Direction 3: Multi-Agent Subspace Consistency.

Trains a second agent (seed=1), repeats direction 0-2 subspace extraction,
then computes:
  - Principal angles between subspaces from different seeds
  - Cross-agent Jacobian alignment (agent A subspace vs agent B Jacobian)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles

from core_mvp_v3.models import PolicyNetwork


def load_trajectory(tag=""):
    suffix = f"_{tag}" if tag else ""
    path = Path(f"results_final/phase0_full_trajectory{suffix}.json")
    with open(path) as f:
        traj = json.load(f)
    states = np.array([t["state"] for t in traj])
    actions = np.array([t["action"] for t in traj])
    hiddens = np.array([t["hidden_state"] for t in traj]).squeeze(1)
    drift = np.array([t["drift"] for t in traj])
    controllability = np.array([t["controllability"] for t in traj])
    env_states = [t["env_state"] for t in traj]
    return states, actions, hiddens, drift, controllability, env_states, traj


def load_backbone(tag=""):
    suffix = f"_{tag}" if tag else ""
    backbone = PolicyNetwork(hidden_dim=32).backbone
    state_dict = torch.load(Path(f"results_final/phase0_policy_net{suffix}.pt"),
                            map_location="cpu", weights_only=True)
    backbone_state = {k.replace("backbone.", ""): v
                      for k, v in state_dict.items()
                      if k.startswith("backbone.")}
    backbone.load_state_dict(backbone_state)
    backbone.eval()
    return backbone


def compute_jacobian(backbone, state_val):
    x = torch.tensor(state_val, dtype=torch.float32).unsqueeze(0)
    x.requires_grad_(True)
    h = backbone(x)
    rows = []
    for i in range(h.shape[1]):
        grad = torch.autograd.grad(h[0, i], x, retain_graph=True)[0]
        rows.append(grad.detach().cpu().numpy().flatten())
    return np.array(rows)


def compute_jacobians(backbone, env_states):
    jacobians = []
    for es in env_states:
        J = compute_jacobian(backbone, es["x"])
        jacobians.append(J)
    return np.array(jacobians)


def fit_probe_dims(hiddens, controllability, drift, top_k=5):
    threshold = np.median(drift)
    dims = {}
    for name in ["low", "high"]:
        idx = np.where(drift <= threshold)[0] if name == "low" else np.where(drift > threshold)[0]
        H = hiddens[idx]
        y = controllability[idx]
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H)
        probe = Ridge(alpha=1.0)
        probe.fit(H_scaled, y)
        coef = np.abs(probe.coef_)
        dims[name] = [int(d) for d in list(np.argsort(coef)[-top_k:])]
        print(f"  {name}_drift R²={probe.score(H_scaled, y):.4f} top-{top_k}: {dims[name]}")
    return dims["low"], dims["high"]


def build_subspace(dims, hidden_dim):
    basis = np.eye(hidden_dim)[dims]
    Q, _ = np.linalg.qr(basis.T)
    return Q


def alignment(Q, J):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    k = Q.shape[1]
    U_top = U[:, :k]
    angles = subspace_angles(Q, U_top)
    return float(np.cos(angles).mean())


def main():
    print("=" * 60)
    print("  DIRECTION 3: MULTI-AGENT SUBSPACE CONSISTENCY")
    print("=" * 60)

    print("\n--- loading agent A (seed=42) ---")
    states_a, actions_a, hiddens_a, drift_a, ctrl_a, env_states_a, traj_a = load_trajectory()
    backbone_a = load_backbone()
    print(f"  {len(traj_a)} steps")

    print("\n--- loading agent B (seed=1) ---")
    states_b, actions_b, hiddens_b, drift_b, ctrl_b, env_states_b, traj_b = load_trajectory("seed1")
    backbone_b = load_backbone("seed1")
    print(f"  {len(traj_b)} steps")

    hidden_dim = hiddens_a.shape[1]

    print("\n--- fitting probes: agent A ---")
    low_dims_a, high_dims_a = fit_probe_dims(hiddens_a, ctrl_a, drift_a, top_k=5)

    print("\n--- fitting probes: agent B ---")
    low_dims_b, high_dims_b = fit_probe_dims(hiddens_b, ctrl_b, drift_b, top_k=5)

    shared_a = list(set(low_dims_a) & set(high_dims_a))
    shared_b = list(set(low_dims_b) & set(high_dims_b))
    print(f"\n  shared A: {shared_a}")
    print(f"  shared B: {shared_b}")

    Q_low_a = build_subspace(low_dims_a, hidden_dim)
    Q_high_a = build_subspace(high_dims_a, hidden_dim)
    Q_shared_a = build_subspace(shared_a, hidden_dim)

    Q_low_b = build_subspace(low_dims_b, hidden_dim)
    Q_high_b = build_subspace(high_dims_b, hidden_dim)
    Q_shared_b = build_subspace(shared_b, hidden_dim)

    print("\n--- STEP 9: principal angles between agents ---")
    angles_low = np.degrees(subspace_angles(Q_low_a, Q_low_b))
    angles_high = np.degrees(subspace_angles(Q_high_a, Q_high_b))
    angles_shared = np.degrees(subspace_angles(Q_shared_a, Q_shared_b))

    # Handle case where shared dims may differ between agents
    if Q_shared_a.shape[1] == 0 or Q_shared_b.shape[1] == 0:
        angles_shared = np.array([90.0])
        print(f"  shared: NO OVERLAP in at least one agent → angle=90°")
    else:
        print(f"  shared principal angles: {np.round(angles_shared, 1)}")
    print(f"  low principal angles:    {np.round(angles_low, 1)}")
    print(f"  high principal angles:   {np.round(angles_high, 1)}")
    print(f"  low  min angle:  {angles_low.min():.1f}°")
    print(f"  high min angle:  {angles_high.min():.1f}°")

    print("\n--- STEP 10: cross-agent Jacobian alignment ---")
    jacobians_a = compute_jacobians(backbone_a, env_states_a)
    jacobians_b = compute_jacobians(backbone_b, env_states_b)

    threshold_a = np.median(drift_a)
    threshold_b = np.median(drift_b)
    J_low_a = jacobians_a[np.where(drift_a <= threshold_a)[0]].mean(axis=0)
    J_high_a = jacobians_a[np.where(drift_a > threshold_a)[0]].mean(axis=0)
    J_low_b = jacobians_b[np.where(drift_b <= threshold_b)[0]].mean(axis=0)
    J_high_b = jacobians_b[np.where(drift_b > threshold_b)[0]].mean(axis=0)

    align_low_cross = alignment(Q_low_a, J_low_b)
    align_high_cross = alignment(Q_high_a, J_high_b)

    print(f"  A_low_subspace vs B_low_Jacobian:    {align_low_cross:.4f}")
    print(f"  A_high_subspace vs B_high_Jacobian:  {align_high_cross:.4f}")

    results = {
        "angles": {
            "low": angles_low.tolist(),
            "high": angles_high.tolist(),
            "shared": angles_shared.tolist(),
            "summary": {
                "low_min": float(angles_low.min()),
                "high_min": float(angles_high.min()),
            },
        },
        "cross_alignment": {
            "low": align_low_cross,
            "high": align_high_cross,
        },
        "dims": {
            "A": {"low": low_dims_a, "high": high_dims_a, "shared": shared_a},
            "B": {"low": low_dims_b, "high": high_dims_b, "shared": shared_b},
        },
    }

    out_path = Path("results_final/direction3_multiagent.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
