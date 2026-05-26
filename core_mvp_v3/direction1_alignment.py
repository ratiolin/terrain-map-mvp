"""Direction 1: Subspace-Jacobian Alignment Analysis.

Splits controllability probe subspaces into shared and condition-specific
axes, then measures alignment between each subspace and the average
Jacobian of the policy backbone under each drift regime. Computes z-scores
against random baselines.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles

from core_mvp_v3.models import PolicyNetwork


def load_trajectory():
    path = Path("results_final/phase0_full_trajectory.json")
    with open(path) as f:
        traj = json.load(f)
    states = np.array([t["state"] for t in traj])
    actions = np.array([t["action"] for t in traj])
    hiddens = np.array([t["hidden_state"] for t in traj]).squeeze(1)
    drift = np.array([t["drift"] for t in traj])
    controllability = np.array([t["controllability"] for t in traj])
    env_states = [t["env_state"] for t in traj]
    return states, actions, hiddens, drift, controllability, env_states, traj


def load_backbone():
    backbone = PolicyNetwork(hidden_dim=32).backbone
    state_dict = torch.load(Path("results_final/phase0_policy_net.pt"),
                            map_location="cpu", weights_only=True)
    backbone_state = {k.replace("backbone.", ""): v
                      for k, v in state_dict.items()
                      if k.startswith("backbone.")}
    backbone.load_state_dict(backbone_state)
    backbone.eval()
    return backbone


def compute_jacobian_backbone(backbone, state_val):
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
        J = compute_jacobian_backbone(backbone, es["x"])
        jacobians.append(J)
    return np.array(jacobians)


def fit_probe_dims(hiddens, controllability, drift, top_k=5):
    threshold = np.median(drift)
    idx_low = np.where(drift <= threshold)[0]
    idx_high = np.where(drift > threshold)[0]

    dims = {}
    for name, idx in [("low", idx_low), ("high", idx_high)]:
        H = hiddens[idx]
        y = controllability[idx]
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H)
        probe = Ridge(alpha=1.0)
        probe.fit(H_scaled, y)
        coef = np.abs(probe.coef_)
        topk_idx = list(np.argsort(coef)[-top_k:])
        dims[name] = topk_idx
        print(f"  {name}_drift probe R² = {probe.score(H_scaled, y):.4f}, "
              f"top-{top_k}: {topk_idx}")

    return dims["low"], dims["high"], idx_low, idx_high


def build_subspace(hidden_dim, dims):
    basis = np.eye(hidden_dim)[dims]
    Q, _ = np.linalg.qr(basis.T)
    return Q


def alignment(Q, J):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    k = Q.shape[1]
    U_top = U[:, :k]
    angles = subspace_angles(Q, U_top)
    return float(np.cos(angles).mean())


def random_baseline(hidden_dim, k, J, n=1000):
    scores = []
    for _ in range(n):
        M = np.random.randn(hidden_dim, k)
        Q_rand, _ = np.linalg.qr(M)
        scores.append(alignment(Q_rand, J))
    return np.array(scores)


def compute_z(Q, J):
    obs = alignment(Q, J)
    base = random_baseline(Q.shape[0], Q.shape[1], J)
    z = (obs - base.mean()) / (base.std() + 1e-8)
    return obs, z


def main():
    print("=" * 60)
    print("  DIRECTION 1: SUBSPACE-JACOBIAN ALIGNMENT")
    print("=" * 60)

    print("\n--- loading data & model ---")
    states, actions, hiddens, drift, controllability, env_states, traj = load_trajectory()
    backbone = load_backbone()
    hidden_dim = hiddens.shape[1]
    print(f"  trajectory: {len(traj)} steps, hidden_dim={hidden_dim}")

    print("\nSTEP 1: fitting probes & building subspaces")
    low_dims, high_dims, idx_low, idx_high = fit_probe_dims(
        hiddens, controllability, drift, top_k=5)

    shared_dims = list(set(low_dims) & set(high_dims))
    low_specific = list(set(low_dims) - set(shared_dims))
    high_specific = list(set(high_dims) - set(shared_dims))
    print(f"  shared: {shared_dims}")
    print(f"  low_specific:  {low_specific}")
    print(f"  high_specific: {high_specific}")

    Q_shared = build_subspace(hidden_dim, shared_dims)
    Q_low = build_subspace(hidden_dim, low_dims)
    Q_high = build_subspace(hidden_dim, high_dims)
    Q_low_spec = build_subspace(hidden_dim, low_specific)
    Q_high_spec = build_subspace(hidden_dim, high_specific)
    print(f"  Q_shared: {Q_shared.shape}, Q_low: {Q_low.shape}, "
          f"Q_high: {Q_high.shape}")
    print(f"  Q_low_spec: {Q_low_spec.shape}, Q_high_spec: {Q_high_spec.shape}")

    print("\nSTEP 2-3: computing Jacobians")
    jacobians = compute_jacobians(backbone, env_states)
    print(f"  jacobians shape: {jacobians.shape}")

    J_low = jacobians[idx_low].mean(axis=0)
    J_high = jacobians[idx_high].mean(axis=0)
    print(f"  J_low: {J_low.shape}, J_high: {J_high.shape}")

    print("\nSTEP 4-7: computing alignments & z-scores")

    results = {}

    print("\n  --- shared axis ---")
    shared_low_obs, shared_low_z = compute_z(Q_shared, J_low)
    shared_high_obs, shared_high_z = compute_z(Q_shared, J_high)
    print(f"  shared_on_low:  obs={shared_low_obs:.4f}, z={shared_low_z:.2f}")
    print(f"  shared_on_high: obs={shared_high_obs:.4f}, z={shared_high_z:.2f}")
    results["shared"] = {
        "low": {"obs": shared_low_obs, "z": shared_low_z},
        "high": {"obs": shared_high_obs, "z": shared_high_z},
    }

    print("\n  --- full subspace ---")
    low_low_obs, low_low_z = compute_z(Q_low, J_low)
    low_high_obs, low_high_z = compute_z(Q_low, J_high)
    print(f"  low_on_low:   obs={low_low_obs:.4f}, z={low_low_z:.2f}")
    print(f"  low_on_high:  obs={low_high_obs:.4f}, z={low_high_z:.2f}")

    high_high_obs, high_high_z = compute_z(Q_high, J_high)
    high_low_obs, high_low_z = compute_z(Q_high, J_low)
    print(f"  high_on_high: obs={high_high_obs:.4f}, z={high_high_z:.2f}")
    print(f"  high_on_low:  obs={high_low_obs:.4f}, z={high_low_z:.2f}")
    results["full"] = {
        "low_on_low": {"obs": low_low_obs, "z": low_low_z},
        "low_on_high": {"obs": low_high_obs, "z": low_high_z},
        "high_on_high": {"obs": high_high_obs, "z": high_high_z},
        "high_on_low": {"obs": high_low_obs, "z": high_low_z},
    }

    print("\n  --- specific subspace ---")
    low_spec_low_obs, low_spec_low_z = compute_z(Q_low_spec, J_low)
    low_spec_high_obs, low_spec_high_z = compute_z(Q_low_spec, J_high)
    print(f"  low_spec_on_low:   obs={low_spec_low_obs:.4f}, z={low_spec_low_z:.2f}")
    print(f"  low_spec_on_high:  obs={low_spec_high_obs:.4f}, z={low_spec_high_z:.2f}")

    high_spec_high_obs, high_spec_high_z = compute_z(Q_high_spec, J_high)
    high_spec_low_obs, high_spec_low_z = compute_z(Q_high_spec, J_low)
    print(f"  high_spec_on_high: obs={high_spec_high_obs:.4f}, z={high_spec_high_z:.2f}")
    print(f"  high_spec_on_low:  obs={high_spec_low_obs:.4f}, z={high_spec_low_z:.2f}")
    results["specific"] = {
        "low_on_low": {"obs": low_spec_low_obs, "z": low_spec_low_z},
        "low_on_high": {"obs": low_spec_high_obs, "z": low_spec_high_z},
        "high_on_high": {"obs": high_spec_high_obs, "z": high_spec_high_z},
        "high_on_low": {"obs": high_spec_low_obs, "z": high_spec_low_z},
    }

    results["dims"] = {
        "low": [int(d) for d in low_dims],
        "high": [int(d) for d in high_dims],
        "shared": [int(d) for d in shared_dims],
        "low_specific": [int(d) for d in low_specific],
        "high_specific": [int(d) for d in high_specific],
    }

    out_path = Path("results_final/direction1_alignment.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  saved to {out_path}")


if __name__ == "__main__":
    main()
