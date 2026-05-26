"""Phase 0 Analysis: Controllability Subspace via Drift-Grouped Probes.

Loads the full trajectory from phase0_full_trajectory.json,
fits controllability probes per drift group, and computes
subspace principal angles between groups.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles


def load_trajectory(path=None):
    if path is None:
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


def group_by_drift(drift):
    threshold = np.median(drift)
    idx_low = np.where(drift <= threshold)[0]
    idx_high = np.where(drift > threshold)[0]
    groups = {
        "low_drift": idx_low,
        "high_drift": idx_high,
    }
    print(f"drift threshold (median): {threshold:.4f}")
    print(f"low_drift: {len(idx_low)} steps, high_drift: {len(idx_high)} steps")
    return groups


def fit_probes(hiddens, controllability, groups, top_k=5):
    subspaces = {}
    for name, idx in groups.items():
        H = hiddens[idx]
        y = controllability[idx]

        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H)

        probe = Ridge(alpha=1.0)
        probe.fit(H_scaled, y)

        coef = np.abs(probe.coef_)
        topk_idx = np.argsort(coef)[-top_k:]

        basis = np.eye(H.shape[1])[topk_idx]
        Q, _ = np.linalg.qr(basis.T)

        subspaces[name] = Q

        r2 = probe.score(H_scaled, y)
        print(f"{name}: R² = {r2:.4f}, top-{top_k} dims: {topk_idx.tolist()}")
    return subspaces


def compute_angles(subspaces):
    Q_low = subspaces["low_drift"]
    Q_high = subspaces["high_drift"]
    angles = np.degrees(subspace_angles(Q_low, Q_high))
    print(f"principal angles: {angles}")
    return angles


def main():
    print("=" * 55)
    print("  PHASE 0: CONTROLLABILITY SUBSPACE ANALYSIS")
    print("=" * 55)

    print("\n--- STEP 1: load data ---")
    states, actions, hiddens, drift, controllability, env_states, traj = load_trajectory()
    print(f"trajectory length: {len(traj)}")
    print(f"hidden dim: {hiddens.shape[1]}")
    print(f"drift range: [{drift.min():.4f}, {drift.max():.4f}]")
    print(f"controllability mean: {controllability.mean():.4f}")

    print("\n--- STEP 2: group by drift ---")
    groups = group_by_drift(drift)

    print(f"\n--- STEP 3: fit controllability probes ---")
    subspaces = fit_probes(hiddens, controllability, groups, top_k=5)

    print(f"\n--- STEP 4: subspace angles ---")
    angles = compute_angles(subspaces)
    print(f"max principal angle: {angles.max():.2f}°")
    print(f"min principal angle: {angles.min():.2f}°")


if __name__ == "__main__":
    main()
