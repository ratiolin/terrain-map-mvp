"""Direction 2c: Functional Validation of the 1D Invariant Axis.

Tests whether the shared probe dimension (cross-agent invariant axis)
carries functional signal: action prediction, mode switching,
controllability decoding, cross-agent generalization, and temporal
switch-point alignment.
"""
import json
from pathlib import Path

import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_trajectory(tag=""):
    suffix = f"_{tag}" if tag else ""
    path = Path(f"results_final/phase0_full_trajectory{suffix}.json")
    with open(path) as f:
        traj = json.load(f)
    hiddens = np.array([t["hidden_state"] for t in traj]).squeeze(1)
    drift = np.array([t["drift"] for t in traj])
    controllability = np.array([t["controllability"] for t in traj])
    actions = np.array([t["action"] for t in traj])
    mode_seq = np.array([t["env_state"]["internal"]["current_drift"]
                         for t in traj])
    return hiddens, drift, controllability, actions, traj, mode_seq


def build_subspace(dims, hidden_dim):
    basis = np.eye(hidden_dim)[dims]
    Q, _ = np.linalg.qr(basis.T)
    return Q


def align_sign(q1, q2):
    if np.dot(q1.squeeze(), q2.squeeze()) < 0:
        return -q2
    return q2


def project_scalar(H, q):
    return H @ q


def controllability_R2(z, y):
    model = LinearRegression()
    model.fit(z.reshape(-1, 1), y)
    return model.score(z.reshape(-1, 1), y)


def fit_controllability_R2(H, y):
    scaler = StandardScaler()
    z = H.reshape(-1, 1)
    zs = scaler.fit_transform(z)
    model = LinearRegression()
    model.fit(zs, y)
    return model.score(zs, y)


def zscore(obs, base):
    return float((obs - base.mean()) / (base.std() + 1e-8))


def random_1d_baseline(H, y, n=200):
    scores = []
    for _ in range(n):
        q = np.random.randn(H.shape[1])
        q /= np.linalg.norm(q)
        z = H @ q
        scores.append(controllability_R2(z, y))
    return np.array(scores)


def main():
    print("=" * 60)
    print("  DIRECTION 2c: 1D INVARIANT AXIS VALIDATION")
    print("=" * 60)

    print("\n--- loading data ---")
    hiddens_A, drift_A, ctrl_A, actions_A, traj_A, _ = load_trajectory()
    hiddens_B, drift_B, ctrl_B, actions_B, traj_B, _ = load_trajectory("seed1")

    hidden_dim = hiddens_A.shape[1]
    print(f"  agent A: {len(traj_A)} steps")
    print(f"  agent B: {len(traj_B)} steps")

    shared_dims_A = [18]
    shared_dims_B = [18]
    Q_shared_A = build_subspace(shared_dims_A, hidden_dim)
    Q_shared_B = build_subspace(shared_dims_B, hidden_dim)

    q_shared_A = Q_shared_A[:, 0].copy()
    q_shared_B = align_sign(q_shared_A, Q_shared_B[:, 0])

    print(f"  |q_A|={np.linalg.norm(q_shared_A):.4f}  "
          f"|q_B|={np.linalg.norm(q_shared_B):.4f}  "
          f"dot={np.dot(q_shared_A, q_shared_B):.4f}")

    z_A = project_scalar(hiddens_A, q_shared_A)
    z_B = project_scalar(hiddens_B, q_shared_B)

    print(f"  z_A range: [{z_A.min():.3f}, {z_A.max():.3f}], "
          f"z_B range: [{z_B.min():.3f}, {z_B.max():.3f}]")

    print("\n--- STEP 3: action predictability ---")
    R2_action_A = controllability_R2(z_A, actions_A.flatten())
    R2_action_B = controllability_R2(z_B, actions_B.flatten())
    print(f"  R²(action|z):  A={R2_action_A:.4f}  B={R2_action_B:.4f}")

    print("\n--- STEP 4: mode classification ---")
    mode_A = np.where(drift_A <= np.median(drift_A), 0, 1)
    mode_B = np.where(drift_B <= np.median(drift_B), 0, 1)
    clf_A = LogisticRegression()
    clf_A.fit(z_A.reshape(-1, 1), mode_A)
    acc_mode_A = clf_A.score(z_A.reshape(-1, 1), mode_A)
    clf_B = LogisticRegression()
    clf_B.fit(z_B.reshape(-1, 1), mode_B)
    acc_mode_B = clf_B.score(z_B.reshape(-1, 1), mode_B)
    print(f"  acc(mode|z):  A={acc_mode_A:.4f}  B={acc_mode_B:.4f}")

    print("\n--- STEP 5: controllability 1D decoding ---")
    R2_ctrl_A = controllability_R2(z_A, ctrl_A)
    R2_ctrl_B = controllability_R2(z_B, ctrl_B)
    print(f"  R²(ctrl|z):   A={R2_ctrl_A:.4f}  B={R2_ctrl_B:.4f}")

    print("\n--- STEP 6: cross-agent generalization ---")
    z_B_from_A = project_scalar(hiddens_B, q_shared_A)
    R2_cross = controllability_R2(z_B_from_A, ctrl_B)
    acc_cross = LogisticRegression().fit(
        z_B_from_A.reshape(-1, 1), mode_B
    ).score(z_B_from_A.reshape(-1, 1), mode_B)
    print(f"  A-axis on B data: R²(ctrl)={R2_cross:.4f}  acc(mode)={acc_cross:.4f}")

    print("\n--- STEP 7: temporal structure ---")
    dz = np.diff(z_A)
    smoothness = float(np.mean(np.abs(dz)))
    p95 = np.percentile(np.abs(dz), 95)
    switch_points = np.where(np.abs(dz) > p95)[0]
    drift_A_arr = np.array(drift_A)
    drift_jumps = np.where(np.abs(np.diff(drift_A_arr)) > 0.3)[0]
    hits = len(set(switch_points) & set(drift_jumps))
    print(f"  smoothness: {smoothness:.4f}")
    print(f"  switch_points (p95): {len(switch_points)}")
    print(f"  drift_jumps: {len(drift_jumps)}, hits: {hits}")

    print("\n--- STEP 8: random 1D baseline ---")
    base_A = random_1d_baseline(hiddens_A, ctrl_A, n=200)
    base_B = random_1d_baseline(hiddens_B, ctrl_B, n=200)
    z_ctrl_A = zscore(R2_ctrl_A, base_A)
    z_ctrl_B = zscore(R2_ctrl_B, base_B)
    z_action_A = zscore(R2_action_A, random_1d_baseline(hiddens_A, actions_A.flatten(), n=200))
    z_action_B = zscore(R2_action_B, random_1d_baseline(hiddens_B, actions_B.flatten(), n=200))
    print(f"  base ctrl: μ={base_A.mean():.4f} σ={base_A.std():.4f}")
    print(f"  z(ctrl|A)={z_ctrl_A:.2f}  z(ctrl|B)={z_ctrl_B:.2f}")
    print(f"  z(action|A)={z_action_A:.2f}  z(action|B)={z_action_B:.2f}")

    results = {
        "action_R2": {"A": float(R2_action_A), "B": float(R2_action_B)},
        "action_z": {"A": z_action_A, "B": z_action_B},
        "mode_acc": {"A": float(acc_mode_A), "B": float(acc_mode_B)},
        "controllability_R2": {
            "A": float(R2_ctrl_A), "B": float(R2_ctrl_B), "cross": float(R2_cross),
        },
        "cross_mode_acc": float(acc_cross),
        "time_structure": {
            "smoothness": smoothness,
            "n_switch_points": len(switch_points),
            "n_drift_jumps": len(drift_jumps),
            "switch_drift_hits": hits,
            "switch_points": [int(s) for s in switch_points],
        },
        "z_scores": {"ctrl_A": z_ctrl_A, "ctrl_B": z_ctrl_B,
                      "action_A": z_action_A, "action_B": z_action_B},
    }

    out_path = Path("results_final/direction2c_invariant_axis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  saved → {out_path}")

    print("\n  --- summary ---")
    flags = []
    if R2_ctrl_A > base_A.mean() + base_A.std():
        flags.append("ctrl decoding > random (+1σ)")
    if R2_cross > 0.1:
        flags.append("cross-agent generalization")
    if hits > 0:
        flags.append(f"switch→drift hits: {hits}")
    for f in flags:
        print(f"  ✓ {f}")


if __name__ == "__main__":
    main()
