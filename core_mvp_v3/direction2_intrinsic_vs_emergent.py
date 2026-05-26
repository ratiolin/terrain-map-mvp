"""Direction 2: Cross-Subspace Probe R² — Intrinsic vs Emergent.

Uses the fitted subspace bases from Direction 0 and tests how well each
subspace decodes controllability under different drift conditions.
Tests the hypothesis that condition-specific subspaces are emergent
(work only in their own regime) while shared axes are intrinsic
(work cross-condition).
"""
import json
from pathlib import Path

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def load_data():
    path = Path("results_final/phase0_full_trajectory.json")
    with open(path) as f:
        traj = json.load(f)
    hiddens = np.array([t["hidden_state"] for t in traj]).squeeze(1)
    drift = np.array([t["drift"] for t in traj])
    controllability = np.array([t["controllability"] for t in traj])
    return hiddens, drift, controllability


def build_subspace(dims, hidden_dim):
    basis = np.eye(hidden_dim)[dims]
    Q, _ = np.linalg.qr(basis.T)
    return Q


def project_features(H, Q):
    return H @ Q


def fit_probe(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(Xs, y)
    return model.score(Xs, y)


def random_R2(H, y, k, n=200):
    scores = []
    for _ in range(n):
        M = np.random.randn(H.shape[1], k)
        Q_rand, _ = np.linalg.qr(M)
        X = H @ Q_rand
        scores.append(fit_probe(X, y))
    return np.array(scores)


def zscore(obs, base):
    return float((obs - base.mean()) / (base.std() + 1e-8))


def main():
    print("=" * 60)
    print("  DIRECTION 2: CROSS-SUBSPACE PROBE R²")
    print("  Intrinsic vs Emergent Subspace Test")
    print("=" * 60)

    hiddens, drift, controllability = load_data()
    hidden_dim = hiddens.shape[1]

    shared_dims = [18]
    low_dims = [1, 10, 28, 13]
    high_dims = [8, 27, 5, 15]

    Q_shared = build_subspace(shared_dims, hidden_dim)
    Q_low = build_subspace(low_dims, hidden_dim)
    Q_high = build_subspace(high_dims, hidden_dim)

    threshold = np.median(drift)
    idx_low = np.where(drift <= threshold)[0]
    idx_high = np.where(drift > threshold)[0]

    H_low = hiddens[idx_low]
    y_low = controllability[idx_low]
    H_high = hiddens[idx_high]
    y_high = controllability[idx_high]

    print(f"\n  low drift: {len(H_low)} samples")
    print(f"  high drift: {len(H_high)} samples")
    print(f"  Q_shared: {Q_shared.shape}, Q_low: {Q_low.shape}, Q_high: {Q_high.shape}")

    print("\n--- fitting cross-subspace probes ---")

    R2_low_shared = fit_probe(project_features(H_low, Q_shared), y_low)
    R2_low_on_low = fit_probe(project_features(H_low, Q_low), y_low)
    R2_low_on_high = fit_probe(project_features(H_low, Q_high), y_low)

    R2_high_shared = fit_probe(project_features(H_high, Q_shared), y_high)
    R2_high_on_high = fit_probe(project_features(H_high, Q_high), y_high)
    R2_high_on_low = fit_probe(project_features(H_high, Q_low), y_high)

    print(f"\n  LOW drift data:")
    print(f"    shared (1d)  → R²={R2_low_shared:.4f}")
    print(f"    correct (4d) → R²={R2_low_on_low:.4f}")
    print(f"    wrong (4d)   → R²={R2_low_on_high:.4f}")

    print(f"\n  HIGH drift data:")
    print(f"    shared (1d)  → R²={R2_high_shared:.4f}")
    print(f"    correct (4d) → R²={R2_high_on_high:.4f}")
    print(f"    wrong (4d)   → R²={R2_high_on_low:.4f}")

    print("\n--- computing random baselines ---")

    base_low_1d = random_R2(H_low, y_low, 1, n=200)
    base_low_4d = random_R2(H_low, y_low, 4, n=200)
    base_high_1d = random_R2(H_high, y_high, 1, n=200)
    base_high_4d = random_R2(H_high, y_high, 4, n=200)

    print(f"  base low 1d:  μ={base_low_1d.mean():.4f}, σ={base_low_1d.std():.4f}")
    print(f"  base low 4d:  μ={base_low_4d.mean():.4f}, σ={base_low_4d.std():.4f}")
    print(f"  base high 1d: μ={base_high_1d.mean():.4f}, σ={base_high_1d.std():.4f}")
    print(f"  base high 4d: μ={base_high_4d.mean():.4f}, σ={base_high_4d.std():.4f}")

    print("\n--- z-scores ---")

    z_low_shared = zscore(R2_low_shared, base_low_1d)
    z_low_on_low = zscore(R2_low_on_low, base_low_4d)
    z_low_on_high = zscore(R2_low_on_high, base_low_4d)

    z_high_shared = zscore(R2_high_shared, base_high_1d)
    z_high_on_high = zscore(R2_high_on_high, base_high_4d)
    z_high_on_low = zscore(R2_high_on_low, base_high_4d)

    print(f"  LOW:  shared z={z_low_shared:.2f}  correct z={z_low_on_low:.2f}  "
          f"wrong z={z_low_on_high:.2f}")
    print(f"  HIGH: shared z={z_high_shared:.2f}  correct z={z_high_on_high:.2f}  "
          f"wrong z={z_high_on_low:.2f}")

    selectivity_low = R2_low_on_low - R2_low_on_high
    selectivity_high = R2_high_on_high - R2_high_on_low
    print(f"\n  selectivity (correct - wrong):")
    print(f"    low:  ΔR²={selectivity_low:+.4f}")
    print(f"    high: ΔR²={selectivity_high:+.4f}")

    results = {
        "low": {
            "shared": {"R2": float(R2_low_shared), "z": z_low_shared},
            "correct": {"R2": float(R2_low_on_low), "z": z_low_on_low},
            "wrong": {"R2": float(R2_low_on_high), "z": z_low_on_high},
            "selectivity": float(selectivity_low),
        },
        "high": {
            "shared": {"R2": float(R2_high_shared), "z": z_high_shared},
            "correct": {"R2": float(R2_high_on_high), "z": z_high_on_high},
            "wrong": {"R2": float(R2_high_on_low), "z": z_high_on_low},
            "selectivity": float(selectivity_high),
        },
        "dims": {
            "shared": shared_dims,
            "low_specific": low_dims,
            "high_specific": high_dims,
        },
    }

    out_path = Path("results_final/direction2_intrinsic_vs_emergent.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  saved → {out_path}")

    if selectivity_low > 0 and selectivity_high > 0:
        print("\n  ✓ both regimes show positive selectivity (emergent subspaces)")
    elif selectivity_low > 0:
        print("\n  ~ only low regime emergent")
    elif selectivity_high > 0:
        print("\n  ~ only high regime emergent")
    else:
        print("\n  ✗ no selectivity — subspaces are fully cross-functional")


if __name__ == "__main__":
    main()
