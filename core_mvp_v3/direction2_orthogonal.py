"""Direction 2 Extension: Orthogonal Complement Probe Test.

Compares R² from fitting probes on the controllability subspace
vs its orthogonal complement. Tests whether the subspace dimensions
are sufficient (complement carries no signal) or necessary
(complement carries significant signal).
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


def orthogonal_complement(Q):
    U, _, _ = np.linalg.svd(Q, full_matrices=True)
    k = Q.shape[1]
    return U[:, k:]


def project(H, Q):
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
    print("  DIRECTION 2b: ORTHOGONAL COMPLEMENT TEST")
    print("=" * 60)

    hiddens, drift, controllability = load_data()
    hidden_dim = hiddens.shape[1]

    shared_dims = [18]
    low_dims = [1, 10, 28, 13]
    high_dims = [8, 27, 5, 15]

    Q_low = build_subspace(low_dims, hidden_dim)
    Q_high = build_subspace(high_dims, hidden_dim)
    Q_shared = build_subspace(shared_dims, hidden_dim)

    Q_low_orth = orthogonal_complement(Q_low)
    Q_high_orth = orthogonal_complement(Q_high)
    Q_shared_orth = orthogonal_complement(Q_shared)

    print(f"  Q_low: {Q_low.shape}  →  Q_low_orth: {Q_low_orth.shape}")
    print(f"  Q_high: {Q_high.shape}  →  Q_high_orth: {Q_high_orth.shape}")
    print(f"  Q_shared: {Q_shared.shape}  →  Q_shared_orth: {Q_shared_orth.shape}")

    threshold = np.median(drift)
    idx_low = np.where(drift <= threshold)[0]
    idx_high = np.where(drift > threshold)[0]

    H_low = hiddens[idx_low]
    y_low = controllability[idx_low]
    H_high = hiddens[idx_high]
    y_high = controllability[idx_high]

    print(f"\n  low: {len(H_low)} samples, high: {len(H_high)} samples")

    print("\n--- fitting subspace vs orthogonal complement ---")

    R2_low_in = fit_probe(project(H_low, Q_low), y_low)
    R2_low_orth = fit_probe(project(H_low, Q_low_orth), y_low)

    R2_high_in = fit_probe(project(H_high, Q_high), y_high)
    R2_high_orth = fit_probe(project(H_high, Q_high_orth), y_high)

    R2_shared_low = fit_probe(project(H_low, Q_shared), y_low)
    R2_shared_low_orth = fit_probe(project(H_low, Q_shared_orth), y_low)

    R2_shared_high = fit_probe(project(H_high, Q_shared), y_high)
    R2_shared_high_orth = fit_probe(project(H_high, Q_shared_orth), y_high)

    print(f"  low  in ({Q_low.shape[1]}d):     R²={R2_low_in:.4f}   orth ({Q_low_orth.shape[1]}d): R²={R2_low_orth:.4f}")
    print(f"  high in ({Q_high.shape[1]}d):    R²={R2_high_in:.4f}   orth ({Q_high_orth.shape[1]}d): R²={R2_high_orth:.4f}")
    print(f"  shared in ({Q_shared.shape[1]}d): low=R²={R2_shared_low:.4f}   orth ({Q_shared_orth.shape[1]}d): R²={R2_shared_low_orth:.4f}")
    print(f"  shared in ({Q_shared.shape[1]}d): high=R²={R2_shared_high:.4f}  orth ({Q_shared_orth.shape[1]}d): R²={R2_shared_high_orth:.4f}")

    print("\n--- random baseline for orthogonal complements ---")
    base_low_orth = random_R2(H_low, y_low, Q_low_orth.shape[1], n=200)
    base_high_orth = random_R2(H_high, y_high, Q_high_orth.shape[1], n=200)
    base_shared_low_orth = random_R2(H_low, y_low, Q_shared_orth.shape[1], n=200)
    base_shared_high_orth = random_R2(H_high, y_high, Q_shared_orth.shape[1], n=200)

    print(f"  base low_orth:    μ={base_low_orth.mean():.4f} σ={base_low_orth.std():.4f}")
    print(f"  base high_orth:   μ={base_high_orth.mean():.4f} σ={base_high_orth.std():.4f}")
    print(f"  base shared_low:  μ={base_shared_low_orth.mean():.4f} σ={base_shared_low_orth.std():.4f}")
    print(f"  base shared_high: μ={base_shared_high_orth.mean():.4f} σ={base_shared_high_orth.std():.4f}")

    z_low_orth = zscore(R2_low_orth, base_low_orth)
    z_high_orth = zscore(R2_high_orth, base_high_orth)
    z_shared_low_orth = zscore(R2_shared_low_orth, base_shared_low_orth)
    z_shared_high_orth = zscore(R2_shared_high_orth, base_shared_high_orth)

    print(f"\n  z(orth): low={z_low_orth:.2f} high={z_high_orth:.2f} "
          f"shared_low={z_shared_low_orth:.2f} shared_high={z_shared_high_orth:.2f}")

    print(f"\n  in/orth gap: low={R2_low_in - R2_low_orth:+.4f} "
          f"high={R2_high_in - R2_high_orth:+.4f}")

    results = {
        "low": {
            "in_subspace": float(R2_low_in),
            "orthogonal": float(R2_low_orth),
            "z_orth": z_low_orth,
            "gap": float(R2_low_in - R2_low_orth),
        },
        "high": {
            "in_subspace": float(R2_high_in),
            "orthogonal": float(R2_high_orth),
            "z_orth": z_high_orth,
            "gap": float(R2_high_in - R2_high_orth),
        },
        "shared": {
            "low_in": float(R2_shared_low),
            "low_orth": float(R2_shared_low_orth),
            "low_z_orth": z_shared_low_orth,
            "high_in": float(R2_shared_high),
            "high_orth": float(R2_shared_high_orth),
            "high_z_orth": z_shared_high_orth,
        },
        "dims": {
            "low": low_dims,
            "high": high_dims,
            "shared": shared_dims,
        },
        "orth_dims": {
            "low_orth": Q_low_orth.shape[1],
            "high_orth": Q_high_orth.shape[1],
            "shared_orth": Q_shared_orth.shape[1],
        },
    }

    out_path = Path("results_final/direction2_orthogonal.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
