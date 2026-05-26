#!/usr/bin/env python3
"""Step G: Coordinate-invariant subspace extraction via SVD(J) and PLS.

Replaces axis-aligned probe-based U with methods that are invariant
under orthogonal transformation, then re-tests F invariance.
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from core_mvp_v3.env import drifting_double_well
from core_mvp_v3.models import PolicyNetwork
from core_mvp_v3.experiment import reset_seed

from analysis_def import _deserialize_rng

TOP_K = 5
EPSILON = 1e-4
SEED = 42
HIDDEN_DIM = 32


def load_data():
    with open("results_final/trajectory_full.pkl", "rb") as f:
        traj = pickle.load(f)
    hiddens = np.array([t["h_t"] for t in traj])
    controllability = np.array([t["controllability"] for t in traj])
    drift_arr = np.array([t["g_t"] for t in traj])
    idx_low = np.where(drift_arr < 0.5)[0]
    idx_high = np.where(drift_arr >= 0.5)[0]
    return traj, hiddens, controllability, drift_arr, idx_low, idx_high


def compute_jacobians(traj, idx, policy_net, epsilon):
    jac_env = drifting_double_well(noise=0.05)
    subsample = max(1, len(idx) // 50)
    idx_ss = idx[::subsample]
    jac_list = []
    for si in idx_ss:
        t = traj[si]
        saved = {
            "state": np.array(t["env_state"]["state"], dtype=np.float32),
            "t": t["env_state"]["t"],
            "_segment_idx": t["env_state"]["_segment_idx"],
            "_segment_t": t["env_state"]["_segment_t"],
            "current_drift": t["env_state"]["current_drift"],
            "rng_state": _deserialize_rng(t["env_state"]["rng_state"]),
        }
        jac_env.restore_state(saved)
        a_plus = t["action_t"] + epsilon
        s_plus = jac_env.step(a_plus)
        jac_env.restore_state(saved)
        a_minus = t["action_t"] - epsilon
        s_minus = jac_env.step(a_minus)
        st_p = torch.tensor(s_plus, dtype=torch.float32).unsqueeze(0)
        st_m = torch.tensor(s_minus, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            h_p = policy_net.backbone(st_p).squeeze(0).numpy()
            h_m = policy_net.backbone(st_m).squeeze(0).numpy()
        J = (h_p - h_m) / (2.0 * epsilon)
        jac_list.append(J)
    return np.array(jac_list)


def alignment_score(U_sub, J_arr):
    J_mean = J_arr.mean(axis=0)
    J_dir = J_mean / (np.linalg.norm(J_mean) + 1e-8)
    return float(np.linalg.norm(U_sub.T @ J_dir.reshape(-1, 1), 'fro'))


# ─── Method A: SVD of stacked Jacobian ───

def subspace_from_jacobian_svd(J_arr, k):
    J_stacked = J_arr  # (N, D)
    _, _, Vt = np.linalg.svd(J_stacked, full_matrices=False)
    U_svd = Vt[:k, :].T  # (D, k) — top-k right singular vectors
    return U_svd


# ─── Method B: PLS between hidden states and controllability ───

def subspace_from_pls(H, c, k):
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)
    c_centered = c - c.mean()

    if k > H_scaled.shape[1]:
        k = H_scaled.shape[1]
    if k > len(c):
        k = len(c) - 1
    if k < 1:
        k = 1

    try:
        pls = PLSRegression(n_components=k, scale=False)
        pls.fit(H_scaled, c_centered.reshape(-1, 1))
        U_pls = pls.x_weights_  # (D, k)
        U_pls, _ = np.linalg.qr(U_pls)  # orthonormalize
        return U_pls
    except Exception:
        # Fallback: covariance-based
        C = H_scaled.T @ c_centered / len(c)
        U_cov = C.reshape(-1, 1) / (np.linalg.norm(C) + 1e-8)
        # Pad to k dimensions with random orthogonal directions
        if k > 1:
            M = np.random.randn(HIDDEN_DIM, k - 1)
            M = M - U_cov @ (U_cov.T @ M)
            M, _ = np.linalg.qr(M)
            U_pls = np.concatenate([U_cov, M], axis=1)
        else:
            U_pls = U_cov
        return U_pls


# ─── Build probe-based subspace (original axis-aligned) for comparison ───

def subspace_from_probe(H, c, k):
    from sklearn.linear_model import Ridge
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)
    probe = Ridge(alpha=1.0)
    probe.fit(H_scaled, c)
    coef = np.abs(probe.coef_)
    topk_idx = np.argsort(coef)[-k:]
    basis = np.eye(HIDDEN_DIM)[topk_idx]
    Q_sub, _ = np.linalg.qr(basis.T)
    return Q_sub


# ─── F-invariance test for a given subspace builder ───

def test_invariance(name, U_builder, hiddens, controllability, J_arr, Q,
                    idx_low, idx_high, k):
    H = hiddens
    c = controllability
    J = J_arr

    print(f"\n─── {name} ───")

    # Build U from original data
    U = U_builder(H, c, J, k)
    alg_orig = alignment_score(U, J)

    # Transform: h_p = Q @ h, J_p = Q @ J
    H_p = (Q @ H.T).T
    J_p_arr = (Q @ J.T).T

    U_p_new = U_builder(H_p, c, J_p_arr, k)
    alg_new = alignment_score(U_p_new, J_p_arr)

    # Algebraic: U_p_direct = Q @ U
    U_p_direct = Q @ U
    alg_direct = alignment_score(U_p_direct, J_p_arr)

    # Overlap: is U_p_new == Q @ U?
    overlap = float(np.linalg.norm(U_p_new.conj().T @ U_p_direct, 'fro'))
    max_overlap = np.sqrt(k)
    overlap_frac = overlap / max_overlap

    delta_new = abs(alg_orig - alg_new)
    delta_direct = abs(alg_orig - alg_direct)

    invariant = np.allclose(alg_orig, alg_new, atol=1e-4) or np.allclose(alg_orig, alg_new, rtol=0.01)

    print(f"  alignment: orig={alg_orig:.6f}  direct(Q@U)={alg_direct:.6f}  new-probe={alg_new:.6f}")
    print(f"  delta_direct: {delta_direct:.2e}  delta_new: {delta_new:.2e}")
    print(f"  overlap(new vs Q@U): {overlap:.4f}  (max={max_overlap:.4f}, frac={overlap_frac:.2%})")
    print(f"  invariant: {invariant}")

    return {
        "alignment_orig": alg_orig,
        "alignment_direct": alg_direct,
        "alignment_new": alg_new,
        "delta_direct": delta_direct,
        "delta_new": delta_new,
        "overlap_frac": overlap_frac,
        "invariant": invariant,
    }


# ─── Main ───

def main():
    print("=" * 60)
    print("  STEP G: COORDINATE-INVARIANT SUBSPACES")
    print("=" * 60)

    np.random.seed(SEED)
    reset_seed(SEED)

    traj, hiddens, controllability, drift_arr, idx_low, idx_high = load_data()
    policy_net = PolicyNetwork(hidden_dim=HIDDEN_DIM)
    sd = torch.load("results_final/phase0_policy_net.pt", map_location="cpu",
                    weights_only=True)
    policy_net.load_state_dict(sd)
    policy_net.eval()

    # Compute Jacobians (full trajectory)
    all_idx = np.arange(len(traj))
    J_all = compute_jacobians(traj, all_idx, policy_net, EPSILON)
    J_low = compute_jacobians(traj, idx_low, policy_net, EPSILON)
    J_high = compute_jacobians(traj, idx_high, policy_net, EPSILON)
    print(f"  J_all: {J_all.shape}, J_low: {J_low.shape}, J_high: {J_high.shape}")

    # Generate Q
    M = np.random.randn(HIDDEN_DIM, HIDDEN_DIM)
    Q, _ = np.linalg.qr(M)
    print(f"  Q: orthogonal={np.allclose(Q @ Q.T, np.eye(HIDDEN_DIM), atol=1e-6)}")

    # ── Builders ──
    def builder_probe(H, c, J, k):
        return subspace_from_probe(H, c, k)

    def builder_svd_jac(H, c, J, k):
        return subspace_from_jacobian_svd(J, k)

    def builder_pls(H, c, J, k):
        return subspace_from_pls(H, c, k)

    methods = {
        "axis_probe": builder_probe,
        "svd_jacobian": builder_svd_jac,
        "pls_hidden_ctrl": builder_pls,
    }

    results = {}
    for method_name, builder in methods.items():
        print(f"\n{'='*40}")
        print(f"  Method: {method_name}")
        print(f"{'='*40}")
        res = test_invariance(method_name, builder, hiddens, controllability,
                              J_all, Q, idx_low, idx_high, TOP_K)
        results[method_name] = res

    # ── Verdict ──
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)

    geom_exists = results["svd_jacobian"]["invariant"] or results["pls_hidden_ctrl"]["invariant"]

    for name, r in results.items():
        mark = "PASS (invariant)" if r["invariant"] else "FAIL (not invariant)"
        print(f"  {name}: {mark}  overlap={r['overlap_frac']:.2%}  delta={r['delta_new']:.2e}")

    if geom_exists:
        print(f"\n  A coordinate-invariant geometric subspace EXISTS.")
        print(f"  The previous axis-aligned probe method was the wrong tool.")
    else:
        print(f"\n  No coordinate-invariant subspace found.")
        print(f"  The structure is purely axis-dependent — emergent, agent-specific,")
        print(f"  and tied to the specific coordinate frame of each agent.")

    out = {
        "methods": {name: r for name, r in results.items()},
        "geometric_subspace_exists": geom_exists,
        "top_k": TOP_K,
        "Q_orthogonal": bool(np.allclose(Q @ Q.T, np.eye(HIDDEN_DIM), atol=1e-6)),
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/step_g_invariance.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  saved to results_final/step_g_invariance.json")


if __name__ == "__main__":
    main()
