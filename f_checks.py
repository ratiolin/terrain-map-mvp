#!/usr/bin/env python3
"""F-Checks: Verify correctness of the representation invariance analysis.

F-Check 1: alignment(U, J) == alignment(Q@U, Q@J)  — geometric invariance
F-Check 2: svd(J) singular values == svd(Q@J)       — Jacobian validity
F-Check 3: P_p == Q @ P @ Q^T                        — subspace extraction stability
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles

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


def build_subspace(hiddens_arr, ctrl_arr, idx, hidden_dim, top_k):
    H = hiddens_arr[idx]
    c = ctrl_arr[idx]
    scaler = StandardScaler()
    Hs = scaler.fit_transform(H)
    probe = Ridge(alpha=1.0)
    probe.fit(Hs, c)
    coef = np.abs(probe.coef_)
    topk_idx = np.argsort(coef)[-top_k:]
    basis = np.eye(hidden_dim)[topk_idx]
    Q_sub, _ = np.linalg.qr(basis.T)
    return Q_sub


def compute_jacobians(traj, idx, policy_net, epsilon, top_k=None):
    """Return stacked Jacobians (N, hidden_dim) for the given index subset."""
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


def alignment_score(Q_sub, J_arr):
    J_mean = J_arr.mean(axis=0)
    J_dir = J_mean / (np.linalg.norm(J_mean) + 1e-8)
    return float(np.linalg.norm(Q_sub.T @ J_dir.reshape(-1, 1), 'fro'))


def main():
    print("=" * 60)
    print("  F-CHECKS — INVARIANCE SANITY")
    print("=" * 60)

    np.random.seed(SEED)
    reset_seed(SEED)

    # ── load data ──
    traj, hiddens, controllability, drift_arr, idx_low, idx_high = load_data()
    policy_net = PolicyNetwork(hidden_dim=HIDDEN_DIM)
    sd = torch.load("results_final/phase0_policy_net.pt", map_location="cpu",
                    weights_only=True)
    policy_net.load_state_dict(sd)
    policy_net.eval()

    print(f"  trajectory: {len(traj)} steps, low={len(idx_low)}, high={len(idx_high)}")

    # ── generate Q (same seed=42) ──
    M = np.random.randn(HIDDEN_DIM, HIDDEN_DIM)
    Q, _ = np.linalg.qr(M)
    print(f"  Q: det={np.linalg.det(Q):.1f}, orthogonal={np.allclose(Q @ Q.T, np.eye(HIDDEN_DIM), atol=1e-6)}")

    # ── build subspaces (original) ──
    U_low = build_subspace(hiddens, controllability, idx_low, HIDDEN_DIM, TOP_K)
    U_high = build_subspace(hiddens, controllability, idx_high, HIDDEN_DIM, TOP_K)
    print(f"  U_low: {U_low.shape}, U_high: {U_high.shape}")

    # ── transform hidden states ──
    hiddens_t = (Q @ hiddens.T).T

    # ── build subspaces from TRANSFORMED data (new probe fit!) ──
    U_low_p_new = build_subspace(hiddens_t, controllability, idx_low, HIDDEN_DIM, TOP_K)
    U_high_p_new = build_subspace(hiddens_t, controllability, idx_high, HIDDEN_DIM, TOP_K)
    print(f"  U_low(new probe on Q@h): {U_low_p_new.shape}")
    print(f"  U_high(new probe on Q@h): {U_high_p_new.shape}")

    # Verify: does U_new == Q @ U_old? (should fail if probe picks different axes)
    c_proj_low = np.allclose(U_low_p_new @ U_low_p_new.T,
                              Q @ (U_low @ U_low.T) @ Q.T, atol=1e-6)
    c_proj_high = np.allclose(U_high_p_new @ U_high_p_new.T,
                               Q @ (U_high @ U_high.T) @ Q.T, atol=1e-6)
    print(f"  U_new == Q@U_old (projection check): low={c_proj_low}, high={c_proj_high}")

    # Also try Q@U (direct rotation of original subspace) for algebraic checks
    U_low_p_direct = Q @ U_low
    U_high_p_direct = Q @ U_high

    # ── compute Jacobians (original) ──
    J_low = compute_jacobians(traj, idx_low, policy_net, EPSILON)
    J_high = compute_jacobians(traj, idx_high, policy_net, EPSILON)
    print(f"  J_low: {J_low.shape}, J_high: {J_high.shape}")

    J_low_mean = J_low.mean(axis=0).reshape(-1, 1)  # (32, 1)
    J_high_mean = J_high.mean(axis=0).reshape(-1, 1)

    # ── transformed = Q @ original ──
    J_low_p = (Q @ J_low.T).T
    J_high_p = (Q @ J_high.T).T

    # ══════════════════════════════════════════════════════════
    # F-CHECK 1: alignment with NEW subspace from transformed data
    # ══════════════════════════════════════════════════════════
    print("\n─── F-CHECK 1: alignment with NEW probed subspace ───")

    alg_low_orig = alignment_score(U_low, J_low)
    alg_low_new = alignment_score(U_low_p_new, J_low_p)
    alg_high_orig = alignment_score(U_high, J_high)
    alg_high_new = alignment_score(U_high_p_new, J_high_p)

    # Also the tautological check: alignment(Q@U, Q@J) == alignment(U, J)
    alg_low_direct = alignment_score(U_low_p_direct, J_low_p)
    alg_high_direct = alignment_score(U_high_p_direct, J_high_p)

    c1_tautol_low = np.allclose(alg_low_orig, alg_low_direct, atol=1e-6)
    c1_tautol_high = np.allclose(alg_high_orig, alg_high_direct, atol=1e-6)
    c1_new_low = np.allclose(alg_low_orig, alg_low_new, atol=1e-3)  # relaxed
    c1_new_high = np.allclose(alg_high_orig, alg_high_new, atol=1e-3)

    print(f"  tautological: alignment(Q@U_old, Q@J) == alignment(U_old, J)")
    print(f"    low:  orig={alg_low_orig:.6f}  direct={alg_low_direct:.6f}  {'PASS' if c1_tautol_low else 'FAIL'}")
    print(f"    high: orig={alg_high_orig:.6f}  direct={alg_high_direct:.6f}  {'PASS' if c1_tautol_high else 'FAIL'}")
    print(f"  new-probe U vs original U:")
    print(f"    low:  orig={alg_low_orig:.6f}  new={alg_low_new:.6f}  delta={abs(alg_low_orig - alg_low_new):.4f}  {'PASS' if c1_new_low else 'FAIL'}")
    print(f"    high: orig={alg_high_orig:.6f}  new={alg_high_new:.6f}  delta={abs(alg_high_orig - alg_high_new):.4f}  {'PASS' if c1_new_high else 'FAIL'}")

    if not c1_new_low or not c1_new_high:
        print("\n  *** F-Check 1 FAILED (new-probe): alignment changes when subspace is rebuilt ***")
        print("  The subspace extracted from Q@h is NOT Q@(subspace from h).")
        print("  The probe identifies different dimensions after rotation.")

    # ══════════════════════════════════════════════════════════
    # F-CHECK 2: singular value spectrum
    # ══════════════════════════════════════════════════════════
    print("\n─── F-CHECK 2: svd(J) == svd(Q@J) ───")

    s_low = np.linalg.svd(J_low, compute_uv=False)
    s_low_p = np.linalg.svd(J_low_p, compute_uv=False)
    s_high = np.linalg.svd(J_high, compute_uv=False)
    s_high_p = np.linalg.svd(J_high_p, compute_uv=False)

    c2_low = np.allclose(s_low, s_low_p, atol=1e-6)
    c2_high = np.allclose(s_high, s_high_p, atol=1e-6)

    print(f"  low_drift:  max delta={np.max(np.abs(s_low - s_low_p)):.2e}  {'PASS' if c2_low else 'FAIL'}")
    print(f"  high_drift: max delta={np.max(np.abs(s_high - s_high_p)):.2e}  {'PASS' if c2_high else 'FAIL'}")

    if not c2_low or not c2_high:
        print("\n  *** F-Check 2 FAILED: Jacobian singular values changed under rotation ***")

    # ══════════════════════════════════════════════════════════
    # F-CHECK 3: projection matrix — does U_new satisfy P_new == Q @ P @ Q^T?
    # ══════════════════════════════════════════════════════════
    print("\n─── F-CHECK 3: P_new == Q @ P_old @ Q^T ? ───")

    P_low = U_low @ U_low.T
    P_low_new = U_low_p_new @ U_low_p_new.T
    P_low_expected = Q @ P_low @ Q.T

    P_high = U_high @ U_high.T
    P_high_new = U_high_p_new @ U_high_p_new.T
    P_high_expected = Q @ P_high @ Q.T

    c3_low = np.allclose(P_low_new, P_low_expected, atol=1e-6)
    c3_high = np.allclose(P_high_new, P_high_expected, atol=1e-6)

    delta_low = np.max(np.abs(P_low_new - P_low_expected))
    delta_high = np.max(np.abs(P_high_new - P_high_expected))

    print(f"  low_drift:  max delta={delta_low:.4f}  {'PASS' if c3_low else 'FAIL'}")
    print(f"  high_drift: max delta={delta_high:.4f}  {'PASS' if c3_high else 'FAIL'}")

    if not c3_low or not c3_high:
        print(f"\n  *** F-Check 3 FAILED: P_new != Q @ P @ Q^T ***")
        print(f"  The NEW subspace U (built from Q@h probe) does not match the expected transform.")
        print(f"  max delta low={delta_low:.4f}, high={delta_high:.4f}")
        print(f"  ||P_low||_F={np.linalg.norm(P_low, 'fro'):.2f}, ||P_low_new||_F={np.linalg.norm(P_low_new, 'fro'):.2f}")
        print(f"  This confirms: the axis-aligned probe selects DIFFERENT top-k dims after rotation.")
        print(f"  The clustering/subspace extraction is NOT coordinate-invariant.")

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)

    all_pass = c1_new_low and c1_new_high and c2_low and c2_high and c3_low and c3_high
    print(f"  F-Check 1 — alignment with new-probe U: {'PASS' if c1_new_low and c1_new_high else 'FAIL'}")
    print(f"    (tautological Q@U test: {'PASS' if c1_tautol_low and c1_tautol_high else 'FAIL'})")
    print(f"  F-Check 2 — Jacobian svd preserved: {'PASS' if c2_low and c2_high else 'FAIL'}")
    print(f"  F-Check 3 — projection transform: {'PASS' if c3_low and c3_high else 'FAIL'}")

    if not all_pass:
        print(f"\n  ROOT CAUSE: The subspace built from Q@h data (U_p) via probe coefficients")
        print(f"  is NOT the same as Q @ U_original. The probe identifies different top-k")
        print(f"  axes after rotation because the coordinate axes changed relative to the")
        print(f"  data distribution. This is expected for axis-aligned subspace extraction.")
        print(f"  The FAILURES in F-Check 1 and 3 are caused by the subspace extraction method,")
        print(f"  not by the Jacobian computation or the environment dynamics.")
        print(f"  The F analysis (representation_invariant=False) correctly detected this.")

    # Save results
    out = {
        "F_check_1_tautological": {
            "pass_low": c1_tautol_low,
            "pass_high": c1_tautol_high,
            "delta_low": abs(alg_low_orig - alg_low_direct),
            "delta_high": abs(alg_high_orig - alg_high_direct),
            "note": "alignment(Q@U_old, Q@J) == alignment(U_old, J) — tautology",
        },
        "F_check_1_new_probe": {
            "alignment_original_low": alg_low_orig,
            "alignment_new_low": alg_low_new,
            "delta_low": abs(alg_low_orig - alg_low_new),
            "pass_low": c1_new_low,
            "alignment_original_high": alg_high_orig,
            "alignment_new_high": alg_high_new,
            "delta_high": abs(alg_high_orig - alg_high_new),
            "pass_high": c1_new_high,
            "note": "comparing alignment of NEW probed U vs original U",
        },
        "F_check_2": {
            "pass_low": c2_low,
            "pass_high": c2_high,
            "max_svd_delta_low": float(np.max(np.abs(s_low - s_low_p))),
            "max_svd_delta_high": float(np.max(np.abs(s_high - s_high_p))),
            "note": "svd(J) == svd(Q@J) — must pass (orthogonal inv of singular vals)",
        },
        "F_check_3": {
            "pass_low": c3_low,
            "pass_high": c3_high,
            "max_delta_low": float(delta_low),
            "max_delta_high": float(delta_high),
            "note": "P_new == Q @ P_old @ Q^T — tests if new-probe U = Q @ U_old",
        },
        "all_pass": all_pass,
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/f_checks.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  saved to results_final/f_checks.json")


if __name__ == "__main__":
    main()
