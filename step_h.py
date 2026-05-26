#!/usr/bin/env python3
"""Step H: Cross-agent geometric subspace alignment via SVD(Jacobian).

Uses coordinate-invariant SVD(Jacobian) subspaces to test whether
the geometric controllability subspace is shared across agents.
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.linalg import subspace_angles

from core_mvp_v3.env import drifting_double_well
from core_mvp_v3.models import PolicyNetwork
from core_mvp_v3.experiment import reset_seed

from analysis_def import _deserialize_rng, _serialize_rng
from analysis_abc import _generate_trajectory

TOP_K = 5
EPSILON = 1e-4
SEED = 42
HIDDEN_DIM = 32


def load_trajectory_a():
    with open("results_final/trajectory_full.pkl", "rb") as f:
        return pickle.load(f)


def generate_trajectory_b(seed=1, total_steps=8000):
    net = PolicyNetwork(hidden_dim=HIDDEN_DIM)
    sd = torch.load("results_final/phase0_policy_net_seed1.pt",
                    map_location="cpu", weights_only=True)
    net.load_state_dict(sd)
    net.eval()
    env_factory = lambda: drifting_double_well(noise=0.05)
    traj = _generate_trajectory(net, env_factory, total_steps, seed=seed)
    return traj


def compute_jacobians(traj, policy_net, epsilon, step_stride=10):
    jac_env = drifting_double_well(noise=0.05)
    jac_list = []
    drift_list = []
    steps_used = list(range(0, len(traj), step_stride))
    for si in steps_used:
        t = traj[si]
        drift_list.append(t["g_t"])
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
    return np.array(jac_list), np.array(drift_list)


def partition_by_drift(drift_arr, threshold=0.5):
    return np.where(drift_arr < threshold)[0], np.where(drift_arr >= threshold)[0]


def subspace_from_jacobian_svd(J_arr, k):
    _, _, Vt = np.linalg.svd(J_arr, full_matrices=False)
    return Vt[:k, :].T


def analyze_pair(name_a, name_b, U_a, U_b, k):
    angles_rad = subspace_angles(U_a, U_b)
    angles_deg = np.degrees(angles_rad)
    mean_angle = float(np.mean(angles_deg))
    max_angle = float(np.max(angles_deg))
    frob = float(np.linalg.norm(U_a.T @ U_b, 'fro'))
    alignment_norm = frob / np.sqrt(k)
    return {
        "angles_deg": angles_deg.tolist(),
        "mean_angle": mean_angle,
        "max_angle": max_angle,
        "frobenius_norm": frob,
        "alignment_normalized": alignment_norm,
    }


def main():
    print("=" * 60)
    print("  STEP H: CROSS-AGENT GEOMETRIC SUBSPACE ALIGNMENT")
    print("=" * 60)

    np.random.seed(SEED)
    reset_seed(SEED)

    # Load agents
    policy_a = PolicyNetwork(hidden_dim=HIDDEN_DIM)
    sd_a = torch.load("results_final/phase0_policy_net.pt",
                      map_location="cpu", weights_only=True)
    policy_a.load_state_dict(sd_a)
    policy_a.eval()

    policy_b = PolicyNetwork(hidden_dim=HIDDEN_DIM)
    sd_b = torch.load("results_final/phase0_policy_net_seed1.pt",
                      map_location="cpu", weights_only=True)
    policy_b.load_state_dict(sd_b)
    policy_b.eval()

    # Load / generate trajectories
    print("  Loading trajectories...")
    traj_a = load_trajectory_a()
    traj_b = generate_trajectory_b(seed=1)
    print(f"  Agent A: {len(traj_a)} steps")
    print(f"  Agent B: {len(traj_b)} steps")

    # Compute Jacobians
    print(f"\n  Computing Jacobians (eps={EPSILON})...")
    J_a, drift_a = compute_jacobians(traj_a, policy_a, EPSILON)
    J_b, drift_b = compute_jacobians(traj_b, policy_b, EPSILON)
    print(f"  Agent A: {J_a.shape}")
    print(f"  Agent B: {J_b.shape}")

    # ── Global subspaces (all drift) ──
    print("\n─── GLOBAL (all drift) ───")
    U_a_global = subspace_from_jacobian_svd(J_a, TOP_K)
    U_b_global = subspace_from_jacobian_svd(J_b, TOP_K)
    result_global = analyze_pair("A_global", "B_global", U_a_global, U_b_global, TOP_K)
    print(f"  mean angle: {result_global['mean_angle']:.2f}°")
    print(f"  max angle:  {result_global['max_angle']:.2f}°")
    print(f"  alignment:  {result_global['alignment_normalized']:.4f}  (frob={result_global['frobenius_norm']:.4f})")
    global_shared = (result_global["alignment_normalized"] > 0.8 or
                     result_global["mean_angle"] < 30.0)
    print(f"  shared geometric subspace: {global_shared}")

    # ── Per-drift subspaces ──
    print("\n─── BY DRIFT REGIME ───")
    idx_low_a, idx_high_a = partition_by_drift(drift_a)
    idx_low_b, idx_high_b = partition_by_drift(drift_b)

    results_by_drift = {}
    for regime, (ia, ib) in [("low_drift", (idx_low_a, idx_low_b)),
                               ("high_drift", (idx_high_a, idx_high_b))]:
        if len(ia) < TOP_K or len(ib) < TOP_K:
            print(f"  {regime}: insufficient samples (A={len(ia)}, B={len(ib)}) — skipping")
            results_by_drift[regime] = {"skipped": True}
            continue

        J_a_r = J_a[ia]
        J_b_r = J_b[ib]
        U_a_r = subspace_from_jacobian_svd(J_a_r, TOP_K)
        U_b_r = subspace_from_jacobian_svd(J_b_r, TOP_K)
        r = analyze_pair(f"A_{regime}", f"B_{regime}", U_a_r, U_b_r, TOP_K)
        results_by_drift[regime] = r

        shared = (r["alignment_normalized"] > 0.8 or r["mean_angle"] < 30.0)
        print(f"  {regime}:")
        print(f"    mean angle: {r['mean_angle']:.2f}°  max: {r['max_angle']:.2f}°")
        print(f"    alignment: {r['alignment_normalized']:.4f}")
        print(f"    shared: {shared}")

    # ── Random baseline ──
    print("\n─── RANDOM BASELINE ──")
    n_rand = 1000
    rand_alignments = []
    rand_mean_angles = []
    for _ in range(n_rand):
        M = np.random.randn(HIDDEN_DIM, TOP_K)
        Qr, _ = np.linalg.qr(M)
        frob_r = float(np.linalg.norm(U_a_global.T @ Qr, 'fro'))
        rand_alignments.append(frob_r / np.sqrt(TOP_K))
        angles_r = np.degrees(subspace_angles(U_a_global, Qr))
        rand_mean_angles.append(float(np.mean(angles_r)))

    rand_alignments = np.array(rand_alignments)
    rand_mean_angles = np.array(rand_mean_angles)

    obs_alignment = result_global["alignment_normalized"]
    z_alignment = (obs_alignment - rand_alignments.mean()) / (rand_alignments.std() + 1e-8)
    p_alignment = float((rand_alignments >= obs_alignment).mean())

    obs_mean_ang = result_global["mean_angle"]
    z_angle = (rand_mean_angles.mean() - obs_mean_ang) / (rand_mean_angles.std() + 1e-8)
    p_angle = float((rand_mean_angles <= obs_mean_ang).mean())

    print(f"  random alignment: mean={rand_alignments.mean():.4f} ± {rand_alignments.std():.4f}")
    print(f"  observed alignment: {obs_alignment:.4f}  z={z_alignment:.2f}  p={p_alignment:.4f}")
    print(f"  random angle: mean={rand_mean_angles.mean():.1f}° ± {rand_mean_angles.std():.1f}°")
    print(f"  observed angle: {obs_mean_ang:.1f}°  z={z_angle:.2f}  p={p_angle:.4f}")
    print(f"  significant: {z_alignment > 2.0 or p_alignment < 0.05}")

    # ── Verdict ──
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)
    overall_shared = global_shared or any(
        r.get("alignment_normalized", 0) > 0.8 or r.get("mean_angle", 90) < 30
        for r in results_by_drift.values() if not r.get("skipped", False)
    )
    if overall_shared:
        print("  A shared geometric (Jacobian) subspace EXISTS across agents.")
    else:
        print("  No shared geometric subspace found across agents.")

    out = {
        "global": result_global,
        "by_drift": results_by_drift,
        "random_baseline": {
            "n_random": n_rand,
            "mean_random_alignment": float(rand_alignments.mean()),
            "std_random_alignment": float(rand_alignments.std()),
            "mean_random_angle": float(rand_mean_angles.mean()),
            "std_random_angle": float(rand_mean_angles.std()),
            "z_alignment": float(z_alignment),
            "p_alignment": float(p_alignment),
            "z_angle": float(z_angle),
            "p_angle": float(p_angle),
        },
        "shared_geometric_subspace": overall_shared,
        "top_k": TOP_K,
        "epsilon": EPSILON,
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/cross_agent_geometric.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  saved to results_final/cross_agent_geometric.json")


if __name__ == "__main__":
    main()
