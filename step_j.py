#!/usr/bin/env python3
"""Step J: Hidden dimension sweep — find capacity threshold for geometric structure.

Trains agents at different hidden_dim values [16, 32, 64, 128],
computes Jacobian subspace stability, cluster strength, alignment,
and cross-agent alignment. Finds the minimum dim where the geometric
controllability subspace emerges.
"""

import sys
import json
import pickle
import copy
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_mvp_v3.env import drifting_double_well
from core_mvp_v3.models import PolicyNetwork
from core_mvp_v3.experiment import (
    reset_seed, ExperimentConfig, train, train_minimal,
)

from step_h import (
    compute_jacobians, subspace_from_jacobian_svd,
    partition_by_drift, analyze_pair,
)
from step_i import (
    load_traj, compute_all_jacobians,
    pairwise_principal_angle_mean, cluster_subspaces,
    intra_inter,
)

WIN = 500
STR = 250
K_TOP = 5
EPS = 1e-4
SEED = 42
TOTAL_STEPS = 4000  # shorter for speed


def train_agent(hidden_dim, seed, steps):
    reset_seed(seed)
    cfg = ExperimentConfig()
    cfg.num_episodes = 1
    cfg.episode_length = steps
    cfg.policy_hidden_dim = hidden_dim
    cfg.pred_hidden_dim = hidden_dim
    cfg.seed = seed
    logs, _, policy_net, _ = train(cfg)
    return policy_net


def generate_traj_for_agent(policy_net, seed, steps):
    from analysis_abc import _generate_trajectory, _serialize_rng
    env_factory = lambda: drifting_double_well(noise=0.05)
    traj = _generate_trajectory(policy_net, env_factory, steps,
                                 seed=seed)
    return traj


def compute_step_jac_angles(J_all):
    angles = []
    for i in range(len(J_all) - 1):
        ni = np.linalg.norm(J_all[i]) + 1e-8
        nj = np.linalg.norm(J_all[i + 1]) + 1e-8
        c = np.abs(np.dot(J_all[i], J_all[i + 1])) / (ni * nj)
        c = min(1.0, max(0.0, c))
        angles.append(float(np.degrees(np.arccos(c))))
    return float(np.mean(angles)) if angles else 0.0


def analyze_agent(traj, policy_net, hidden_dim, jac_stride=10):
    T = len(traj)
    drift_all = np.array([t["g_t"] for t in traj])
    J_all = compute_all_jacobians(traj, policy_net, EPS, stride=jac_stride)
    n_jac = J_all.shape[0]

    mean_step_angle = compute_step_jac_angles(J_all)

    wi = WIN // jac_stride
    si = STR // jac_stride
    windows = [(s, s + wi) for s in range(0, n_jac - wi + 1, si)]
    if not windows:
        return None
    subs = []
    for ws, we in windows:
        Jw = J_all[ws:we]
        _, _, Vt = np.linalg.svd(Jw, full_matrices=False)
        k_eff = min(K_TOP, Vt.shape[0])
        U = Vt[:k_eff, :].T
        subs.append(U)
    if len(subs) < 2:
        return None
    D = pairwise_principal_angle_mean(subs)
    nk, labels, sil = cluster_subspaces(subs, D, max_k=3, seed=SEED)
    intra_j, inter_j, ni, nj = intra_inter(D, labels)
    ratio_j = inter_j / (intra_j + 1e-8)

    n_bins = 10
    window_centers = np.array([(ws + wi // 2) * jac_stride for ws, _ in windows])
    drift_at_centers = np.array([drift_all[min(int(c), T - 1)] for c in window_centers])
    drift_bins = np.digitize(drift_at_centers, np.linspace(0, 2, n_bins + 1))
    from sklearn.metrics import mutual_info_score
    mi = mutual_info_score(np.array(labels), drift_bins)

    structure = sil > 0.4 and ratio_j > 2.0

    # Within-agent Jacobian alignment (high drift only for consistency)
    idx_low, idx_high = partition_by_drift(
        np.array([drift_all[min(s, T - 1)] for s in range(0, T, jac_stride)])[:n_jac])
    if len(idx_high) >= K_TOP:
        J_high = J_all[idx_high]
        U_high = subspace_from_jacobian_svd(J_high, K_TOP)
        J_high_mean = J_high.mean(axis=0)
        J_high_dir = J_high_mean / (np.linalg.norm(J_high_mean) + 1e-8)
        alignment_high = float(np.linalg.norm(U_high.T @ J_high_dir.reshape(-1, 1), 'fro'))
    else:
        alignment_high = 0.0

    return {
        "hidden_dim": hidden_dim,
        "num_clusters": nk,
        "silhouette": sil,
        "intra_angle": intra_j,
        "inter_angle": inter_j,
        "ratio": ratio_j,
        "mutual_info": float(mi),
        "structure_present": structure,
        "mean_step_jacobian_angle": mean_step_angle,
        "jacobian_alignment_high": alignment_high,
        "n_windows": len(subs),
        "jacobian_count": n_jac,
    }


def main():
    print("=" * 60)
    print("  STEP J: HIDDEN DIMENSION SWEEP")
    print("=" * 60)

    dims = [16, 32, 64, 128]
    results = []

    for hd in dims:
        t0 = time.time()
        print(f"\n─── hidden_dim={hd} ───")
        print(f"  training agent (seed={SEED}, steps={TOTAL_STEPS})...")

        policy = train_agent(hd, SEED, TOTAL_STEPS)
        print(f"  generating trajectory...")
        traj = generate_traj_for_agent(policy, SEED, TOTAL_STEPS)

        print(f"  analyzing Jacobian subspaces...")
        r = analyze_agent(traj, policy, hd)
        if r:
            results.append(r)
            elapsed = time.time() - t0
            print(f"  K={r['num_clusters']} sil={r['silhouette']:.3f} "
                  f"ratio={r['ratio']:.2f} structure={r['structure_present']} "
                  f"align={r['jacobian_alignment_high']:.3f} "
                  f"({elapsed:.0f}s)")
        else:
            print(f"  FAILED (insufficient windows)")
            results.append({"hidden_dim": hd, "error": "insufficient data"})

    # ── Find capacity threshold ──
    print("\n" + "=" * 60)
    print("  CAPACITY THRESHOLD")
    print("=" * 60)

    valid = [r for r in results if "structure_present" in r]
    threshold = None
    for r in valid:
        if r["structure_present"] and r["jacobian_alignment_high"] > 0.5:
            if threshold is None or r["hidden_dim"] < threshold:
                threshold = r["hidden_dim"]

    print(f"  dims scanned: {dims}")
    for r in valid:
        flags = []
        if r["structure_present"]:
            flags.append("STRUCTURE")
        if r["jacobian_alignment_high"] > 0.5:
            flags.append("ALIGNED")
        print(f"  dim={r['hidden_dim']:4d}: sil={r['silhouette']:.3f}  "
              f"ratio={r['ratio']:.2f}  align={r['jacobian_alignment_high']:.3f}  "
              f"{' '.join(flags)}")
    if threshold:
        print(f"\n  CAPACITY THRESHOLD d* = {threshold}")
        print(f"  Geometric controllability subspace emerges at hidden_dim >= {threshold}")
    else:
        print(f"\n  No clear threshold found — all dims have structure (or none do)")

    out = {
        "dimensions_scanned": dims,
        "results": results,
        "capacity_threshold": threshold,
        "total_steps": TOTAL_STEPS,
        "top_k": K_TOP,
        "window": WIN,
        "stride": STR,
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/dimension_sweep.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  saved to results_final/dimension_sweep.json")


if __name__ == "__main__":
    main()
