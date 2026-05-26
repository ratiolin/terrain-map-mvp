#!/usr/bin/env python3
"""Analysis D/E/F — Signal Validity, Behavioral Consistency, Representation Invariance."""

import json
import sys
import pickle
import random
import math
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles, orth
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, mutual_info_score, adjusted_rand_score
from sklearn.manifold import MDS

from core_mvp_v3.env import drifting_double_well
from core_mvp_v3.models import PolicyNetwork
from core_mvp_v3.experiment import reset_seed

from analysis_abc import (
    _load_trajectory, _load_policy, _serialize_rng, _deserialize_rng,
    _build_window_subspaces, _pairwise_principal_angle_mean,
    _cluster_subspaces, _intra_inter, _generate_trajectory,
    _subspace_clusters_from_traj,
)


def _safe_load_json(path):
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# D. SIGNAL VALIDITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_signal_validity(seed=42, window=500, stride=250, top_k=5):
    print("=" * 60)
    print("  D. SIGNAL VALIDITY CHECK")
    print("=" * 60)

    np.random.seed(seed)
    random.seed(seed)

    traj = _load_trajectory()
    T = len(traj)

    clusters = _safe_load_json("results_final/subspace_clusters.json")
    labels = np.array(clusters["labels"])
    window_starts = np.array(clusters["window_starts"])

    controllability = np.array([t["controllability"] for t in traj])
    ctrl_max = float(controllability.max())
    ctrl_min = float(controllability.min())
    ctrl_range = ctrl_max - ctrl_min

    if ctrl_range < 1e-8:
        panic_signal = np.zeros(T)
    else:
        panic_signal = (ctrl_max - controllability) / ctrl_range

    # Compute validity metrics
    std_val = float(panic_signal.std())
    max_val = float(panic_signal.max())
    min_val = float(panic_signal.min())
    saturated = (std_val < 0.05) or ((max_val - min_val) < 0.1)

    print(f"  controllability range: [{ctrl_min:.5f}, {ctrl_max:.5f}]")
    print(f"  panic signal: min={min_val:.4f} max={max_val:.4f} std={std_val:.4f}")
    print(f"  saturated: {saturated}")

    # Build alternative signals
    alternatives = {
        "controllability_raw": controllability,
        "risk": np.array([abs(t["obs_t"][0]) for t in traj]),
        "action_magnitude": np.array([abs(t["action_t"]) for t in traj]),
        "drift_g": np.array([t["g_t"] for t in traj]),
    }

    # Score each signal: P(signal_high | cluster) separation + mutual_info
    n_bins = 10
    drift_binned = np.digitize(np.array([t["g_t"] for t in traj]),
                               np.linspace(0, 2, n_bins + 1))

    signal_scores = {}
    best_signal_name = "panic_inverted"
    best_mi = -1.0

    for name, signal in alternatives.items():
        window_signal = np.array([signal[s:min(s + window, T)].mean()
                                   for s in window_starts])
        sig_median = np.median(window_signal)
        sig_high = (window_signal > sig_median).astype(int)

        # P(signal_high | cluster)
        p_high_given = {}
        for cid in sorted(set(labels)):
            mask = labels == cid
            p_high_given[int(cid)] = float(sig_high[mask].mean()) if mask.any() else 0.0

        # mutual_info
        sig_binned = np.digitize(window_signal,
                                 np.linspace(window_signal.min(), window_signal.max(), n_bins + 1))
        mi = mutual_info_score(labels, sig_binned)

        # per-cluster stats
        per_cluster_mean = {}
        per_cluster_std = {}
        for cid in sorted(set(labels)):
            mask = labels == cid
            per_cluster_mean[int(cid)] = float(window_signal[mask].mean()) if mask.any() else 0.0
            per_cluster_std[int(cid)] = float(window_signal[mask].std()) if mask.any() else 0.0

        signal_scores[name] = {
            "mutual_info": float(mi),
            "P_signal_high_given_cluster": p_high_given,
            "per_cluster_mean": per_cluster_mean,
            "per_cluster_std": per_cluster_std,
        }

        print(f"\n  [{name}]")
        print(f"    mutual_info: {mi:.4f}")
        print(f"    P(high|cluster): {p_high_given}")
        print(f"    per-cluster mean: {per_cluster_mean}")

        if mi > best_mi and name != "drift_g":
            best_mi = mi
            best_signal_name = name

    # drift_g is the oracle (we know drift correlates with clusters), exclude from "best"
    used_signal = best_signal_name
    print(f"\n  selected signal: '{used_signal}' (mutual_info={best_mi:.4f})")

    out = {
        "original_panic": {
            "min": min_val, "max": max_val, "std": std_val,
            "saturated": saturated,
        },
        "alternative_signals": signal_scores,
        "selected_signal": used_signal,
        "selected_mutual_info": float(best_mi),
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/signal_validity.json", "w") as f:
        json.dump(out, f, indent=2)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# E. BEHAVIORAL CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_behavioral_consistency(seed=42, total_steps=8000):
    print("=" * 60)
    print("  E. BEHAVIORAL CONSISTENCY")
    print("=" * 60)

    reset_seed(seed)

    policy_a = _load_policy("results_final/phase0_policy_net.pt")
    policy_b = _load_policy("results_final/phase0_policy_net_seed1.pt")

    # Run both agents from identical initial state
    env_seed = 42
    env_a = drifting_double_well(noise=0.05)
    env_a.set_rng_seed(env_seed)
    env_a.reset()
    env_b = env_a.clone()

    states_a, actions_a, hiddens_a = [], [], []
    states_b, actions_b, hiddens_b = [], [], []

    drift_seq = []

    for step in range(total_steps):
        drift_seq.append(env_a.current_drift)

        st_a = env_a.state.copy()
        st_t_a = torch.tensor(st_a, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            ash_a, aad_a, h_a = policy_a(st_t_a)
        av_a = float(ash_a.item())
        states_a.append(st_a)
        actions_a.append(av_a)
        hiddens_a.append(h_a.detach().cpu().numpy().squeeze(0))
        env_a.step(av_a)

        st_b = env_b.state.copy()
        st_t_b = torch.tensor(st_b, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            ash_b, aad_b, h_b = policy_b(st_t_b)
        av_b = float(ash_b.item())
        states_b.append(st_b)
        actions_b.append(av_b)
        hiddens_b.append(h_b.detach().cpu().numpy().squeeze(0))
        env_b.step(av_b)

    actions_a = np.array(actions_a)
    actions_b = np.array(actions_b)
    states_a = np.array(states_a).squeeze()
    states_b = np.array(states_b).squeeze()
    drift_seq = np.array(drift_seq)

    # (a) Action similarity
    dot = np.sum(actions_a * actions_b)
    norm_a = np.sqrt(np.sum(actions_a ** 2))
    norm_b = np.sqrt(np.sum(actions_b ** 2))
    cos_sim = float(dot / (norm_a * norm_b + 1e-8)) if norm_a > 0 and norm_b > 0 else 0.0

    # Per-segment cos_sim
    segment_len = 2000
    n_segments = total_steps // segment_len
    segment_cos = []
    for si in range(n_segments):
        a_s = actions_a[si * segment_len:(si + 1) * segment_len]
        b_s = actions_b[si * segment_len:(si + 1) * segment_len]
        dot_s = np.sum(a_s * b_s)
        n_s = np.sqrt(np.sum(a_s ** 2) * np.sum(b_s ** 2) + 1e-8)
        segment_cos.append(float(dot_s / n_s))

    # (b) Switch points — use action sign flips as proxy
    def find_switches(actions_series, window=5):
        sig = np.array(actions_series)
        smoothed = np.convolve(sig, np.ones(window) / window, mode='same')
        switches = []
        for i in range(1, len(smoothed)):
            if smoothed[i - 1] * smoothed[i] < 0:
                switches.append(i)
        return np.array(switches)

    sw_a = find_switches(actions_a)
    sw_b = find_switches(actions_b)

    if len(sw_a) > 0 and len(sw_b) > 0:
        temporal_distances = []
        for sa in sw_a:
            if len(sw_b) > 0:
                nearest = sw_b[np.argmin(np.abs(sw_b - sa))]
                temporal_distances.append(abs(sa - nearest))
        mean_temporal_dist = float(np.mean(temporal_distances)) if temporal_distances else float(total_steps)
    elif len(sw_a) == 0 and len(sw_b) == 0:
        mean_temporal_dist = 0.0
        temporal_distances = []
    else:
        mean_temporal_dist = float(total_steps)
        temporal_distances = []

    # (c) Behavioral outcomes
    in_zone_thresh = 2.0
    in_zone_a = float((np.abs(states_a) < in_zone_thresh).mean())
    in_zone_b = float((np.abs(states_b) < in_zone_thresh).mean())
    mean_risk_a = float(np.abs(states_a).mean())
    mean_risk_b = float(np.abs(states_b).mean())

    # Per-regime outcomes
    low_mask = drift_seq < 0.5
    high_mask = drift_seq >= 0.5
    risk_low_a = float(np.abs(states_a[low_mask]).mean()) if low_mask.any() else 0.0
    risk_low_b = float(np.abs(states_b[low_mask]).mean()) if low_mask.any() else 0.0
    risk_high_a = float(np.abs(states_a[high_mask]).mean()) if high_mask.any() else 0.0
    risk_high_b = float(np.abs(states_b[high_mask]).mean()) if high_mask.any() else 0.0

    # Verdict
    action_similar = cos_sim > 0.7
    risk_diff = abs(mean_risk_a - mean_risk_b)
    switch_diff_pct = (mean_temporal_dist / total_steps) * 100.0
    behaviorally_consistent = (action_similar or risk_diff < 0.1) and (switch_diff_pct < 10.0)

    print(f"  action cos_sim: {cos_sim:.4f}  (>0.7: {action_similar})")
    print(f"  segment cos_sim: {segment_cos}")
    print(f"  switch points: A={len(sw_a)}  B={len(sw_b)}")
    print(f"  mean temporal distance: {mean_temporal_dist:.0f} steps ({switch_diff_pct:.1f}%)")
    print(f"  in_zone rate: A={in_zone_a:.3f}  B={in_zone_b:.3f}")
    print(f"  mean risk: A={mean_risk_a:.3f}  B={mean_risk_b:.3f}  diff={risk_diff:.3f}")
    print(f"  risk low drift: A={risk_low_a:.3f}  B={risk_low_b:.3f}")
    print(f"  risk high drift: A={risk_high_a:.3f}  B={risk_high_b:.3f}")
    print(f"  behaviorally consistent: {behaviorally_consistent}")

    out = {
        "action_cos_sim": cos_sim,
        "segment_cos_sim": segment_cos,
        "action_similarity_threshold": 0.7,
        "action_similar": action_similar,
        "n_switches_A": int(len(sw_a)),
        "n_switches_B": int(len(sw_b)),
        "mean_temporal_distance_steps": float(mean_temporal_dist),
        "temporal_distance_pct": float(switch_diff_pct),
        "switch_timing_threshold_pct": 10.0,
        "in_zone_rate_A": in_zone_a,
        "in_zone_rate_B": in_zone_b,
        "mean_risk_A": mean_risk_a,
        "mean_risk_B": mean_risk_b,
        "mean_risk_difference": float(risk_diff),
        "risk_low_drift_A": risk_low_a,
        "risk_low_drift_B": risk_low_b,
        "risk_high_drift_A": risk_high_a,
        "risk_high_drift_B": risk_high_b,
        "behaviorally_consistent": behaviorally_consistent,
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/behavior_consistency.json", "w") as f:
        json.dump(out, f, indent=2)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# F. REPRESENTATION INVARIANCE
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_invariance(seed=42, window=500, stride=250, top_k=5, epsilon=1e-4):
    print("=" * 60)
    print("  F. REPRESENTATION INVARIANCE TEST")
    print("=" * 60)

    np.random.seed(seed)
    random.seed(seed)

    traj = _load_trajectory()
    T = len(traj)
    hiddens = np.array([t["h_t"] for t in traj])  # (T, 32)
    controllability = np.array([t["controllability"] for t in traj])
    hidden_dim = hiddens.shape[1]

    # ── generate random orthogonal matrix Q ──
    M = np.random.randn(hidden_dim, hidden_dim)
    Q, _ = np.linalg.qr(M)
    print(f"  Q: shape={Q.shape}, det={np.linalg.det(Q):.2f} (should be ±1)")

    # ── transform hidden states ──
    hiddens_t = (Q @ hiddens.T).T  # (T, 32) with rotation applied
    print(f"  hidden states: original shape={hiddens.shape}, norm preserved: "
          f"{np.allclose(np.linalg.norm(hiddens, axis=1), np.linalg.norm(hiddens_t, axis=1))}")

    # ── run subspace + clustering on both ──
    print("\n  ── original (identity) ──")
    result_orig = _subspace_clusters_from_traj(traj, window, stride, top_k, seed)

    print("\n  ── transformed (Q @ h) ──")
    # Create a modified trajectory with transformed hidden states
    traj_t = []
    for i, t in enumerate(traj):
        tt = dict(t)
        tt["h_t"] = hiddens_t[i].tolist()
        traj_t.append(tt)
    result_trans = _subspace_clusters_from_traj(traj_t, window, stride, top_k, seed)

    # ── compute Jacobian alignment on both ──
    print("\n  ── Jacobian alignment comparison ──")
    policy_net = _load_policy()
    jac_env = drifting_double_well(noise=0.05)

    drift_arr = np.array([t["g_t"] for t in traj])
    idx_low = np.where(drift_arr < 0.5)[0]
    idx_high = np.where(drift_arr >= 0.5)[0]

    def compute_jacobians_for_group(idx, Q_transform=None):
        jac_list = []
        subsample = max(1, len(idx) // 50)
        idx_ss = idx[::subsample]
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
            if Q_transform is not None:
                h_p = Q_transform @ h_p
                h_m = Q_transform @ h_m
            J = (h_p - h_m) / (2.0 * epsilon)
            jac_list.append(J)
        return np.array(jac_list)

    J_orig_low = compute_jacobians_for_group(idx_low, Q_transform=None)
    J_orig_high = compute_jacobians_for_group(idx_high, Q_transform=None)
    J_trans_low = compute_jacobians_for_group(idx_low, Q_transform=Q)
    J_trans_high = compute_jacobians_for_group(idx_high, Q_transform=Q)

    # Build subspaces from probe coefficients
    def build_subspace(hiddens_arr, ctrl_arr, idx, hidden_dim):
        H = hiddens_arr[idx]
        c = ctrl_arr[idx]
        scaler = StandardScaler()
        Hs = scaler.fit_transform(H)
        probe = Ridge(alpha=1.0)
        probe.fit(Hs, c)
        coef = np.abs(probe.coef_)
        topk_idx = np.argsort(coef)[-top_k:]
        basis = np.eye(hidden_dim)[topk_idx]
        sub, _ = np.linalg.qr(basis.T)
        return sub

    def alignment_score(Q_sub, J_arr):
        J_mean = J_arr.mean(axis=0)
        J_dir = J_mean / (np.linalg.norm(J_mean) + 1e-8)
        return float(np.linalg.norm(Q_sub.T @ J_dir.reshape(-1, 1), 'fro'))

    # Original
    Q_orig_low = build_subspace(hiddens, controllability, idx_low, hidden_dim)
    Q_orig_high = build_subspace(hiddens, controllability, idx_high, hidden_dim)
    alg_orig_low = alignment_score(Q_orig_low, J_orig_low)
    alg_orig_high = alignment_score(Q_orig_high, J_orig_high)

    # Transformed
    Q_trans_low = build_subspace(hiddens_t, controllability, idx_low, hidden_dim)
    Q_trans_high = build_subspace(hiddens_t, controllability, idx_high, hidden_dim)
    alg_trans_low = alignment_score(Q_trans_low, J_trans_low)
    alg_trans_high = alignment_score(Q_trans_high, J_trans_high)

    print(f"  alignment orig:  low={alg_orig_low:.4f}  high={alg_orig_high:.4f}")
    print(f"  alignment trans: low={alg_trans_low:.4f}  high={alg_trans_high:.4f}")

    # ── verdict ──
    k_preserved = abs(result_orig["num_clusters"] - result_trans["num_clusters"]) <= 1
    sil_ok = result_trans["silhouette"] > 0.4
    ratio_ok = result_trans["inter_angle"] / (result_trans["intra_angle"] + 1e-8) > 2.0
    alg_low_ok = alg_trans_low > alg_orig_low * 0.5
    alg_high_ok = alg_trans_high > alg_orig_high * 0.5
    invariant = k_preserved and sil_ok and ratio_ok and (alg_low_ok or alg_high_ok)

    print(f"\n  K preserved (±1): {k_preserved} (orig={result_orig['num_clusters']}, trans={result_trans['num_clusters']})")
    print(f"  silhouette > 0.4: {sil_ok} ({result_trans['silhouette']:.3f})")
    print(f"  inter/intra > 2: {ratio_ok} ({result_trans['inter_angle']/(result_trans['intra_angle']+1e-8):.2f})")
    print(f"  Jacobian alignment preserved: low={alg_low_ok} high={alg_high_ok}")
    print(f"  representation-invariant: {invariant}")

    out = {
        "original": {
            "num_clusters": result_orig["num_clusters"],
            "silhouette": result_orig["silhouette"],
            "intra_angle": result_orig["intra_angle"],
            "inter_angle": result_orig["inter_angle"],
            "jacobian_alignment_low": alg_orig_low,
            "jacobian_alignment_high": alg_orig_high,
        },
        "transformed": {
            "num_clusters": result_trans["num_clusters"],
            "silhouette": result_trans["silhouette"],
            "intra_angle": result_trans["intra_angle"],
            "inter_angle": result_trans["inter_angle"],
            "jacobian_alignment_low": alg_trans_low,
            "jacobian_alignment_high": alg_trans_high,
        },
        "K_preserved": k_preserved,
        "silhouette_ok": sil_ok,
        "ratio_ok": ratio_ok,
        "jacobian_alignment_preserved_low": alg_low_ok,
        "jacobian_alignment_preserved_high": alg_high_ok,
        "representation_invariant": invariant,
        "Q_orthogonal": bool(np.allclose(Q @ Q.T, np.eye(hidden_dim), atol=1e-6)),
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/invariance_test.json", "w") as f:
        json.dump(out, f, indent=2)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    seed = 42

    print("╔" + "═" * 58 + "╗")
    print("║  ANALYSIS D/E/F — SIGNAL, BEHAVIOR, INVARIANCE")
    print("╚" + "═" * 58 + "╝")
    print()

    analyze_signal_validity(seed=seed)
    print()
    analyze_behavioral_consistency(seed=seed)
    print()
    analyze_invariance(seed=seed)
    print()

    print("=" * 60)
    print("  ALL ANALYSIS COMPLETE")
    print("=" * 60)
    print("  Outputs:")
    print("    results_final/signal_validity.json")
    print("    results_final/behavior_consistency.json")
    print("    results_final/invariance_test.json")


if __name__ == "__main__":
    main()
