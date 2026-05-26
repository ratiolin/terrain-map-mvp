#!/usr/bin/env python3
"""Step I: Jacobian subspace time dynamics.

Replaces the probe-based (axis-aligned) subspace with the coordinate-invariant
SVD(Jacobian) subspace. Tests whether the discrete clustering structure found
in Step 2 survives when using a geometrically meaningful subspace definition.
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.manifold import MDS

from core_mvp_v3.env import drifting_double_well
from core_mvp_v3.models import PolicyNetwork
from core_mvp_v3.experiment import reset_seed

from analysis_def import _deserialize_rng

WINDOW = 500
STRIDE = 250
TOP_K = 5
EPSILON = 1e-4
SEED = 42
HIDDEN_DIM = 32


def load_traj():
    with open("results_final/trajectory_full.pkl", "rb") as f:
        return pickle.load(f)


def compute_all_jacobians(traj, policy_net, epsilon, stride=1):
    jac_env = drifting_double_well(noise=0.05)
    jac_list = []
    for si in range(0, len(traj), stride):
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


def pairwise_principal_angle_mean(subspaces):
    n = len(subspaces)
    from scipy.linalg import subspace_angles
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ang = float(np.degrees(subspace_angles(subspaces[i], subspaces[j])).mean())
            D[i, j] = D[j, i] = ang
    return D


def cluster_subspaces(subspaces, D, max_k=3, seed=42):
    best_k, best_score = 2, -1.0
    best_labels = None
    for k in range(2, max_k + 1):
        try:
            aff = np.exp(-D / (D[D > 0].mean() + 1e-8))
            labels = SpectralClustering(
                n_clusters=k, affinity='precomputed', random_state=seed,
                assign_labels='kmeans').fit_predict(aff)
            if len(set(labels)) <= 1:
                continue
            emb = MDS(n_components=min(5, len(subspaces) - 1), dissimilarity='precomputed',
                      random_state=seed, normalized_stress='auto')
            pts = emb.fit_transform(D)
            sc = silhouette_score(pts, labels)
            if sc > best_score:
                best_score = sc
                best_k = k
                best_labels = labels
        except Exception:
            continue
    if best_labels is None:
        best_k = 2
        aff = np.exp(-D / (D[D > 0].mean() + 1e-8))
        best_labels = SpectralClustering(
            n_clusters=2, affinity='precomputed', random_state=seed,
            assign_labels='kmeans').fit_predict(aff)
    return int(best_k), list(best_labels), float(best_score)


def intra_inter(D, labels):
    u_lbl = sorted(set(labels))
    intra = []
    inter = []
    for cid in u_lbl:
        mem = [i for i, l in enumerate(labels) if l == cid]
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                intra.append(float(D[mem[a], mem[b]]))
    for c1 in u_lbl:
        for c2 in u_lbl:
            if c1 >= c2:
                continue
            m1 = [i for i, l in enumerate(labels) if l == c1]
            m2 = [i for i, l in enumerate(labels) if l == c2]
            for i in m1:
                for j in m2:
                    inter.append(float(D[i, j]))
    return (float(np.mean(intra)) if intra else 0.0,
            float(np.mean(inter)) if inter else 0.0,
            len(intra), len(inter))


def main():
    print("=" * 60)
    print("  STEP I: JACOBIAN SUBSPACE TIME DYNAMICS")
    print("=" * 60)

    np.random.seed(SEED)
    reset_seed(SEED)

    traj = load_traj()
    policy_net = PolicyNetwork(hidden_dim=HIDDEN_DIM)
    sd = torch.load("results_final/phase0_policy_net.pt", map_location="cpu",
                    weights_only=True)
    policy_net.load_state_dict(sd)
    policy_net.eval()

    T = len(traj)
    drift_all = np.array([t["g_t"] for t in traj])

    # ── Compute Jacobians ──
    jac_stride = 10
    print(f"  Computing Jacobians every {jac_stride} steps...")
    J_all = compute_all_jacobians(traj, policy_net, EPSILON, stride=jac_stride)
    n_jac = J_all.shape[0]
    jac_times = np.arange(0, T, jac_stride)[:n_jac]
    print(f"  J_all: {J_all.shape}  (every {jac_stride} steps)")

    # ── Between-timestep Jacobian subspace distance ──
    angles_step = []
    for i in range(n_jac - 1):
        Ji = J_all[i]
        Jj = J_all[i + 1]
        ni = np.linalg.norm(Ji) + 1e-8
        nj = np.linalg.norm(Jj) + 1e-8
        cos_ang = np.abs(np.dot(Ji, Jj)) / (ni * nj)
        cos_ang = min(1.0, max(0.0, cos_ang))
        angles_step.append(float(np.degrees(np.arccos(cos_ang))))

    mean_step_angle = float(np.mean(angles_step))
    print(f"  mean step-to-step Jacobian angle: {mean_step_angle:.2f} deg  (n={len(angles_step)})")

    # ── Window-based subspace analysis ──
    windows = []
    for start in range(0, n_jac - (WINDOW // jac_stride) + 1, STRIDE // jac_stride):
        end = start + WINDOW // jac_stride
        windows.append((start, end))

    n_windows = len(windows)
    print(f"  windows: {n_windows}  ({WINDOW} steps each, {STRIDE} stride)")

    subspaces_jac = []
    window_starts = []
    for ws, we in windows:
        J_win = J_all[ws:we]  # (W/k, D)
        _, _, Vt = np.linalg.svd(J_win, full_matrices=False)
        U = Vt[:TOP_K, :].T  # (D, k)
        subspaces_jac.append(U)
        window_starts.append(int(ws * jac_stride))

    D_jac = pairwise_principal_angle_mean(subspaces_jac)
    nk, labels_jac, sil_jac = cluster_subspaces(subspaces_jac, D_jac, max_k=3, seed=SEED)
    intra_j, inter_j, ni, nj = intra_inter(D_jac, labels_jac)
    ratio_j = inter_j / (intra_j + 1e-8)

    print(f"\n  ── Jacobian subspace clustering ──")
    print(f"  K = {nk}, silhouette = {sil_jac:.3f}")
    print(f"  intra = {intra_j:.2f} deg, inter = {inter_j:.2f} deg, ratio = {ratio_j:.2f}")

    # ── Compare with drift ──
    centers = np.array(window_starts) + WINDOW // 2
    drift_at_center = np.array([drift_all[min(c, T - 1)] for c in centers])
    n_bins = 10
    drift_bins = np.digitize(drift_at_center, np.linspace(0, 2, n_bins + 1))
    mi = mutual_info_score(np.array(labels_jac), drift_bins)

    # Per-cluster drift stats
    per_cluster_drift = {}
    for cid in sorted(set(labels_jac)):
        mask = np.array(labels_jac) == cid
        per_cluster_drift[int(cid)] = {
            "mean_drift": float(drift_at_center[mask].mean()),
            "std_drift": float(drift_at_center[mask].std()),
            "count": int(mask.sum()),
        }

    print(f"  mutual_info(cluster, drift_bin): {mi:.4f}")
    for cid, v in per_cluster_drift.items():
        print(f"  cluster {cid}: drift mean={v['mean_drift']:.3f} ± {v['std_drift']:.3f} (n={v['count']})")

    # ── Also compute the original probe-based results for comparison ──
    from analysis_abc import _subspace_clusters_from_traj
    result_probe = _subspace_clusters_from_traj(traj, WINDOW, STRIDE, TOP_K, SEED)

    # ── Verdict ──
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)

    jacobian_has_structure = (sil_jac > 0.4 and ratio_j > 2.0)
    probe_has_structure = (result_probe["silhouette"] > 0.4 and
                           result_probe["inter_angle"] / (result_probe["intra_angle"] + 1e-8) > 2.0)

    print(f"  Probe-based (Step 2):     sil={result_probe['silhouette']:.3f}  "
          f"ratio={result_probe['inter_angle']/(result_probe['intra_angle']+1e-8):.2f}  "
          f"structure={probe_has_structure}")
    print(f"  Jacobian-based (Step I):  sil={sil_jac:.3f}  "
          f"ratio={ratio_j:.2f}  structure={jacobian_has_structure}")
    print(f"  mean step-to-step Jacobian angle: {mean_step_angle:.1f} deg")

    if jacobian_has_structure:
        print(f"\n  ==> Discrete dynamics ARE a genuine geometric structure.")
        print(f"  The Jacobian subspace clustering confirms: the hidden space")
        print(f"  develops distinct geometric regimes that match drift boundaries.")
    else:
        print(f"\n  ==> Discrete dynamics are LIKELY a probe artifact.")
        print(f"  The Jacobian subspace does not cluster cleanly by drift regime.")
        print(f"  The original finding (Step 2) was an axis-aligned artifact of")
        print(f"  the probe-based subspace extraction method.")

    out = {
        "jacobian_subspace": {
            "num_clusters": nk,
            "silhouette": sil_jac,
            "intra_angle": intra_j,
            "inter_angle": inter_j,
            "ratio": ratio_j,
            "structure_present": jacobian_has_structure,
            "labels": [int(l) for l in labels_jac],
            "window_starts": [int(ws) for ws in window_starts],
            "mutual_info_with_drift": float(mi),
            "per_cluster_drift": per_cluster_drift,
        },
        "probe_subspace_comparison": {
            "num_clusters": result_probe["num_clusters"],
            "silhouette": result_probe["silhouette"],
            "intra_angle": result_probe["intra_angle"],
            "inter_angle": result_probe["inter_angle"],
            "ratio": result_probe["inter_angle"] / (result_probe["intra_angle"] + 1e-8),
        },
        "step_jacobian_angle_mean_deg": mean_step_angle,
        "genuine_geometric_structure": jacobian_has_structure,
        "window": WINDOW,
        "stride": STRIDE,
        "top_k": TOP_K,
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/jacobian_subspace_dynamics.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  saved to results_final/jacobian_subspace_dynamics.json")


if __name__ == "__main__":
    main()
