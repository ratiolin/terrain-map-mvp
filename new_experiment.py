#!/usr/bin/env python3
"""New Experiment: Controllability Subspace Analysis Pipeline.

Step 0: Determinism Check
Step 1: Generate Trajectories with Full Fields
Step 2: Subspace Time Stability (Direction 0)
Step 3: Jacobian Alignment (Direction 1)
Step 4: Decision Node
"""

import json
import sys
import pickle
import random
import math
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles

from core_mvp_v3.env import drifting_double_well
from core_mvp_v3.models import PolicyNetwork


# ─── seed fix ────────────────────────────────────────────────────────────────

def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─── controllability (inline to avoid heavy train() import) ─────────────────

def _rollout(env, action_val, k):
    total_risk = 0.0
    for _ in range(k):
        state = env.step(action_val)
        total_risk += abs(state[0])
    return total_risk / k


def compute_controllability(env, action_shape_val, action_adapt_val, k, n_rollouts=5):
    shape_risks = []
    adapt_risks = []
    for _ in range(n_rollouts):
        env_s = env.clone()
        env_a = env.clone()
        shared_seed = env_s.get_rng_seed()
        env_s.set_rng_seed(shared_seed)
        env_a.set_rng_seed(shared_seed)
        shape_risks.append(_rollout(env_s, action_shape_val, k))
        adapt_risks.append(_rollout(env_a, action_adapt_val, k))
    med_shape = float(np.median(shape_risks))
    med_adapt = float(np.median(adapt_risks))
    controllability = max(0.0, med_adapt - med_shape)
    return controllability, med_shape, med_adapt


def _serialize_rng(rng_tuple):
    return [
        rng_tuple[0],
        rng_tuple[1].tolist(),
        int(rng_tuple[2]),
        int(rng_tuple[3]),
        float(rng_tuple[4]),
    ]


def _deserialize_rng(rng_tuple_ser):
    return (
        rng_tuple_ser[0],
        np.array(rng_tuple_ser[1], dtype=np.uint32),
        int(rng_tuple_ser[2]),
        int(rng_tuple_ser[3]),
        float(rng_tuple_ser[4]),
    )


# ─── STEP 0 ──────────────────────────────────────────────────────────────────

def step0_determinism_check(seed=42, n_steps=200):
    print("=" * 60)
    print("  STEP 0: ENVIRONMENT DETERMINISM CHECK")
    print("=" * 60)

    reset_seed(seed)

    env1 = drifting_double_well(noise=0.05)
    env1.set_rng_seed(seed)
    s0 = env1.reset()

    env2 = env1.clone()

    assert np.array_equal(env1.state, env2.state), "Initial state mismatch"
    assert env1.t == env2.t, "Initial time mismatch"
    assert env1.current_drift == env2.current_drift, "Initial drift mismatch"

    actions = [float(np.random.uniform(-1, 1)) for _ in range(n_steps)]

    match = True
    steps_checked = 0

    for a in actions:
        obs1 = env1.step(a)
        obs2 = env2.step(a)

        if not np.array_equal(obs1, obs2):
            match = False
            print(f"  MISMATCH at step {steps_checked}: {obs1} != {obs2}")
            break
        steps_checked += 1

    if match:
        assert (obs1 == obs2).all(), f"Final step mismatch: {obs1} vs {obs2}"

    result = {"match": match, "steps_checked": steps_checked}

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/determinism_check.json", "w") as f:
        json.dump(result, f)

    status = "PASS" if match else "FAIL"
    print(f"  [{status}] match={match}  steps_checked={steps_checked}")
    return match


# ─── STEP 1 ──────────────────────────────────────────────────────────────────

def step1_generate_trajectory(seed=42, total_steps=8000):
    print("=" * 60)
    print("  STEP 1: GENERATE FULL TRAJECTORY")
    print("=" * 60)

    reset_seed(seed)

    policy_net = PolicyNetwork(hidden_dim=32)
    state_dict = torch.load("results_final/phase0_policy_net.pt",
                            map_location="cpu", weights_only=True)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    env = drifting_double_well(noise=0.05)
    env.set_rng_seed(seed)
    s0 = env.reset()

    trajectory = []
    controllability = 0.0

    T_CTRL = 10
    K_ROLLOUT = 7
    N_ROLLOUTS = 5

    t_start = time.time()
    for step in range(total_steps):
        state_t = env.state.copy()
        drift_t = env.current_drift

        state_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_shape, action_adapt, h = policy_net(state_tensor)

        env_state_snapshot = env.save_state()
        env_state_serializable = {
            "state": env_state_snapshot["state"].tolist(),
            "t": int(env_state_snapshot["t"]),
            "_segment_idx": int(env_state_snapshot["_segment_idx"]),
            "_segment_t": int(env_state_snapshot["_segment_t"]),
            "current_drift": float(env_state_snapshot["current_drift"]),
            "rng_state": _serialize_rng(env_state_snapshot["rng_state"]),
        }

        action_val = float(action_shape.item())

        state_next = env.step(action_val)

        if step % T_CTRL == 0:
            ctrl, _, _ = compute_controllability(
                env,
                float(action_shape.item()),
                float(action_adapt.item()),
                K_ROLLOUT,
                N_ROLLOUTS,
            )
            controllability = ctrl

        trajectory.append({
            "obs_t": state_t.tolist(),
            "action_t": float(action_val),
            "h_t": h.detach().cpu().numpy().squeeze(0).tolist(),
            "g_t": float(drift_t),
            "controllability": float(controllability),
            "env_state": env_state_serializable,
        })

        if (step + 1) % 1000 == 0:
            print(f"  ... {step + 1}/{total_steps} steps "
                  f"({time.time() - t_start:.0f}s)")

    elapsed = time.time() - t_start
    print(f"  trajectory: {len(trajectory)} steps in {elapsed:.1f}s")

    n_low = sum(1 for t in trajectory if t["g_t"] < 0.5)
    n_high = sum(1 for t in trajectory if t["g_t"] >= 0.5)
    print(f"  low drift steps: {n_low}  (target: >= 2000)")
    print(f"  high drift steps: {n_high}  (target: >= 2000)")

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/trajectory_full.pkl", "wb") as f:
        pickle.dump(trajectory, f)

    traj_json = []
    for t in trajectory:
        traj_json.append({
            "obs_t": t["obs_t"],
            "action_t": t["action_t"],
            "h_t": t["h_t"],
            "g_t": t["g_t"],
            "controllability": t["controllability"],
            "env_state": t["env_state"],
        })
    with open("results_final/trajectory_full.json", "w") as f:
        json.dump(traj_json, f)

    return trajectory, policy_net


# ─── shared subspace builder ──────────────────────────────────────────────────

def _build_window_subspaces(hiddens, controllability, window, stride, top_k):
    hidden_dim = hiddens.shape[1]
    T = len(hiddens)

    windows = []
    for start in range(0, T - window + 1, stride):
        windows.append((start, start + window))

    subspaces = []
    window_starts = []
    for s_i, e_i in windows:
        H = hiddens[s_i:e_i]
        c = controllability[s_i:e_i]

        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H)
        probe = Ridge(alpha=1.0)
        probe.fit(H_scaled, c)
        coef = np.abs(probe.coef_)
        topk_idx = np.argsort(coef)[-top_k:]
        basis = np.eye(hidden_dim)[topk_idx]
        Q, _ = np.linalg.qr(basis.T)
        subspaces.append(Q)
        window_starts.append(s_i)

    return subspaces, window_starts, len(windows)


def _pairwise_principal_angle_mean(subspaces):
    n = len(subspaces)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            angles_rad = subspace_angles(subspaces[i], subspaces[j])
            mean_ang = float(np.degrees(angles_rad).mean())
            D[i, j] = mean_ang
            D[j, i] = mean_ang
    return D


# ─── STEP 2 ──────────────────────────────────────────────────────────────────

def step2_subspace_dynamics(trajectory, window=500, stride=250, top_k=5):
    print("=" * 60)
    print("  STEP 2: SUBSPACE TIME STABILITY")
    print("=" * 60)

    hiddens = np.array([t["h_t"] for t in trajectory])
    controllability = np.array([t["controllability"] for t in trajectory])

    subspaces, window_starts, n_windows = _build_window_subspaces(
        hiddens, controllability, window, stride, top_k)
    print(f"  windows: {n_windows}  (window={window}, stride={stride})")

    all_pair_angles = []
    for i in range(n_windows - 1):
        angles_rad = subspace_angles(subspaces[i], subspaces[i + 1])
        angles_deg = np.degrees(angles_rad)
        all_pair_angles.extend(angles_deg.tolist())

    if all_pair_angles:
        mean_angle = float(np.mean(all_pair_angles))
        max_angle = float(np.max(all_pair_angles))
    else:
        mean_angle = 0.0
        max_angle = 0.0

    print(f"  mean_angle: {mean_angle:.2f} deg")
    print(f"  max_angle:  {max_angle:.2f} deg")
    print(f"  n_angle_pairs: {len(all_pair_angles)}")

    result = {
        "angles": all_pair_angles,
        "mean": mean_angle,
        "max": max_angle,
        "window": window,
        "stride": stride,
        "top_k": top_k,
        "n_windows": n_windows,
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/subspace_dynamics.json", "w") as f:
        json.dump(result, f)

    classification = ("static (<15°)" if mean_angle < 15 else
                      "dynamic (>30°)" if mean_angle > 30 else
                      "transitional (15°-30°)")
    print(f"  classification: {classification}")

    return result


# ─── STEP 2.5 ────────────────────────────────────────────────────────────────

def step2_5_subspace_clustering(trajectory, subspace_result, window=500,
                                 stride=250, top_k=5, max_k=3, seed=42):
    print("=" * 60)
    print("  STEP 2.5: SUBSPACE CLUSTERING")
    print("=" * 60)

    from sklearn.cluster import KMeans, SpectralClustering

    hiddens = np.array([t["h_t"] for t in trajectory])
    controllability = np.array([t["controllability"] for t in trajectory])

    subspaces, window_starts, n_windows = _build_window_subspaces(
        hiddens, controllability, window, stride, top_k)
    print(f"  windows: {n_windows}")

    D = _pairwise_principal_angle_mean(subspaces)
    print(f"  distance matrix: {D.shape}, mean={D[D>0].mean():.2f} deg")

    # Choose K by silhouette on distance matrix (via MDS embedding)
    from sklearn.manifold import MDS
    best_k = 2
    best_score = -1.0
    for k in range(2, max_k + 1):
        try:
            labels = SpectralClustering(
                n_clusters=k, affinity='precomputed', random_state=seed,
                assign_labels='kmeans',
            ).fit_predict(np.exp(-D / (D[D > 0].mean() + 1e-8)))
            # Compute silhouette on the embedded points
            embedding = MDS(n_components=min(5, n_windows - 1), dissimilarity='precomputed',
                           random_state=seed, normalized_stress='auto')
            pts = embedding.fit_transform(D)
            from sklearn.metrics import silhouette_score
            if len(set(labels)) > 1:
                sc = silhouette_score(pts, labels)
                if sc > best_score:
                    best_score = sc
                    best_k = k
        except Exception:
            continue

    print(f"  best K = {best_k}  (silhouette={best_score:.3f})")

    labels = SpectralClustering(
        n_clusters=best_k, affinity='precomputed', random_state=seed,
        assign_labels='kmeans',
    ).fit_predict(np.exp(-D / (D[D > 0].mean() + 1e-8)))

    cluster_ids = sorted(set(labels.tolist()))

    # Compute intra- and inter-cluster angles
    intra_angles = []
    inter_angles = []

    for cid in cluster_ids:
        members = [i for i, lbl in enumerate(labels) if lbl == cid]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                intra_angles.append(float(D[i, j]))

    for c1 in cluster_ids:
        for c2 in cluster_ids:
            if c1 >= c2:
                continue
            mem1 = [i for i, lbl in enumerate(labels) if lbl == c1]
            mem2 = [i for i, lbl in enumerate(labels) if lbl == c2]
            for i in mem1:
                for j in mem2:
                    inter_angles.append(float(D[i, j]))

    intra_mean = float(np.mean(intra_angles)) if intra_angles else 0.0
    inter_mean = float(np.mean(inter_angles)) if inter_angles else 0.0

    print(f"  intra-cluster mean angle: {intra_mean:.2f} deg  (n={len(intra_angles)})")
    print(f"  inter-cluster mean angle: {inter_mean:.2f} deg  (n={len(inter_angles)})")
    print(f"  ratio inter/intra: {inter_mean / (intra_mean + 1e-8):.2f}")

    result = {
        "num_clusters": best_k,
        "intra_angle": intra_mean,
        "inter_angle": inter_mean,
        "labels": [int(l) for l in labels],
        "window_starts": window_starts,
        "distance_matrix_mean": float(D[D > 0].mean()),
        "silhouette": float(best_score),
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/subspace_clusters.json", "w") as f:
        json.dump(result, f)

    return result


# ─── STEP 3 ──────────────────────────────────────────────────────────────────

def _compute_jacobian_at_step(policy_net, env, saved_state, action_val, epsilon):
    env.restore_state(saved_state)
    a_plus = action_val + epsilon
    s_plus = env.step(a_plus)
    state_tensor_plus = torch.tensor(s_plus, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        h_plus = policy_net.backbone(state_tensor_plus).squeeze(0).numpy()

    env.restore_state(saved_state)
    a_minus = action_val - epsilon
    s_minus = env.step(a_minus)
    state_tensor_minus = torch.tensor(s_minus, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        h_minus = policy_net.backbone(state_tensor_minus).squeeze(0).numpy()

    J = (h_plus - h_minus) / (2.0 * epsilon)
    return J


def step3_jacobian_alignment(trajectory, policy_net, top_k=5, epsilon=1e-4,
                              n_random=1000, seed=42):
    print("=" * 60)
    print("  STEP 3: JACOBIAN ALIGNMENT")
    print("=" * 60)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    hiddens = np.array([t["h_t"] for t in trajectory])
    controllability_arr = np.array([t["controllability"] for t in trajectory])
    drift_arr = np.array([t["g_t"] for t in trajectory])
    hidden_dim = hiddens.shape[1]
    T = len(trajectory)

    idx_low = np.where(drift_arr < 0.5)[0]
    idx_high = np.where(drift_arr >= 0.5)[0]
    print(f"  low drift steps: {len(idx_low)}")
    print(f"  high drift steps: {len(idx_high)}")

    # Fit probes for each drift group, build subspaces
    subspaces = {}
    for name, idx in [("low_drift", idx_low), ("high_drift", idx_high)]:
        H = hiddens[idx]
        c = controllability_arr[idx]

        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H)
        probe = Ridge(alpha=1.0)
        probe.fit(H_scaled, c)
        coef = np.abs(probe.coef_)
        topk_idx = np.argsort(coef)[-top_k:]
        basis = np.eye(hidden_dim)[topk_idx]
        Q, _ = np.linalg.qr(basis.T)
        subspaces[name] = Q

        r2 = probe.score(H_scaled, c)
        print(f"  {name}: R² = {r2:.4f}, top-{top_k} dims: {topk_idx.tolist()}")

    # Create a temporary env for Jacobian computation (preserve original env state)
    jac_env = drifting_double_well(noise=0.05)

    # Compute Jacobians at selected steps (subsample for performance)
    subsample = 10
    jac_steps = list(range(0, T, subsample))
    print(f"  computing Jacobians at {len(jac_steps)} steps "
          f"(every {subsample} steps, epsilon={epsilon})")

    jacobians = {}  # step index -> J vector
    jac_drift = []  # drift at each Jacobian step
    jac_hidden = []  # hidden state at each Jacobian step

    t_start = time.time()
    for s in jac_steps:
        t = trajectory[s]
        saved = {
            "state": np.array(t["env_state"]["state"], dtype=np.float32),
            "t": t["env_state"]["t"],
            "_segment_idx": t["env_state"]["_segment_idx"],
            "_segment_t": t["env_state"]["_segment_t"],
            "current_drift": t["env_state"]["current_drift"],
            "rng_state": _deserialize_rng(t["env_state"]["rng_state"]),
        }
        J = _compute_jacobian_at_step(
            policy_net, jac_env, saved, t["action_t"], epsilon)
        jacobians[s] = J
        jac_drift.append(t["g_t"])
        jac_hidden.append(t["h_t"])

    elapsed = time.time() - t_start
    print(f"  Jacobian computation done in {elapsed:.1f}s")

    # Group Jacobians by drift
    J_low = np.array([jacobians[s] for s in jac_steps
                      if trajectory[s]["g_t"] < 0.5])
    J_high = np.array([jacobians[s] for s in jac_steps
                       if trajectory[s]["g_t"] >= 0.5])

    print(f"  J_low shape: {J_low.shape}  J_high shape: {J_high.shape}")

    # Mean Jacobian direction (average the J vectors within each group)
    J_low_mean = J_low.mean(axis=0)    # (hidden_dim,)
    J_high_mean = J_high.mean(axis=0)  # (hidden_dim,)

    # Normalize to unit vectors (the top eigenvector of J J^T for rank-1 is J/|J|)
    J_low_dir = J_low_mean / (np.linalg.norm(J_low_mean) + 1e-8)
    J_high_dir = J_high_mean / (np.linalg.norm(J_high_mean) + 1e-8)

    # Alignment computation
    def alignment_score(Q, J_dir):
        return float(np.linalg.norm(Q.T @ J_dir.reshape(-1, 1), 'fro'))

    # Random baseline
    def random_baseline(J_dir, k, n):
        scores = []
        for _ in range(n):
            M = np.random.randn(hidden_dim, k)
            Q_rand, _ = np.linalg.qr(M)
            scores.append(alignment_score(Q_rand, J_dir))
        return np.array(scores)

    results = {}
    print("\n  --- alignment scores ---")

    for name, Q, J_dir, J_arr in [
        ("high_drift", subspaces["high_drift"], J_high_dir, J_high),
        ("low_drift", subspaces["low_drift"], J_low_dir, J_low),
    ]:
        align = alignment_score(Q, J_dir)
        rand_scores = random_baseline(J_dir, top_k, n_random)
        mean_rand = float(rand_scores.mean())
        std_rand = float(rand_scores.std() + 1e-8)
        z_score = (align - mean_rand) / std_rand
        percentile = float((rand_scores < align).mean()) * 100.0

        significant = z_score > 2.0 or percentile > 95.0

        print(f"  {name}:")
        print(f"    alignment:  {align:.4f}")
        print(f"    z_score:    {z_score:.2f}")
        print(f"    percentile: {percentile:.1f}%")
        print(f"    significant: {significant}")

        results[name] = {
            "alignment": align,
            "z_score": z_score,
            "percentile": percentile,
            "significant": significant,
            "mean_random": mean_rand,
            "std_random": std_rand,
            "n_random": n_random,
            "top_k": top_k,
            "epsilon": epsilon,
        }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/jacobian_alignment.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─── STEP 4 ──────────────────────────────────────────────────────────────────

def step4_decision(subspace_result, jacobian_result):
    print("=" * 60)
    print("  STEP 4: DECISION NODE")
    print("=" * 60)

    mean_angle = subspace_result["mean"]
    max_angle = subspace_result["max"]

    # Use the most significant alignment across both drift groups
    align_significant = False
    for key in jacobian_result:
        if jacobian_result[key].get("significant", False):
            align_significant = True
            break

    # Determine alignment percentile threshold
    max_percentile = max(
        jacobian_result[key].get("percentile", 0) for key in jacobian_result
    )
    alignment_above_90th = max_percentile > 90.0

    if mean_angle < 15.0 and align_significant:
        conclusion = "STATIC_ALIGNED"
    elif mean_angle > 30.0 and align_significant:
        conclusion = "DYNAMIC_ALIGNED"
    elif not alignment_above_90th:
        conclusion = "NOT_ALIGNED"
    else:
        conclusion = "UNCERTAIN"

    decision = {
        "case": conclusion,
        "inputs": {
            "mean_angle": mean_angle,
            "max_angle": max_angle,
            "max_percentile": max_percentile,
            "align_significant": align_significant,
        },
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    print(f"  mean_angle: {mean_angle:.2f} deg")
    print(f"  max_angle:  {max_angle:.2f} deg")
    print(f"  max_percentile: {max_percentile:.1f}%")
    print(f"  align_significant: {align_significant}")
    print(f"  CONCLUSION: {conclusion}")

    return decision


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    seed = 42

    # Step 0: Determinism check
    match = step0_determinism_check(seed=seed, n_steps=200)
    if not match:
        print("\n  FATAL: Environment is NOT deterministic. Exiting.")
        return

    print()

    # Step 1: Generate trajectory
    trajectory, policy_net = step1_generate_trajectory(seed=seed)

    print()

    # Step 2: Subspace time stability
    subspace_result = step2_subspace_dynamics(trajectory, window=500, stride=250,
                                               top_k=5)

    print()

    # Step 2.5: Subspace clustering
    cluster_result = step2_5_subspace_clustering(trajectory, subspace_result,
                                                  window=500, stride=250, top_k=5,
                                                  max_k=3, seed=seed)

    print()

    # Step 3: Jacobian alignment
    jacobian_result = step3_jacobian_alignment(trajectory, policy_net, top_k=5,
                                                epsilon=1e-4, n_random=1000,
                                                seed=seed)

    print()

    # Step 4: Decision
    decision = step4_decision(subspace_result, jacobian_result)

    print()
    print("=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Outputs:")
    print(f"    results_final/determinism_check.json")
    print(f"    results_final/trajectory_full.pkl")
    print(f"    results_final/subspace_dynamics.json")
    print(f"    results_final/subspace_clusters.json")
    print(f"    results_final/jacobian_alignment.json")
    print(f"    results_final/decision.json")


if __name__ == "__main__":
    main()
