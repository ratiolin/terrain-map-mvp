#!/usr/bin/env python3
"""Analysis A/B/C — Panic, Continuous Drift, Multi-Agent Consistency."""

import json
import sys
import pickle
import random
import math
import time
import copy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS

from core_mvp_v3.env import drifting_double_well, DriftingDoubleWellSchedule
from core_mvp_v3.models import PolicyNetwork, PredictionNetwork
from core_mvp_v3.experiment import reset_seed, ExperimentConfig, train


def _load_trajectory():
    with open("results_final/trajectory_full.pkl", "rb") as f:
        return pickle.load(f)


def _load_policy(path="results_final/phase0_policy_net.pt"):
    net = PolicyNetwork(hidden_dim=32)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(sd)
    net.eval()
    return net


def _serialize_rng(rng_tuple):
    return [rng_tuple[0], rng_tuple[1].tolist(),
            int(rng_tuple[2]), int(rng_tuple[3]), float(rng_tuple[4])]


def _deserialize_rng(rng_tuple_ser):
    return (rng_tuple_ser[0], np.array(rng_tuple_ser[1], dtype=np.uint32),
            int(rng_tuple_ser[2]), int(rng_tuple_ser[3]), float(rng_tuple_ser[4]))


def _build_window_subspaces(hiddens, controllability, window, stride, top_k):
    hidden_dim = hiddens.shape[1]
    T = len(hiddens)
    windows = [(s, s + window) for s in range(0, T - window + 1, stride)]
    subs = []
    starts = []
    for si, ei in windows:
        H = hiddens[si:ei]
        c = controllability[si:ei]
        scaler = StandardScaler()
        Hs = scaler.fit_transform(H)
        probe = Ridge(alpha=1.0)
        probe.fit(Hs, c)
        coef = np.abs(probe.coef_)
        topk = np.argsort(coef)[-top_k:]
        basis = np.eye(hidden_dim)[topk]
        Q, _ = np.linalg.qr(basis.T)
        subs.append(Q)
        starts.append(si)
    return subs, starts, len(subs)


def _pairwise_principal_angle_mean(subspaces):
    n = len(subspaces)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ang = float(np.degrees(subspace_angles(subspaces[i], subspaces[j])).mean())
            D[i, j] = D[j, i] = ang
    return D


def _cluster_subspaces(subspaces, D, max_k=3, seed=42):
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


def _intra_inter(D, labels):
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


def _generate_trajectory(policy_net, env_factory, total_steps, seed=42, rng_seed=0):
    reset_seed(seed)
    env = env_factory()
    env.set_rng_seed(rng_seed)
    env.reset()
    traj = []
    controllability = 0.0
    for step in range(total_steps):
        st = env.state.copy()
        dr = env.current_drift
        st_t = torch.tensor(st, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            ash, aad, h = policy_net(st_t)
        av = float(ash.item())
        snap = env.save_state()
        ser = {
            "state": snap["state"].tolist(),
            "t": int(snap["t"]),
            "_segment_idx": int(snap["_segment_idx"]),
            "_segment_t": int(snap["_segment_t"]),
            "current_drift": float(snap["current_drift"]),
            "rng_state": _serialize_rng(snap["rng_state"]),
        }
        env.step(av)
        if step % 10 == 0:
            from new_experiment import compute_controllability
            ctrl, _, _ = compute_controllability(env, float(ash.item()), float(aad.item()), 7, 5)
            controllability = ctrl
        traj.append({
            "obs_t": st.tolist(),
            "action_t": av,
            "h_t": h.detach().cpu().numpy().squeeze(0).tolist(),
            "g_t": float(dr),
            "controllability": float(controllability),
            "env_state": ser,
        })
        if (step + 1) % 2000 == 0:
            print(f"    ... {step + 1}/{total_steps}")
    return traj


def _subspace_clusters_from_traj(traj, window=500, stride=250, top_k=5, seed=42):
    hiddens = np.array([t["h_t"] for t in traj])
    controllability = np.array([t["controllability"] for t in traj])
    subs, starts, nw = _build_window_subspaces(hiddens, controllability, window, stride, top_k)
    D = _pairwise_principal_angle_mean(subs)
    nk, labels, sil = _cluster_subspaces(subs, D, max_k=3, seed=seed)
    intra, inter, ni, nj = _intra_inter(D, labels)
    return {
        "subspaces": subs, "window_starts": starts, "n_windows": nw,
        "distance_matrix": D, "num_clusters": nk, "labels": labels,
        "silhouette": sil, "intra_angle": intra, "inter_angle": inter,
        "intra_n": ni, "inter_n": nj,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# A. PANIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_panic(seed=42):
    print("=" * 60)
    print("  A. PANIC vs SUBSPACE CLUSTER ANALYSIS")
    print("=" * 60)

    np.random.seed(seed)
    random.seed(seed)

    with open("results_final/subspace_clusters.json") as f:
        clusters = json.load(f)
    labels = np.array(clusters["labels"])
    window_starts = np.array(clusters["window_starts"])
    window = 500

    traj = _load_trajectory()
    controllability = np.array([t["controllability"] for t in traj])
    T = len(traj)

    # panic proxy: inverted normalized controllability
    raw_ctrl = controllability.copy()
    ctrl_max = raw_ctrl.max()
    ctrl_range = ctrl_max - raw_ctrl.min()
    if ctrl_range < 1e-8:
        panic_signal = np.ones(T)
    else:
        panic_signal = (ctrl_max - raw_ctrl) / (ctrl_range + 1e-8)

    # per-window mean panic
    window_panic = np.array([
        panic_signal[s:min(s + window, T)].mean()
        for s in window_starts
    ])

    # align: each window maps to its center time for CCF
    centers = window_starts + window // 2
    cluster_at_center = np.array(labels, dtype=int)
    panic_at_center = np.array([panic_signal[min(int(c), T - 1)] for c in centers])

    # --- 1. P(panic_high | cluster=k) ---
    panic_median = np.median(window_panic)
    panic_high = (window_panic > panic_median).astype(int)

    cluster_ids = sorted(set(labels))
    p_panic_given_cluster = {}
    for cid in cluster_ids:
        wmask = labels == cid
        p_panic_given_cluster[int(cid)] = float(panic_high[wmask].mean()) if wmask.any() else 0.0

    # --- 2. P(cluster=k | panic_high) ---
    p_cluster_given_panic = {}
    for cid in cluster_ids:
        mask = panic_high == 1
        p_cluster_given_panic[int(cid)] = float((labels[mask] == cid).mean()) if mask.any() else 0.0

    # --- 3. cross-correlation of cluster-2 indicator vs panic ---
    cluster2_indicator = (cluster_at_center == 2).astype(float)
    max_lag = 50
    lags = list(range(-max_lag, max_lag + 1))
    ccf = []
    for lag in lags:
        if lag < 0:
            x = cluster2_indicator[:lag]
            y = panic_at_center[-lag:]
        elif lag > 0:
            x = cluster2_indicator[lag:]
            y = panic_at_center[:-lag]
        else:
            x = cluster2_indicator
            y = panic_at_center
        if len(x) > 2:
            c = np.corrcoef(x, y)[0, 1]
            ccf.append(float(0.0 if np.isnan(c) else c))
        else:
            ccf.append(0.0)

    max_ccf = max(abs(v) for v in ccf) if ccf else 0.0
    max_ccf_lag = lags[np.argmax(np.abs(ccf))] if ccf else 0

    # --- significance: shuffle test ---
    n_shuffle = 1000
    null_ccf_max = []
    for _ in range(n_shuffle):
        shuf = panic_at_center.copy()
        np.random.shuffle(shuf)
        shuf_ccf = []
        for lag in lags:
            if lag < 0:
                x = cluster2_indicator[:lag]
                y = shuf[-lag:]
            elif lag > 0:
                x = cluster2_indicator[lag:]
                y = shuf[:-lag]
            else:
                x = cluster2_indicator
                y = shuf
            if len(x) > 2:
                c = np.corrcoef(x, y)[0, 1]
                shuf_ccf.append(float(0.0 if np.isnan(c) else c))
            else:
                shuf_ccf.append(0.0)
        null_ccf_max.append(max(abs(v) for v in shuf_ccf))
    null_ccf_max = np.array(null_ccf_max)
    z_score = (max_ccf - null_ccf_max.mean()) / (null_ccf_max.std() + 1e-8)
    p_value = float((null_ccf_max >= max_ccf).mean())

    cluster2_sync = (z_score > 2.0) or (p_value < 0.05)

    # per-cluster mean panic
    per_cluster_panic = {}
    for cid in cluster_ids:
        mask = labels == cid
        per_cluster_panic[int(cid)] = float(window_panic[mask].mean()) if mask.any() else 0.0

    print(f"  controllability range: [{ctrl_max:.4f}, {raw_ctrl.min():.4f}]")
    print(f"  per-cluster mean panic: {per_cluster_panic}")
    print(f"  P(panic_high|cluster): {p_panic_given_cluster}")
    print(f"  P(cluster|panic_high): {p_cluster_given_panic}")
    print(f"  max |CCF| = {max_ccf:.4f} at lag {max_ccf_lag}")
    print(f"  z_score = {z_score:.2f}  p_value = {p_value:.4f}")
    print(f"  cluster-2 synchronous boundary signal: {cluster2_sync}")

    result = {
        "controllability_range": [float(ctrl_max), float(raw_ctrl.min())],
        "per_cluster_mean_panic": per_cluster_panic,
        "P_panic_given_cluster": p_panic_given_cluster,
        "P_cluster_given_panic_high": p_cluster_given_panic,
        "max_ccf_abs": max_ccf,
        "max_ccf_lag": max_ccf_lag,
        "ccf_values": ccf,
        "lags": lags,
        "z_score": float(z_score),
        "p_value": float(p_value),
        "cluster2_synchronous": cluster2_sync,
        "n_shuffle": n_shuffle,
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/panic_alignment.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# B. CONTINUOUS DRIFT TEST
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousDriftEnv:
    def __init__(self, drift_func, noise=0.05, state_clip=5.0,
                 force_scale=0.1, action_scale=0.1):
        schedule = [(100000, (0.1, 0.3))]
        self._env = DriftingDoubleWellSchedule(
            schedule=schedule, noise=noise, state_clip=state_clip,
            force_scale=force_scale, action_scale=action_scale)
        self.drift_func = drift_func

    def reset(self):
        return self._env.reset()

    def step(self, action):
        self._env.current_drift = float(self.drift_func(self._env.t))
        return self._env.step(action)

    @property
    def state(self):
        return self._env.state

    @property
    def t(self):
        return self._env.t

    @property
    def current_drift(self):
        return self._env.current_drift

    def clone(self):
        c = ContinuousDriftEnv.__new__(ContinuousDriftEnv)
        c.drift_func = self.drift_func
        c._env = copy.deepcopy(self._env)
        return c

    def save_state(self):
        return self._env.save_state()

    def restore_state(self, saved):
        return self._env.restore_state(saved)

    def set_rng_seed(self, seed):
        self._env.set_rng_seed(seed)

    def get_rng_seed(self):
        return self._env.get_rng_seed()


def _continuous_drift_sine(t, period=2000):
    return 1.0 + 1.0 * math.sin(2.0 * math.pi * t / max(period, 1))


def analyze_continuous_drift(seed=42, total_steps=6000, window=500, stride=250, top_k=5):
    print("=" * 60)
    print("  B. CONTINUOUS DRIFT STRUCTURE TEST")
    print("=" * 60)

    reset_seed(seed)

    env_factory = lambda: ContinuousDriftEnv(_continuous_drift_sine)
    policy_net = _load_policy()

    print(f"  generating {total_steps}-step trajectory with continuous (sinusoidal) drift")
    traj = _generate_trajectory(policy_net, env_factory, total_steps, seed=seed)
    print(f"  trajectory: {len(traj)} steps")
    print(f"  drift range: [{min(t['g_t'] for t in traj):.3f}, {max(t['g_t'] for t in traj):.3f}]")

    print(f"\n  running subspace + clustering pipeline (w={window}, s={stride})")
    result = _subspace_clusters_from_traj(traj, window=window, stride=stride,
                                           top_k=top_k, seed=seed)

    nk = result["num_clusters"]
    sil = result["silhouette"]
    intra = result["intra_angle"]
    inter = result["inter_angle"]
    ratio = inter / (intra + 1e-8)

    print(f"  K = {nk}, silhouette = {sil:.3f}")
    print(f"  intra = {intra:.2f} deg, inter = {inter:.2f} deg, ratio = {ratio:.2f}")

    structure_preserved = (sil > 0.4 and ratio > 2.0)
    print(f"  discrete structure preserved: {structure_preserved}")

    # mutual_info(cluster_id vs drift_bin)
    centers = np.array(result["window_starts"]) + window // 2
    labels = np.array(result["labels"])
    drift_at_center = np.array([traj[min(c, len(traj) - 1)]["g_t"] for c in centers])
    n_bins = 10
    drift_bins = np.digitize(drift_at_center, np.linspace(0, 2, n_bins + 1))
    from sklearn.metrics import mutual_info_score
    mi = mutual_info_score(labels, drift_bins)
    print(f"  mutual_info(cluster, drift_bin): {mi:.4f}")

    # per-cluster drift stats
    cluster_drifts = {}
    for cid in sorted(set(labels)):
        mask = labels == cid
        cluster_drifts[int(cid)] = {
            "mean_drift": float(drift_at_center[mask].mean()),
            "std_drift": float(drift_at_center[mask].std()),
            "count": int(mask.sum()),
        }
        print(f"  cluster {cid}: drift mean={cluster_drifts[int(cid)]['mean_drift']:.3f} "
              f"± {cluster_drifts[int(cid)]['std_drift']:.3f} (n={cluster_drifts[int(cid)]['count']})")

    out = {
        "continuous_drift_type": "sinusoidal",
        "total_steps": total_steps,
        "num_clusters": nk,
        "silhouette": sil,
        "intra_angle": intra,
        "inter_angle": inter,
        "inter_intra_ratio": ratio,
        "structure_preserved": structure_preserved,
        "mutual_info": float(mi),
        "cluster_drift_stats": cluster_drifts,
        "labels": [int(l) for l in labels],
        "window_starts": result["window_starts"],
        "window": window,
        "stride": stride,
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/continuous_drift_test.json", "w") as f:
        json.dump(out, f, indent=2)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# C. MULTI-AGENT CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_multi_agent(seed=42, total_steps=8000, window=500, stride=250, top_k=5):
    print("=" * 60)
    print("  C. MULTI-AGENT SUBSPACE CONSISTENCY")
    print("=" * 60)

    reset_seed(seed)

    # Agent A (seed 42, already done) — load from trajectory_full.pkl
    print("  Agent A: loading existing trajectory + clusters...")
    traj_a = _load_trajectory()
    result_a = _subspace_clusters_from_traj(traj_a, window, stride, top_k, seed)
    subs_a = result_a["subspaces"]
    labels_a = np.array(result_a["labels"])
    nk_a = result_a["num_clusters"]

    # Build per-cluster subspace for agent A (union of dimension sets)
    hiddens_a = np.array([t["h_t"] for t in traj_a])
    controllability_a = np.array([t["controllability"] for t in traj_a])
    cluster_subs_a = {}
    for cid in sorted(set(labels_a)):
        idx = np.where(labels_a == cid)[0]
        H = np.concatenate([hiddens_a[result_a["window_starts"][i]:
                                       result_a["window_starts"][i] + window]
                            for i in idx])
        c = np.concatenate([controllability_a[result_a["window_starts"][i]:
                                               result_a["window_starts"][i] + window]
                            for i in idx])
        scaler = StandardScaler()
        Hs = scaler.fit_transform(H)
        probe = Ridge(alpha=1.0)
        probe.fit(Hs, c)
        coef = np.abs(probe.coef_)
        topk = np.argsort(coef)[-top_k:]
        basis = np.eye(32)[topk]
        Q, _ = np.linalg.qr(basis.T)
        cluster_subs_a[int(cid)] = Q
        r2 = probe.score(Hs, c)
        print(f"  Agent A cluster {cid}: R²={r2:.4f}, dims={topk.tolist()}")

    # Agent B (different seed)
    # Check if seed1 model exists, otherwise train
    seed1_path = Path("results_final/phase0_policy_net_seed1.pt")
    if not seed1_path.exists():
        print("  Training Agent B (seed=1)...")
        cfg = ExperimentConfig()
        cfg.num_episodes = 1
        cfg.episode_length = 8000
        cfg.seed = 1
        _, _, policy_b, _ = train(cfg, save_tag="seed1")
        print("  Agent B trained & saved.")
    else:
        print("  Agent B: loading existing model (seed1)")

    policy_b = _load_policy("results_final/phase0_policy_net_seed1.pt")

    print("  Agent B: generating trajectory...")
    env_factory = lambda: drifting_double_well(noise=0.05)
    traj_b = _generate_trajectory(policy_b, env_factory, total_steps, seed=1)

    result_b = _subspace_clusters_from_traj(traj_b, window, stride, top_k, seed)
    subs_b = result_b["subspaces"]
    labels_b = np.array(result_b["labels"])
    nk_b = result_b["num_clusters"]

    hiddens_b = np.array([t["h_t"] for t in traj_b])
    controllability_b = np.array([t["controllability"] for t in traj_b])
    cluster_subs_b = {}
    for cid in sorted(set(labels_b)):
        idx = np.where(labels_b == cid)[0]
        H = np.concatenate([hiddens_b[result_b["window_starts"][i]:
                                       result_b["window_starts"][i] + window]
                            for i in idx])
        c = np.concatenate([controllability_b[result_b["window_starts"][i]:
                                               result_b["window_starts"][i] + window]
                            for i in idx])
        scaler = StandardScaler()
        Hs = scaler.fit_transform(H)
        probe = Ridge(alpha=1.0)
        probe.fit(Hs, c)
        coef = np.abs(probe.coef_)
        topk = np.argsort(coef)[-top_k:]
        basis = np.eye(32)[topk]
        Q, _ = np.linalg.qr(basis.T)
        cluster_subs_b[int(cid)] = Q
        r2 = probe.score(Hs, c)
        print(f"  Agent B cluster {cid}: R²={r2:.4f}, dims={topk.tolist()}")

    # Cross-agent alignment
    print("\n  Cross-agent subspace alignment:")
    alignments = {}
    n_random = 1000
    hidden_dim = 32

    all_cids_a = sorted(cluster_subs_a.keys())
    all_cids_b = sorted(cluster_subs_b.keys())

    for ca in all_cids_a:
        for cb in all_cids_b:
            Qa = cluster_subs_a[ca]
            Qb = cluster_subs_b[cb]
            align = float(np.linalg.norm(Qa.T @ Qb, 'fro'))

            rand_scores = []
            for _ in range(n_random):
                M = np.random.randn(hidden_dim, Qa.shape[1])
                Qr, _ = np.linalg.qr(M)
                rand_scores.append(float(np.linalg.norm(Qa.T @ Qr, 'fro')))
            rand_scores = np.array(rand_scores)
            z = (align - rand_scores.mean()) / (rand_scores.std() + 1e-8)
            pct = float((rand_scores < align).mean()) * 100.0
            consistent = pct > 95.0

            key = f"A_cluster{ca}_vs_B_cluster{cb}"
            alignments[key] = {
                "alignment": align,
                "z_score": float(z),
                "percentile": pct,
                "consistent": consistent,
            }
            print(f"    {key}: align={align:.4f}, z={z:.2f}, pct={pct:.1f}% "
                  f"-> {'CONSISTENT' if consistent else 'not consistent'}")

    out = {
        "agent_A": {
            "num_clusters": nk_a,
            "labels": [int(l) for l in labels_a],
            "silhouette": result_a["silhouette"],
        },
        "agent_B": {
            "num_clusters": nk_b,
            "labels": [int(l) for l in labels_b],
            "silhouette": result_b["silhouette"],
        },
        "cross_agent_alignments": alignments,
        "n_random_baseline": n_random,
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/multi_agent_alignment.json", "w") as f:
        json.dump(out, f, indent=2)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 6.5. CROSS-AGENT PROBE (FUNCTIONAL CONSISTENCY)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_cross_agent_probe(seed=42, total_steps=8000, window=500, stride=250, top_k=5):
    print("=" * 60)
    print("  6.5. CROSS-AGENT PROBE (FUNCTIONAL CONSISTENCY)")
    print("=" * 60)

    reset_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Agent A (seed=42) — loaded from existing trajectory
    print("  Agent A: loading existing trajectory + clustering...")
    traj_a = _load_trajectory()
    result_a = _subspace_clusters_from_traj(traj_a, window, stride, top_k, seed)
    labels_a = np.array(result_a["labels"])
    window_starts_a = np.array(result_a["window_starts"])
    nk_a = result_a["num_clusters"]
    print(f"  Agent A: {nk_a} clusters, silhouette={result_a['silhouette']:.3f}")

    # Agent B (seed=1) — generate trajectory
    print("  Agent B: loading policy + generating trajectory...")
    policy_b = _load_policy("results_final/phase0_policy_net_seed1.pt")
    env_factory = lambda: drifting_double_well(noise=0.05)
    traj_b = _generate_trajectory(policy_b, env_factory, total_steps, seed=1)
    result_b = _subspace_clusters_from_traj(traj_b, window, stride, top_k, seed)
    labels_b = np.array(result_b["labels"])
    window_starts_b = np.array(result_b["window_starts"])
    nk_b = result_b["num_clusters"]
    print(f"  Agent B: {nk_b} clusters, silhouette={result_b['silhouette']:.3f}")

    hiddens_a = np.array([t["h_t"] for t in traj_a])
    ctrl_a = np.array([t["controllability"] for t in traj_a])
    hiddens_b = np.array([t["h_t"] for t in traj_b])
    ctrl_b = np.array([t["controllability"] for t in traj_b])

    # ── within-agent baseline: per-cluster R² ──
    print("\n  ── within-agent R² (A→A) ──")
    within_a = {}
    for cid in sorted(set(labels_a)):
        idx = np.where(labels_a == cid)[0]
        H = np.concatenate([hiddens_a[window_starts_a[i]:window_starts_a[i] + window]
                            for i in idx])
        c = np.concatenate([ctrl_a[window_starts_a[i]:window_starts_a[i] + window]
                            for i in idx])
        scaler = StandardScaler()
        Hs = scaler.fit_transform(H)
        probe = Ridge(alpha=1.0)
        probe.fit(Hs, c)
        r2 = probe.score(Hs, c)
        within_a[int(cid)] = float(r2)
        print(f"    cluster {cid}: within-A R²={r2:.4f}")

    # ── cross-agent probe: train on A, test on B ──
    print("\n  ── cross-agent probe (train on A, test on B) ──")
    cross_results = {}

    for cid in sorted(set(labels_a)):
        # Train probe on Agent A at this cluster's windows
        idx_a = np.where(labels_a == cid)[0]
        H_a = np.concatenate([hiddens_a[window_starts_a[i]:window_starts_a[i] + window]
                              for i in idx_a])
        c_a = np.concatenate([ctrl_a[window_starts_a[i]:window_starts_a[i] + window]
                              for i in idx_a])
        scaler_a = StandardScaler()
        H_a_scaled = scaler_a.fit_transform(H_a)
        probe_cross = Ridge(alpha=1.0)
        probe_cross.fit(H_a_scaled, c_a)
        r2_train = probe_cross.score(H_a_scaled, c_a)

        # Test on Agent B at THE SAME window positions
        H_b = np.concatenate([hiddens_b[window_starts_a[i]:window_starts_a[i] + window]
                              for i in idx_a])
        c_b = np.concatenate([ctrl_b[window_starts_a[i]:window_starts_a[i] + window]
                              for i in idx_a])
        H_b_scaled = scaler_a.transform(H_b)
        c_b_pred = probe_cross.predict(H_b_scaled)
        r2_cross = 1.0 - np.sum((c_b - c_b_pred) ** 2) / (np.sum((c_b - c_b.mean()) ** 2) + 1e-8)
        pearson = float(np.corrcoef(c_b, c_b_pred)[0, 1]) if len(c_b) > 1 else 0.0
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(c_b, c_b_pred)

        cross_results[int(cid)] = {
            "R2_train_on_A": float(r2_train),
            "R2_cross_on_B": float(r2_cross),
            "pearson_r": float(0.0 if np.isnan(pearson) else pearson),
            "spearman_r": float(0.0 if np.isnan(spearman) else spearman),
            "n_train": len(c_a),
            "n_test": len(c_b),
        }
        print(f"    cluster {cid}: R²(A→A)={r2_train:.4f}  "
              f"R²(A→B)={r2_cross:.4f}  r={pearson:.4f}  ρ={spearman:.4f}")

    # ── baseline 1: shuffle labels (permute A's cluster assignments) ──
    print("\n  ── baseline 1: shuffle cluster labels ──")
    n_shuffle = 100
    shuffle_scores = {int(cid): {"R2": [], "pearson": []} for cid in sorted(set(labels_a))}
    for _ in range(n_shuffle):
        shuf_labels = labels_a.copy()
        np.random.shuffle(shuf_labels)
        for cid in sorted(set(shuf_labels)):
            idx_a = np.where(shuf_labels == cid)[0]
            if len(idx_a) == 0:
                continue
            H_a_s = np.concatenate([hiddens_a[window_starts_a[i]:window_starts_a[i] + window]
                                    for i in idx_a])
            c_a_s = np.concatenate([ctrl_a[window_starts_a[i]:window_starts_a[i] + window]
                                    for i in idx_a])
            try:
                scaler_s = StandardScaler()
                H_a_s_sc = scaler_s.fit_transform(H_a_s)
                probe_s = Ridge(alpha=1.0)
                probe_s.fit(H_a_s_sc, c_a_s)
                H_b_s = np.concatenate([hiddens_b[window_starts_a[i]:window_starts_a[i] + window]
                                        for i in idx_a])
                c_b_s = np.concatenate([ctrl_b[window_starts_a[i]:window_starts_a[i] + window]
                                        for i in idx_a])
                H_b_s_sc = scaler_s.transform(H_b_s)
                c_b_s_pred = probe_s.predict(H_b_s_sc)
                r2_shuf = 1.0 - np.sum((c_b_s - c_b_s_pred) ** 2) / (np.sum((c_b_s - c_b_s.mean()) ** 2) + 1e-8)
                pr = float(np.corrcoef(c_b_s, c_b_s_pred)[0, 1]) if len(c_b_s) > 1 else 0.0
                shuffle_scores[int(cid)]["R2"].append(r2_shuf)
                shuffle_scores[int(cid)]["pearson"].append(0.0 if np.isnan(pr) else pr)
            except Exception:
                pass

    shuffle_stats = {}
    for cid in sorted(set(labels_a)):
        vals_r2 = np.array(shuffle_scores[int(cid)]["R2"])
        vals_pr = np.array(shuffle_scores[int(cid)]["pearson"])
        if len(vals_r2) > 0:
            shuffle_stats[int(cid)] = {
                "mean_R2": float(vals_r2.mean()),
                "std_R2": float(vals_r2.std()),
                "mean_pearson": float(vals_pr.mean()),
                "std_pearson": float(vals_pr.std()),
            }
            z_r2 = (cross_results[int(cid)]["R2_cross_on_B"] - vals_r2.mean()) / (vals_r2.std() + 1e-8)
            p_r2 = float((vals_r2 >= cross_results[int(cid)]["R2_cross_on_B"]).mean())
            shuffle_stats[int(cid)]["z_R2"] = float(z_r2)
            shuffle_stats[int(cid)]["p_R2"] = float(p_r2)
            print(f"    cluster {cid}: R² z={z_r2:.2f} p={p_r2:.4f}")

    # ── baseline 2: random subspace ──
    print("\n  ── baseline 2: random subspace probes ──")
    n_rand = 100
    rand_scores = {int(cid): {"R2": [], "pearson": []} for cid in sorted(set(labels_a))}
    full_hidden_dim = 32
    for cid in sorted(set(labels_a)):
        idx_a = np.where(labels_a == cid)[0]
        if len(idx_a) == 0:
            continue
        H_b_full = np.concatenate([hiddens_b[window_starts_a[i]:window_starts_a[i] + window]
                                   for i in idx_a])
        c_b_full = np.concatenate([ctrl_b[window_starts_a[i]:window_starts_a[i] + window]
                                   for i in idx_a])
        H_a_full = np.concatenate([hiddens_a[window_starts_a[i]:window_starts_a[i] + window]
                                   for i in idx_a])
        c_a_full = np.concatenate([ctrl_a[window_starts_a[i]:window_starts_a[i] + window]
                                   for i in idx_a])
        for _ in range(n_rand):
            rand_dims = np.random.choice(full_hidden_dim, top_k, replace=False)
            H_a_rand = H_a_full[:, rand_dims]
            H_b_rand = H_b_full[:, rand_dims]
            scaler_r = StandardScaler()
            try:
                H_a_r_sc = scaler_r.fit_transform(H_a_rand)
                probe_r = Ridge(alpha=1.0)
                probe_r.fit(H_a_r_sc, c_a_full)
                H_b_r_sc = scaler_r.transform(H_b_rand)
                c_b_r_pred = probe_r.predict(H_b_r_sc)
                r2_r = 1.0 - np.sum((c_b_full - c_b_r_pred) ** 2) / (np.sum((c_b_full - c_b_full.mean()) ** 2) + 1e-8)
                pr_r = float(np.corrcoef(c_b_full, c_b_r_pred)[0, 1]) if len(c_b_full) > 1 else 0.0
                rand_scores[int(cid)]["R2"].append(r2_r)
                rand_scores[int(cid)]["pearson"].append(0.0 if np.isnan(pr_r) else pr_r)
            except Exception:
                pass

    rand_stats = {}
    for cid in sorted(set(labels_a)):
        vals_r2_r = np.array(rand_scores[int(cid)]["R2"])
        vals_pr_r = np.array(rand_scores[int(cid)]["pearson"])
        if len(vals_r2_r) > 0:
            rand_stats[int(cid)] = {
                "mean_R2": float(vals_r2_r.mean()),
                "std_R2": float(vals_r2_r.std()),
                "mean_pearson": float(vals_pr_r.mean()),
                "std_pearson": float(vals_pr_r.std()),
            }
            z_r2_r = (cross_results[int(cid)]["R2_cross_on_B"] - vals_r2_r.mean()) / (vals_r2_r.std() + 1e-8)
            p_r2_r = float((vals_r2_r >= cross_results[int(cid)]["R2_cross_on_B"]).mean())
            rand_stats[int(cid)]["z_R2"] = float(z_r2_r)
            rand_stats[int(cid)]["p_R2"] = float(p_r2_r)
            print(f"    cluster {cid}: R² z={z_r2_r:.2f} p={p_r2_r:.4f} (vs random {top_k}D)")

    # ── verdict ──
    functional_consistent = False
    for cid in sorted(set(labels_a)):
        cs = cross_results[int(cid)]
        ss = shuffle_stats.get(int(cid), {})
        rs = rand_stats.get(int(cid), {})
        above_shuffle = ss.get("z_R2", -99) > 2.0 or ss.get("p_R2", 1.0) < 0.05
        above_random = rs.get("z_R2", -99) > 2.0 or rs.get("p_R2", 1.0) < 0.05
        if above_shuffle and cs["R2_cross_on_B"] > 0:
            functional_consistent = True

    print(f"\n  functional consistency (cross-agent probe > baselines): {functional_consistent}")

    out = {
        "within_agent_A_R2": {str(k): v for k, v in within_a.items()},
        "cross_agent_probe": {str(k): v for k, v in cross_results.items()},
        "baseline_shuffle": {str(k): v for k, v in shuffle_stats.items()},
        "baseline_random_subspace": {str(k): v for k, v in rand_stats.items()},
        "functional_consistency": functional_consistent,
        "n_shuffle": n_shuffle,
        "n_random_subspaces_per_cluster": n_rand,
    }

    Path("results_final").mkdir(exist_ok=True)
    with open("results_final/cross_agent_probe.json", "w") as f:
        json.dump(out, f, indent=2)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    seed = 42

    print("╔" + "═" * 58 + "╗")
    print("║  ANALYSIS A/B/C — PANIC, CONTINUOUS DRIFT, MULTI-AGENT")
    print("╚" + "═" * 58 + "╝")
    print()

    # A
    panic_result = analyze_panic(seed=seed)
    print()

    # B
    continuous_result = analyze_continuous_drift(seed=seed)
    print()

    # C
    multi_result = analyze_multi_agent(seed=seed)
    print()

    # 6.5
    cross_probe_result = analyze_cross_agent_probe(seed=seed)
    print()

    print("=" * 60)
    print("  ALL ANALYSIS COMPLETE")
    print("=" * 60)
    print("  Outputs:")
    print("    results_final/panic_alignment.json")
    print("    results_final/continuous_drift_test.json")
    print("    results_final/multi_agent_alignment.json")
    print("    results_final/cross_agent_probe.json")


if __name__ == "__main__":
    main()
