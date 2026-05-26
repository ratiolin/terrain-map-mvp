"""Glassy Physics in Multi-Agent Gradient-Coupled Systems.

Exp1: RSB detection — 50 seeds at α=0.4, P(q) overlap distribution
Exp2: Representation effective rank vs criticality across dimensions
Exp3: Drift-induced metastability — sinusoidal α modulation
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def train_one_seed(alpha=0.4, hidden_dim=32, T=2500, N=1000, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    schedule = [(625, (0.1, 0.3)), (625, (1.0, 2.0)),
                (625, (0.1, 0.3)), (625, (1.0, 2.0))]
    env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
    pol_A = PolicyNetwork(hidden_dim=hidden_dim)
    pol_B = PolicyNetwork(hidden_dim=hidden_dim)
    params = list(pol_A.parameters()) + list(pol_B.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    enc_A = list(pol_A.backbone.parameters())
    enc_B = list(pol_B.backbone.parameters())
    env.reset()
    for _ in range(T):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        aA = float(pol_A(x)[0].item())
        aB = float(pol_B(x)[0].item())
        env.step(aA, aB)
        loss = torch.tensor(abs(float(env.state[0])), requires_grad=True)
        opt.zero_grad()
        loss.backward(retain_graph=True)
        for pA, pB in zip(enc_A, enc_B):
            if pA.grad is not None and pB.grad is not None:
                gA, gB = pA.grad.clone(), pB.grad.clone()
                mix = alpha * 0.5 * (gA + gB)
                pA.grad = (1.0 - alpha) * gA + mix
                pB.grad = (1.0 - alpha) * gB + mix
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

    H, A = [], []
    env.reset()
    for _ in range(N):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a_s_A, _, h = pol_A(x)
        env.step(float(a_s_A.item()), 0.0)
        H.append(h.numpy().flatten())
        A.append(float(a_s_A.item()))
    return np.array(H), np.array(A), pol_A


def effective_rank(S, threshold=0.95):
    total = (S ** 2).sum()
    cumulative = 0.0
    for i, s in enumerate(S):
        cumulative += s ** 2
        if cumulative / total >= threshold:
            return i + 1
    return len(S)


def main():
    print("=" * 60)
    print("  GLASSY PHYSICS IN GRADIENT-COUPLED SYSTEMS")
    print("=" * 60)
    all_results = {}

    # =============================================================
    # Exp1: RSB P(q) overlap distribution
    # =============================================================
    print("\n  Exp1: RSB — 50 seeds at α=0.4")
    n_seeds = 50
    rep_vectors = []
    H_all_seeds = []

    for seed in range(n_seeds):
        H, A, pol = train_one_seed(alpha=0.4, T=2500, N=800, seed=seed)
        _, S, Vt = np.linalg.svd(H, full_matrices=False)
        u = Vt[0]
        if u[0] < 0:
            u = -u
        rep_vectors.append(u)
        H_all_seeds.append(H)

    n = len(rep_vectors)
    q_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            q = float(np.dot(rep_vectors[i], rep_vectors[j]))
            q_matrix[i, j] = q
            q_matrix[j, i] = q

    q_vals = q_matrix[np.triu_indices(n, k=1)]

    hist, edges = np.histogram(q_vals, bins=30)
    peak_regions = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 5:
            peak_regions.append(round(float((edges[i] + edges[i+1])/2), 4))

    n_modes = len(peak_regions)
    classification = "unimodal" if n_modes <= 1 else ("bimodal" if n_modes == 2 else f"multimodal({n_modes})")
    print(f"    q range: [{q_vals.min():.3f}, {q_vals.max():.3f}]")
    print(f"    q mean: {q_vals.mean():.3f}, q std: {q_vals.std():.3f}")
    print(f"    modes: {n_modes} → {classification}")
    print(f"    peak locations: {peak_regions}")

    dist = 1.0 - q_matrix
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist)
    Z = linkage(condensed, method='ward')
    clusters_2 = fcluster(Z, t=2, criterion='maxclust')
    sizes = [int((clusters_2 == k).sum()) for k in [1, 2]]
    print(f"    hierarchical 2-cluster sizes: {sizes}")

    all_results["exp1_rsb"] = {
        "n_seeds": n_seeds,
        "q_mean": float(q_vals.mean()),
        "q_std": float(q_vals.std()),
        "q_min": float(q_vals.min()),
        "q_max": float(q_vals.max()),
        "n_modes": n_modes,
        "peak_locations": peak_regions,
        "classification": classification,
        "cluster_sizes": sizes,
        "histogram": {"counts": hist.tolist(), "edges": edges.tolist()},
    }

    # =============================================================
    # Exp2: Representation effective rank vs dimension
    # =============================================================
    print("\n  Exp2: Eff. rank vs criticality across dims")
    dims = [8, 16, 24, 32, 48, 64, 96]
    rank_data = []

    for dim in dims:
        H, A, pol = train_one_seed(alpha=0.0, hidden_dim=dim, T=2000, N=800, seed=42)
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        er = effective_rank(S, threshold=0.95)
        er_99 = effective_rank(S, threshold=0.99)
        rank_data.append({
            "dim": dim, "eff_rank_95": er, "eff_rank_99": er_99,
            "pc1_var_pct": round(float(S[0]**2/(S**2).sum()), 3),
            "dim_ratio_95": round(er/dim, 3),
            "dim_ratio_99": round(er_99/dim, 3),
        })
        print(f"    dim={dim:3d}: eff_rank(95%)={er:2d} ratio={er/dim:.2f}  "
              f"pc1_var={S[0]**2/(S**2).sum():.3f}")

    all_results["exp2_eff_rank"] = rank_data

    # =============================================================
    # Exp3: Drift-induced metastability
    # =============================================================
    print("\n  Exp3: Sinusoidal α drift — metastability detection")
    T_drift = 4000
    t_vals = np.arange(T_drift)
    alpha_drift = 0.4 + 0.15 * np.sin(2 * np.pi * t_vals / 500)
    seed = 123

    torch.manual_seed(seed)
    np.random.seed(seed)
    schedule = [(1000, (0.1, 0.3)), (1000, (1.0, 2.0)),
                (1000, (0.1, 0.3)), (1000, (1.0, 2.0))]
    env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
    pol_A = PolicyNetwork(hidden_dim=32)
    pol_B = PolicyNetwork(hidden_dim=32)
    params = list(pol_A.parameters()) + list(pol_B.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    enc_A = list(pol_A.backbone.parameters())
    enc_B = list(pol_B.backbone.parameters())

    env.reset()
    corr_hist, ent_hist = [], []
    for t in range(T_drift):
        alpha = float(alpha_drift[t])
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        aA = float(pol_A(x)[0].item())
        aB = float(pol_B(x)[0].item())
        env.step(aA, aB)
        loss = torch.tensor(abs(float(env.state[0])), requires_grad=True)
        opt.zero_grad()
        loss.backward(retain_graph=True)
        for pA, pB in zip(enc_A, enc_B):
            if pA.grad is not None and pB.grad is not None:
                gA, gB = pA.grad.clone(), pB.grad.clone()
                mix = alpha * 0.5 * (gA + gB)
                pA.grad = (1.0 - alpha) * gA + mix
                pB.grad = (1.0 - alpha) * gB + mix
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        if t >= 100 and t % 50 == 0:
            H_buf, aA_buf, aB_buf = [], [], []
            env_snap = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                             state_clip=5.0, force_scale=0.1,
                                             action_scale=0.1, action_mix=0.5)
            env_snap.reset()
            for _ in range(200):
                xs = torch.tensor(env_snap.state.copy(), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    as_A, _, h = pol_A(xs); as_B, _, _ = pol_B(xs)
                env_snap.step(float(as_A.item()), float(as_B.item()))
                H_buf.append(h.numpy().flatten())
                aA_buf.append(float(as_A.item())); aB_buf.append(float(as_B.item()))
            Hnp = np.array(H_buf)
            _, S, Vt = np.linalg.svd(Hnp, full_matrices=False)
            Z = Hnp @ Vt.T
            WA = pol_A.head_shape.weight.detach().numpy().flatten()
            WB = pol_B.head_shape.weight.detach().numpy().flatten()
            wA, wB = WA @ Vt.T, WB @ Vt.T
            contribs = np.array([wA[i] * wB[i] * float(np.var(Z[:, i])) for i in range(len(S))])
            p = np.abs(contribs) / (np.abs(contribs).sum() + 1e-12)
            ent = float(-np.sum(p * np.log(p + 1e-12)))
            corr = float(np.abs(np.corrcoef(np.array(aA_buf), np.array(aB_buf))[0, 1]))
            corr_hist.append(corr)
            ent_hist.append(ent)

    corr_h = np.array(corr_hist)
    ent_h = np.array(ent_hist)
    dcorr = np.abs(np.diff(corr_h))
    threshold = np.percentile(dcorr, 90)
    jumps = int((dcorr > threshold).sum())
    dwell_times = []
    current_dwell = 0
    for dc in dcorr:
        if dc < threshold:
            current_dwell += 1
        else:
            if current_dwell > 0:
                dwell_times.append(current_dwell)
            current_dwell = 0
    if current_dwell > 0:
        dwell_times.append(current_dwell)

    print(f"    samples: {len(corr_h)}")
    print(f"    corr∈[{corr_h.min():.3f},{corr_h.max():.3f}] "
          f"ent∈[{ent_h.min():.3f},{ent_h.max():.3f}]")
    print(f"    jumps (>p90): {jumps}")
    print(f"    mean dwell: {np.mean(dwell_times):.1f} (±{np.std(dwell_times):.1f})")
    metastable = jumps > 0 and np.mean(dwell_times) > 5
    print(f"    metastable: {'YES' if metastable else 'no'}")

    all_results["exp3_drift"] = {
        "T": T_drift, "samples": len(corr_h),
        "corr_min": float(corr_h.min()), "corr_max": float(corr_h.max()),
        "ent_min": float(ent_h.min()), "ent_max": float(ent_h.max()),
        "n_jumps": jumps, "mean_dwell": float(np.mean(dwell_times)),
        "std_dwell": float(np.std(dwell_times)),
        "metastable": bool(metastable),
    }

    out = Path("results_final/glassy_physics.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
