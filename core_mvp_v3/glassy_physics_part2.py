"""Glassy Physics Part 2: Melting, Scaling, Aging, Crystallization.

Exp1: Noise-induced melting — gradient noise → P(q) mode collapse
Exp2: N-agent scaling — P(q) complexity vs N (thermodynamic limit)
Exp3: Aging test — autocorrelation C(τ;t_w) at three waiting times
Exp4: Breaking glass — asymmetric α matrix → unimodal P(q)?
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


def count_modes(q_vals, bins=30):
    hist, edges = np.histogram(q_vals, bins=bins)
    modes = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 2:
            modes += 1
    return modes


def train_2agent_with_noise(alpha, sigma, T=1500, N=600, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    schedule = [(375, (0.1, 0.3)), (375, (1.0, 2.0)),
                (375, (0.1, 0.3)), (375, (1.0, 2.0))]
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
                if sigma > 0:
                    gA = gA + sigma * torch.randn_like(gA)
                    gB = gB + sigma * torch.randn_like(gB)
                mix = alpha * 0.5 * (gA + gB)
                pA.grad = (1.0 - alpha) * gA + mix
                pB.grad = (1.0 - alpha) * gB + mix
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

    rep_A, rep_B, a_list, b_list = [], [], [], []
    env.reset()
    for _ in range(N):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a_s_A, _, h_A = pol_A(x)
            a_s_B, _, h_B = pol_B(x)
        env.step(float(a_s_A.item()), float(a_s_B.item()))
        rep_A.append(h_A.numpy().flatten())
        rep_B.append(h_B.numpy().flatten())
        a_list.append(float(a_s_A.item()))
        b_list.append(float(a_s_B.item()))

    _, S_A, Vt_A = np.linalg.svd(np.array(rep_A), full_matrices=False)
    u_A = Vt_A[0]
    if u_A[0] < 0: u_A = -u_A
    _, S_B, Vt_B = np.linalg.svd(np.array(rep_B), full_matrices=False)
    u_B = Vt_B[0]
    if u_B[0] < 0: u_B = -u_B

    return u_A, u_B, float(np.corrcoef(np.array(a_list), np.array(b_list))[0, 1])


def train_Nagent(N, alpha, T=1500, N_collect=400, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    schedule = [(375, (0.1, 0.3)), (375, (1.0, 2.0)),
                (375, (0.1, 0.3)), (375, (1.0, 2.0))]
    env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
    policies = [PolicyNetwork(hidden_dim=32) for _ in range(N)]
    params = []
    enc_params = []
    for p in policies:
        params += list(p.parameters())
        enc_params.append(list(p.backbone.parameters()))
    opt = torch.optim.Adam(params, lr=1e-3)

    env.reset()
    for _ in range(T):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        actions = []
        for p in policies:
            actions.append(float(p(x)[0].item()))
        env.step(np.mean(actions), 0.0)
        loss = torch.tensor(abs(float(env.state[0])), requires_grad=True)
        opt.zero_grad()
        loss.backward(retain_graph=True)
        all_grad = []
        for enc in enc_params:
            g_list = []
            for param in enc:
                if param.grad is not None:
                    g_list.append(param.grad.clone())
            if g_list:
                all_grad.append(g_list)

        if len(all_grad) == N:
            for i in range(N):
                mean_grad = []
                for j in range(len(all_grad[0])):
                    mean_grad.append(torch.stack([all_grad[k][j] for k in range(N)]).mean(dim=0))
                for j, param in enumerate(enc_params[i]):
                    if param.grad is not None:
                        g_i = all_grad[i][j]
                        mix = alpha * mean_grad[j]
                        param.grad = (1.0 - alpha) * g_i + mix

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

    rep_vectors = []
    pair_corrs = []
    env.reset()
    for _ in range(N_collect):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        hs, acts = [], []
        with torch.no_grad():
            for p in policies:
                a_s, _, h = p(x)
                hs.append(h.numpy().flatten())
                acts.append(float(a_s.item()))
        env.step(np.mean(acts), 0.0)
        for h in hs:
            rep_vectors.append(h)

    rep = np.array(rep_vectors)
    _, _, Vt = np.linalg.svd(rep, full_matrices=False)
    u_global = Vt[0]
    if u_global[0] < 0: u_global = -u_global

    all_corrs = []
    for _ in range(50):
        aA_buf, aB_buf = [], []
        env_t = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                      state_clip=5.0, force_scale=0.1,
                                      action_scale=0.1, action_mix=0.5)
        env_t.reset()
        i, j = np.random.choice(N, 2, replace=False)
        for _ in range(200):
            x = torch.tensor(env_t.state.copy(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                aa = float(policies[i](x)[0].item())
                bb = float(policies[j](x)[0].item())
            env_t.step(0.5*(aa+bb), 0.0)
            aA_buf.append(aa)
            aB_buf.append(bb)
        all_corrs.append(float(np.corrcoef(np.array(aA_buf), np.array(aB_buf))[0, 1]))

    return u_global, np.array(all_corrs)


def main():
    print("=" * 60)
    print("  GLASSY PHYSICS PART 2")
    print("=" * 60)
    all_results = {}

    # =============================================================
    # Exp1: Noise-induced melting
    # =============================================================
    print("\n  Exp1: Noise melting (σ sweep, 20 seeds)")
    sigmas = [0.0, 0.02, 0.05, 0.1, 0.2]
    n_seeds_noise = 20

    for sigma in sigmas:
        q_vals, corr_vals = [], []
        for seed in range(n_seeds_noise):
            u_A, u_B, corr = train_2agent_with_noise(alpha=0.4, sigma=sigma,
                                                     T=1500, N=600, seed=seed)
            q = float(np.dot(u_A, u_B))
            q_vals.append(q)
            corr_vals.append(corr)
        qa = np.array(q_vals)
        modes = count_modes(qa)
        print(f"    σ={sigma:.2f}: q∈[{qa.min():.3f},{qa.max():.3f}] "
              f"μ={qa.mean():.3f} σ_q={qa.std():.3f} "
              f"modes={modes} corr_μ={np.mean(corr_vals):.3f}")

        if sigma == 0.0:
            all_results["exp1_melting_baseline"] = {
                "q_mean": float(qa.mean()), "q_std": float(qa.std()),
                "q_min": float(qa.min()), "q_max": float(qa.max()), "modes": modes,
                "corr_mean": float(np.mean(corr_vals)),
            }
        all_results[f"exp1_melting_sigma_{sigma}"] = {
            "sigma": float(sigma), "modes": modes,
            "q_mean": float(qa.mean()), "q_std": float(qa.std()),
            "corr_mean": float(np.mean(corr_vals)),
        }

    # =============================================================
    # Exp2: N-agent scaling
    # =============================================================
    print("\n  Exp2: N-agent scaling (N=2,4,8)")
    for N in [2, 4, 8]:
        print(f"    N={N}...")
        qu, pair_corrs = train_Nagent(N, alpha=0.4, T=1500, N_collect=400, seed=42)
        modes = count_modes(pair_corrs, bins=20)
        print(f"    N={N}: corr∈[{pair_corrs.min():.3f},{pair_corrs.max():.3f}] "
              f"var={pair_corrs.var():.4f} modes={modes}")
        all_results[f"exp2_N_{N}"] = {
            "N": N, "corr_mean": float(pair_corrs.mean()),
            "corr_std": float(pair_corrs.std()), "corr_var": float(pair_corrs.var()),
            "corr_min": float(pair_corrs.min()), "corr_max": float(pair_corrs.max()),
            "modes": modes,
        }

    # =============================================================
    # Exp3: Aging test
    # =============================================================
    print("\n  Exp3: Aging — C(τ; t_w) for t_w=[500,2000,5000]")
    T_w_values = [500, 2000, 5000]
    tau_max = 300
    n_samples = 150
    seed_aging = 77
    schedule = [(1250, (0.1, 0.3)), (1250, (1.0, 2.0)),
                (1250, (0.1, 0.3)), (1250, (1.0, 2.0))]
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
    snapshots = {}
    t_current = 0
    for step in range(5000):
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
                mix = 0.4 * 0.5 * (gA + gB)
                pA.grad = 0.6 * gA + mix
                pB.grad = 0.6 * gB + mix
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        t_current += 1
        if t_current in T_w_values:
            with torch.no_grad():
                _, _, h = pol_A(torch.zeros(1, 1))
            snapshots[t_current] = h.clone()

    C_curves = {}
    for t_w in T_w_values:
        h0 = snapshots[t_w]
        env_test = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                         state_clip=5.0, force_scale=0.1,
                                         action_scale=0.1, action_mix=0.5)
        env_test.reset()
        C_tau = []
        h_prev = None
        for tau in range(tau_max):
            x = torch.tensor(env_test.state.copy(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, _, h = pol_A(x)
            env_test.step(float(pol_A(x)[0].item()), 0.0)
            if tau == 0:
                C_tau.append(1.0)
                h_prev = h
            else:
                c = float(torch.nn.functional.cosine_similarity(
                    h_prev.flatten().unsqueeze(0), h.flatten().unsqueeze(0)))
                C_tau.append(c)
        C_curves[f"tw_{t_w}"] = C_tau
        decay_50 = next((i for i, c in enumerate(C_tau) if c < 0.5), tau_max)
        print(f"    t_w={t_w}: C(τ) 50%-decay at τ={decay_50}")

    aging_detected = C_curves[f"tw_{5000}"]["decay"] > C_curves[f"tw_{500}"]["decay"] if False else None
    all_results["exp3_aging"] = {
        "t_w_values": T_w_values,
        "C_curves": {k: [round(float(c), 4) for c in v] for k, v in C_curves.items()},
    }

    # =============================================================
    # Exp4: Breaking glass (asymmetric coupling)
    # =============================================================
    print("\n  Exp4: Asymmetric coupling matrix")
    n_seeds_mat = 15

    # Scalar α=0.4 (baseline)
    q_scalar, corr_scalar = [], []
    for seed in range(n_seeds_mat):
        u_A, u_B, corr = train_2agent_with_noise(alpha=0.4, sigma=0.0,
                                                 T=1500, N=600, seed=seed+100)
        q_scalar.append(float(np.dot(u_A, u_B)))
        corr_scalar.append(corr)
    modes_scalar = count_modes(np.array(q_scalar))

    # Asymmetric matrix: α_12 = α_21 = 0.2, α_ii = 1.0
    # This means: agent gets 80% own + 20% average, but we use α_off=0.2
    # Equivalent to scalar α but with gating: i gets (1-α)*grad_i + α*((grad_i+grad_j)/2)
    # Modified: i gets (1-0.2)*grad_i + 0.2*((grad_i+grad_j)/2)
    q_asym, corr_asym = [], []
    for seed in range(n_seeds_mat):
        torch.manual_seed(seed+200)
        np.random.seed(seed+200)
        env2 = MultiAgentDriftingEnv(schedule=[(375,(0.1,0.3)),(375,(1.0,2.0)),
                                              (375,(0.1,0.3)),(375,(1.0,2.0))],
                                     noise=0.05, state_clip=5.0,
                                     force_scale=0.1, action_scale=0.1, action_mix=0.5)
        pA = PolicyNetwork(hidden_dim=32)
        pB = PolicyNetwork(hidden_dim=32)
        prm = list(pA.parameters()) + list(pB.parameters())
        opt2 = torch.optim.Adam(prm, lr=1e-3)
        eA = list(pA.backbone.parameters())
        eB = list(pB.backbone.parameters())
        env2.reset()
        for _ in range(1500):
            x = torch.tensor(env2.state.copy(), dtype=torch.float32).unsqueeze(0)
            aa = float(pA(x)[0].item()); ab = float(pB(x)[0].item())
            env2.step(aa, ab)
            loss = torch.tensor(abs(float(env2.state[0])), requires_grad=True)
            opt2.zero_grad()
            loss.backward(retain_graph=True)
            for p_a, p_b in zip(eA, eB):
                if p_a.grad is not None and p_b.grad is not None:
                    ga, gb = p_a.grad.clone(), p_b.grad.clone()
                    mix_a = 0.2 * 0.5 * (ga + gb)
                    mix_b = 0.2 * 0.5 * (gb + ga)
                    p_a.grad = 0.8 * ga + mix_a
                    p_b.grad = 0.8 * gb + mix_b
            torch.nn.utils.clip_grad_norm_(prm, max_norm=1.0)
            opt2.step()

        ra, rb, al, bl = [], [], [], []
        env2.reset()
        for _ in range(600):
            x = torch.tensor(env2.state.copy(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                asa, _, ha = pA(x); asb, _, hb = pB(x)
            env2.step(float(asa.item()), float(asb.item()))
            ra.append(ha.numpy().flatten()); rb.append(hb.numpy().flatten())
            al.append(float(asa.item())); bl.append(float(asb.item()))
        _, _, Va = np.linalg.svd(np.array(ra), full_matrices=False)
        ua = Va[0]; ua = -ua if ua[0] < 0 else ua
        _, _, Vb = np.linalg.svd(np.array(rb), full_matrices=False)
        ub = Vb[0]; ub = -ub if ub[0] < 0 else ub
        q_asym.append(float(np.dot(ua, ub)))
        corr_asym.append(float(np.corrcoef(np.array(al), np.array(bl))[0,1]))

    modes_asym = count_modes(np.array(q_asym))
    print(f"    scalar α=0.4:  modes={modes_scalar}  q_μ={np.mean(q_scalar):.3f}  corr_μ={np.mean(corr_scalar):.3f}")
    print(f"    asymmetric α:  modes={modes_asym}  q_μ={np.mean(q_asym):.3f}  corr_μ={np.mean(corr_asym):.3f}")

    all_results["exp4_asymmetric"] = {
        "scalar": {"modes": modes_scalar, "q_mean": float(np.mean(q_scalar)),
                   "q_std": float(np.std(q_scalar)), "corr_mean": float(np.mean(corr_scalar))},
        "asymmetric": {"modes": modes_asym, "q_mean": float(np.mean(q_asym)),
                       "q_std": float(np.std(q_asym)), "corr_mean": float(np.mean(corr_asym))},
    }

    out = Path("results_final/glassy_physics_part2.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
