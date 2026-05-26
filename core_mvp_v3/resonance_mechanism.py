"""Resonance Mechanism Test: Jacobian Singularity at α=0.4, 0.8.

Trains two agents at key coupling strengths α, then runs:
  Exp1 — Jacobian spectral analysis (SVD, condition number, rank)
  Exp2 — Controllability subspace vs Jacobian alignment
  Exp3 — Orthogonal complement probe signal leakage
  Exp4 — Policy mapping linearity (H→A R²)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler


def train_alpha(alpha, T=4000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    schedule = [
        (1000, (0.1, 0.3)),
        (1000, (1.0, 2.0)),
        (1000, (0.1, 0.3)),
        (1000, (1.0, 2.0)),
    ]
    env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
    policy_A = PolicyNetwork(hidden_dim=32)
    policy_B = PolicyNetwork(hidden_dim=32)
    params = list(policy_A.parameters()) + list(policy_B.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    enc_A = list(policy_A.backbone.parameters())
    enc_B = list(policy_B.backbone.parameters())

    env.reset()
    for _ in range(T):
        state = env.state.copy()
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        a_s_A, _, h_A = policy_A(x)
        a_s_B, _, h_B = policy_B(x)
        aA, aB = float(a_s_A.item()), float(a_s_B.item())
        next_state = env.step(aA, aB)
        risk = abs(float(next_state[0]))
        loss = torch.tensor(risk, dtype=torch.float32, requires_grad=True)
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

    return policy_A, policy_B


def collect_data(policy_A, policy_B, N=2000):
    schedule = [
        (500, (0.1, 0.3)),
        (500, (1.0, 2.0)),
        (500, (0.1, 0.3)),
        (500, (1.0, 2.0)),
    ]
    env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
    env.reset()
    H, A, S = [], [], []
    for _ in range(N):
        state = env.state.copy()
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a_s, _, h = policy_A(x)
        a = float(a_s.item())
        env.step(a, 0.0)
        H.append(h.numpy().flatten())
        A.append(a)
        S.append(state.copy())
    return np.array(H), np.array(A), np.array(S)


def compute_jacobian(policy, state_np):
    x = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
    x.requires_grad_(False)
    with torch.no_grad():
        a_s, _, h_detached = policy(x)
    h = h_detached.clone().detach().requires_grad_(True)
    a = policy.head_shape(h)
    J = []
    for i in range(a.shape[-1]):
        grad = torch.autograd.grad(a[0, i], h, retain_graph=True)[0]
        J.append(grad.detach().numpy().flatten())
    return np.stack(J, axis=0)


def orthogonal_complement(Q):
    U, _, _ = np.linalg.svd(Q, full_matrices=True)
    return U[:, Q.shape[1]:]


def fit_probe_r2(X, y, alpha=1.0):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(Xs, y)
    return model.score(Xs, y)


def main():
    print("=" * 60)
    print("  RESONANCE MECHANISM TEST")
    print("=" * 60)

    alphas = [0.0, 0.4, 0.8, 1.0]
    data = {}

    for alpha in alphas:
        print(f"\n  α={alpha:.1f} — training...")
        pol_A, pol_B = train_alpha(alpha, T=4000, seed=42)
        H, A_act, S = collect_data(pol_A, pol_B, N=2000)
        data[alpha] = {"policy_A": pol_A, "H": H, "A": A_act, "S": S}

    print("\n" + "=" * 60)
    print("  RUNNING 4 EXPERIMENTS")
    print("=" * 60)

    results = {}

    for alpha in alphas:
        pol = data[alpha]["policy_A"]
        H = data[alpha]["H"]
        S = data[alpha]["S"]
        A_act = data[alpha]["A"]
        r = {}

        print(f"\n{'─' * 50}")
        print(f"  α = {alpha:.1f}")
        print(f"{'─' * 50}")

        # Exp 1: Jacobian spectral analysis
        J_list = []
        for i in range(0, len(H), 5):
            J_list.append(compute_jacobian(pol, S[i]))
        J_all = np.concatenate(J_list, axis=0)
        U, S, Vt = np.linalg.svd(J_all, full_matrices=False)
        r["exp1_spectral"] = {
            "min_singular": float(S.min()),
            "max_singular": float(S.max()),
            "condition_number": float(S.max() / (S.min() + 1e-8)),
            "effective_rank": int((S > 1e-3).sum()),
            "sv_ratio_top5": float(S[:5].sum() / (S.sum() + 1e-8)),
        }
        print(f"  Exp1: cond={r['exp1_spectral']['condition_number']:.1f}  "
              f"eff_rank={r['exp1_spectral']['effective_rank']}  "
              f"SV5%/total={r['exp1_spectral']['sv_ratio_top5']:.3f}  "
              f"S_min={r['exp1_spectral']['min_singular']:.6f}")

        # Exp 2: Controllability subspace alignment
        scaler = StandardScaler()
        Hs = scaler.fit_transform(H)
        ctrl_probe = Ridge(alpha=1.0)
        ctrl_probe.fit(Hs, A_act)
        coef = np.abs(ctrl_probe.coef_)
        top4 = np.argsort(coef)[-4:]
        Q_ctrl = np.eye(H.shape[1])[top4]
        Q_ctrl, _ = np.linalg.qr(Q_ctrl.T)
        alignment_val = float(np.linalg.norm(J_all @ Q_ctrl) / (np.linalg.norm(J_all) + 1e-8))
        r["exp2_alignment"] = {"alignment": alignment_val, "top4_dims": top4.tolist()}
        print(f"  Exp2: alignment={alignment_val:.4f}  top4={top4.tolist()}")

        # Exp 3: Orthogonal complement signal leakage
        X_ctrl = H @ Q_ctrl
        Q_orth = orthogonal_complement(Q_ctrl)
        X_orth = H @ Q_orth
        R2_sub = fit_probe_r2(X_ctrl, A_act)
        R2_orth = fit_probe_r2(X_orth, A_act)
        leakage = R2_orth / (R2_sub + 1e-8)
        r["exp3_orthogonal"] = {"R2_sub": round(R2_sub, 4), "R2_orth": round(R2_orth, 4),
                                "leakage_ratio": round(leakage, 2)}
        print(f"  Exp3: R2_sub={R2_sub:.4f}  R2_orth={R2_orth:.4f}  "
              f"leakage={leakage:.2f}")

        # Exp 4: Policy mapping linearity
        model = LinearRegression()
        model.fit(H, A_act)
        R2_linear = model.score(H, A_act)
        r["exp4_linear"] = {"R2_H_to_A": round(R2_linear, 4)}
        print(f"  Exp4: R²(H→A)={R2_linear:.4f}")

        results[f"alpha_{alpha}"] = r

    out_path = Path("results_final/resonance_mechanism.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
