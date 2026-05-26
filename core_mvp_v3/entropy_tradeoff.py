"""Entropy-Alignment Tradeoff: Pareto Frontier at α=0.4.

Sweeps entropy regularization strength λ to map the Pareto frontier
of spectral entropy vs action correlation. Tests whether the tradeoff
is structural (no point achieves both high H and high corr).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def spectral_entropy_torch(S):
    p = S / (S.sum() + 1e-12)
    return -torch.sum(p * torch.log(p + 1e-12))


def train_with_lambda(lam, alpha=0.4, T=2500, N=1200, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    schedule = [(625, (0.1, 0.3)), (625, (1.0, 2.0)),
                (625, (0.1, 0.3)), (625, (1.0, 2.0))]
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
        _, _, h_A = pol_A(x)
        _, _, h_B = pol_B(x)
        aA = float(pol_A.head_shape(h_A).item())
        aB = float(pol_B.head_shape(h_B).item())
        env.step(aA, aB)
        risk = torch.tensor(abs(float(env.state[0])), requires_grad=True)
        H_cat = torch.cat([h_A, h_B], dim=1)
        _, S_h, _ = torch.linalg.svd(H_cat, full_matrices=False)
        ent = spectral_entropy_torch(S_h)
        loss = risk - lam * ent
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

    H, a_list, b_list = [], [], []
    env.reset()
    for _ in range(N):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a_s_A, _, h = pol_A(x); a_s_B, _, _ = pol_B(x)
        env.step(float(a_s_A.item()), float(a_s_B.item()))
        H.append(h.numpy().flatten())
        a_list.append(float(a_s_A.item())); b_list.append(float(a_s_B.item()))

    Hnp = np.array(H)
    WA = pol_A.head_shape.weight.detach().numpy().flatten()
    WB = pol_B.head_shape.weight.detach().numpy().flatten()
    _, S, Vt = np.linalg.svd(Hnp, full_matrices=False)
    Z = Hnp @ Vt.T
    wA, wB = WA @ Vt.T, WB @ Vt.T
    contribs = np.array([wA[i] * wB[i] * float(np.var(Z[:, i])) for i in range(len(S))])
    p = np.abs(contribs) / (np.abs(contribs).sum() + 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    corr = float(np.abs(np.corrcoef(np.array(a_list), np.array(b_list))[0, 1]))
    return entropy, corr


def main():
    print("=" * 60)
    print("  ENTROPY-ALIGNMENT PARETO FRONTIER")
    print("=" * 60)

    lambdas = np.logspace(-4, 0.5, 24)
    results = []

    for lam in lambdas:
        H_val, corr_abs = train_with_lambda(lam, alpha=0.4, T=2500, N=1200)
        results.append({"lambda": float(lam), "entropy": round(H_val, 4),
                        "|corr|": round(corr_abs, 4)})
        print(f"  λ={lam:.4f}: H={H_val:.4f} |corr|={corr_abs:.4f}")

    H_vals = np.array([r["entropy"] for r in results])
    C_vals = np.array([r["|corr|"] for r in results])
    score = H_vals * C_vals
    max_score = float(score.max())
    max_possible = float(H_vals.max() * C_vals.max())
    tradeoff_ratio = max_score / (max_possible + 1e-8)

    print(f"\n  max(H×|corr|) = {max_score:.4f}")
    print(f"  max(H)×max(|corr|) = {max_possible:.4f}")
    print(f"  tradeoff ratio = {tradeoff_ratio:.4f} ({'STRUCTURAL' if tradeoff_ratio < 0.5 else 'weak'})")

    pareto_front = []
    for i in range(len(results)):
        dominated = False
        for j in range(len(results)):
            if i != j and H_vals[j] >= H_vals[i] and C_vals[j] >= C_vals[i]:
                if H_vals[j] > H_vals[i] or C_vals[j] > C_vals[i]:
                    dominated = True; break
        if not dominated:
            pareto_front.append({"lambda": results[i]["lambda"],
                                 "entropy": results[i]["entropy"],
                                 "|corr|": results[i]["|corr|"]})

    print(f"  Pareto points: {len(pareto_front)}")

    out = Path("results_final/entropy_alignment_tradeoff.json")
    with open(out, "w") as f:
        json.dump({"sweep": results, "pareto_front": pareto_front,
                   "max_score": max_score, "max_possible": max_possible,
                   "tradeoff_ratio": tradeoff_ratio}, f, indent=2)
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
