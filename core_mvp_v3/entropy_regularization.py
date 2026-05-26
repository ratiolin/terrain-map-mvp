"""Entropy Regularization: Preventing Latent Space Collapse.

Tests whether adding a spectral entropy penalty to the training loss
can prevent the monopole concentration at α=0.4 while maintaining
coordination.
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


def train_with_entropy_reg(alpha, lam, T=3000, N=1500, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    schedule = [
        (750, (0.1, 0.3)), (750, (1.0, 2.0)),
        (750, (0.1, 0.3)), (750, (1.0, 2.0)),
    ]
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

        risk_loss = torch.tensor(abs(float(env.state[0])), dtype=torch.float32, requires_grad=True)

        if lam > 0:
            h = torch.cat([pol_A.backbone(x), pol_B.backbone(x)], dim=1)
            _, S_h, _ = torch.linalg.svd(h, full_matrices=False)
            ent_loss = -spectral_entropy_torch(S_h)
            total_loss = risk_loss + lam * ent_loss
        else:
            total_loss = risk_loss

        opt.zero_grad()
        total_loss.backward(retain_graph=True)
        for pA, pB in zip(enc_A, enc_B):
            if pA.grad is not None and pB.grad is not None:
                gA, gB = pA.grad.clone(), pB.grad.clone()
                mix = alpha * 0.5 * (gA + gB)
                pA.grad = (1.0 - alpha) * gA + mix
                pB.grad = (1.0 - alpha) * gB + mix
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

    H, aA_list, aB_list = [], [], []
    env.reset()
    for _ in range(N):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a_s_A, _, h = pol_A(x)
            a_s_B, _, _ = pol_B(x)
        env.step(float(a_s_A.item()), float(a_s_B.item()))
        H.append(h.numpy().flatten())
        aA_list.append(float(a_s_A.item()))
        aB_list.append(float(a_s_B.item()))

    Hnp = np.array(H)
    W_A = pol_A.head_shape.weight.detach().numpy().flatten()
    W_B = pol_B.head_shape.weight.detach().numpy().flatten()

    U, S, Vt = np.linalg.svd(Hnp, full_matrices=False)
    Z = Hnp @ Vt.T
    wA_pc = W_A @ Vt.T
    wB_pc = W_B @ Vt.T
    contribs = np.array([wA_pc[i] * wB_pc[i] * float(np.var(Z[:, i])) for i in range(len(S))])

    p = np.abs(contribs) / (np.abs(contribs).sum() + 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    corr = float(np.corrcoef(np.array(aA_list), np.array(aB_list))[0, 1])

    return entropy, corr


def main():
    print("=" * 60)
    print("  ENTROPY REGULARIZATION: PREVENT COLLAPSE")
    print("=" * 60)

    alpha = 0.4
    lambdas = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    results = []

    for lam in lambdas:
        entropy, corr = train_with_entropy_reg(alpha, lam, T=3000, N=1500)
        results.append({"lambda": lam, "entropy": round(entropy, 4),
                        "corr": round(corr, 4)})
        status = "COLLAPSE" if entropy < 0.01 else ("STABLE" if entropy > 0.5 else "partial")
        print(f"  λ={lam:.2f}: entropy={entropy:.4f}  corr={corr:+.4f}  [{status}]")

    out = Path("results_final/entropy_regularization.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out}")

    baseline_ent = results[0]["entropy"]
    effective = [r for r in results if r["entropy"] > 0.5 and abs(r["corr"]) > 0.5]
    if effective:
        print(f"  effective λ range: {[(r['lambda'], r['entropy'], r['corr']) for r in effective]}")
    else:
        print(f"  no λ successfully prevented collapse while maintaining correlation")


if __name__ == "__main__":
    main()
