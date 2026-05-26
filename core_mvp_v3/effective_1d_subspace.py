"""Effective 1D Subspace: Action Projection Onto First PC Direction.

For each coupling strength α, computes the first principal component of
the hidden states and projects both agents' head_shape weights onto it.
The sign and magnitude of alpha_A * alpha_B reveals whether agents
align, oppose, or become orthogonal in the principal latent direction.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def train_and_collect(alpha, T=4000, N=2000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    schedule = [
        (1000, (0.1, 0.3)), (1000, (1.0, 2.0)),
        (1000, (0.1, 0.3)), (1000, (1.0, 2.0)),
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
        a_s_A, _, _ = policy_A(x)
        a_s_B, _, _ = policy_B(x)
        aA, aB = float(a_s_A.item()), float(a_s_B.item())
        env.step(aA, aB)
        risk = abs(float(env.state[0]))
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

    H, A_act, aB_list = [], [], []
    env.reset()
    for _ in range(N):
        state = env.state.copy()
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a_s_A, _, h = policy_A(x)
            a_s_B, _, _ = policy_B(x)
        env.step(float(a_s_A.item()), float(a_s_B.item()))
        H.append(h.numpy().flatten())
        A_act.append(float(a_s_A.item()))
        aB_list.append(float(a_s_B.item()))

    return np.array(H), np.array(A_act), np.array(aB_list), policy_A, policy_B


def main():
    print("=" * 60)
    print("  EFFECTIVE 1D SUBSPACE PROJECTION")
    print("=" * 60)

    alphas = np.linspace(0, 1, 11)
    results = []

    for alpha in alphas:
        print(f"  α={alpha:.1f} ", end="", flush=True)
        H, aA, aB, pol_A, pol_B = train_and_collect(alpha, T=4000, N=2000)

        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        u = Vt[0]
        u /= np.linalg.norm(u)

        W_A = pol_A.head_shape.weight.detach().numpy().flatten()
        W_B = pol_B.head_shape.weight.detach().numpy().flatten()

        proj_A = float(W_A @ u)
        proj_B = float(W_B @ u)
        product = proj_A * proj_B
        action_corr = float(np.corrcoef(aA, aB)[0, 1])

        results.append({
            "alpha": float(alpha),
            "proj_A": round(proj_A, 4),
            "proj_B": round(proj_B, 4),
            "product": round(product, 6),
            "action_corr": round(action_corr, 4),
            "pc1_var_pct": round(float(S[0]**2 / (S**2).sum()), 3),
        })
        sign = "aligned" if product > 0 else "opposed"
        print(f"proj=({proj_A:+.4f},{proj_B:+.4f})  "
              f"product={product:+.6f}  corr={action_corr:+.4f}  [{sign}]")

    out = Path("results_final/effective_1d_subspace.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out}")

    products = [r["product"] for r in results]
    corrs = [r["action_corr"] for r in results]
    print(f"  product range: [{min(products):+.6f}, {max(products):+.6f}]")
    print(f"  corr range: [{min(corrs):+.4f}, {max(corrs):+.4f}]")


if __name__ == "__main__":
    main()
