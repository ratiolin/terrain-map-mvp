"""PC Decomposition: Per-Dimension Contribution to Action Correlation.

For each α, decomposes the total action product W_A·h × W_B·h into
contributions from each PC dimension. Reveals which latent directions
drive alignment vs opposition at resonance points.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def train_alpha(alpha, T=4000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    schedule = [
        (1000, (0.1, 0.3)), (1000, (1.0, 2.0)),
        (1000, (0.1, 0.3)), (1000, (1.0, 2.0)),
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
        loss = torch.tensor(abs(float(env.state[0])), dtype=torch.float32, requires_grad=True)
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
    return pol_A, pol_B


def collect(pol_A, pol_B, N=2000):
    schedule = [
        (500, (0.1, 0.3)), (500, (1.0, 2.0)),
        (500, (0.1, 0.3)), (500, (1.0, 2.0)),
    ]
    env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
    env.reset()
    H, aA, aB = [], [], []
    for _ in range(N):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            as_A, _, h = pol_A(x)
            as_B, _, _ = pol_B(x)
        env.step(float(as_A.item()), float(as_B.item()))
        H.append(h.numpy().flatten())
        aA.append(float(as_A.item()))
        aB.append(float(as_B.item()))
    return np.array(H), np.array(aA), np.array(aB)


def main():
    print("=" * 60)
    print("  PC DECOMPOSITION: PER-DIM CONTRIBUTION")
    print("=" * 60)

    alphas = [0.0, 0.4, 0.8, 1.0]
    all_results = {}

    for alpha in alphas:
        print(f"\n  α={alpha:.1f}")
        pol_A, pol_B = train_alpha(alpha, T=4000)
        H, aA, aB = collect(pol_A, pol_B, N=2000)

        W_A = pol_A.head_shape.weight.detach().numpy().flatten()
        W_B = pol_B.head_shape.weight.detach().numpy().flatten()

        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        Z = H @ Vt.T
        wA_pc = W_A @ Vt.T
        wB_pc = W_B @ Vt.T

        contributions = []
        for i in range(len(S)):
            var_i = float(np.var(Z[:, i]))
            contrib = float(wA_pc[i] * wB_pc[i] * var_i)
            contributions.append({
                "pc": int(i),
                "singular_value": float(S[i]),
                "var": round(var_i, 6),
                "wA_proj": round(float(wA_pc[i]), 6),
                "wB_proj": round(float(wB_pc[i]), 6),
                "contrib": round(contrib, 8),
            })

        total = sum(abs(c["contrib"]) for c in contributions)
        top5_idx = np.argsort([abs(c["contrib"]) for c in contributions])[-5:][::-1]

        print(f"    total |contrib|: {total:.6f}")
        print(f"    top-5 PCs:")
        for idx in top5_idx:
            c = contributions[idx]
            sign = "+" if c["contrib"] > 0 else "-"
            print(f"      PC{c['pc']:2d}: {sign}{abs(c['contrib']):.6f}  "
                  f"(wA={c['wA_proj']:+.4f} wB={c['wB_proj']:+.4f} var={c['var']:.4f})")

        all_results[f"alpha_{alpha}"] = {
            "contributions": contributions,
            "action_corr": float(np.corrcoef(aA, aB)[0, 1]),
        }

    out = Path("results_final/pc_decomposition.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
