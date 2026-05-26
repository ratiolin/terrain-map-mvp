"""Finding the Theorem Boundary: What Actually Breaks Rank-1.

1. Anti-consensus: negative α (gradient repulsion)
2. Strong orthogonalization: hard orthogonality penalty on representations
3. Nonlinear coupling: multiplicative agent interaction
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def effective_rank(S, thresh=0.95):
    cum = np.cumsum(S ** 2)
    return int((cum / cum[-1] > thresh).argmax()) + 1


def collect_metrics(env, pA, pB, N=600):
    H, A, B = [], [], []
    env.reset()
    for _ in range(N):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            asA, _, hA = pA(x); asB, _, hB = pB(x)
        env.step(float(asA.item()), float(asB.item()))
        H.append(hA.numpy().flatten())
        A.append(float(asA.item())); B.append(float(asB.item()))
    Hnp = np.array(H)
    _, S, _ = np.linalg.svd(Hnp, full_matrices=False)
    er = effective_rank(S)
    corr = float(np.corrcoef(np.array(A), np.array(B))[0, 1])
    return er, corr, float(S[0]**2/(S**2).sum()), float(S[1]**2/(S**2).sum())


def make_env(T=1500):
    return MultiAgentDriftingEnv(schedule=[(T//4,(0.1,0.3)),(T//4,(1.0,2.0)),
                                           (T//4,(0.1,0.3)),(T//4,(1.0,2.0))],
                                 noise=0.05, state_clip=5.0, force_scale=0.1,
                                 action_scale=0.1, action_mix=0.5)


# =============================================================
# Method 1: Anti-consensus — negative α (gradient repulsion)
# =============================================================
def train_anti_consensus(alpha_neg, T=1500, N=600, seed=0):
    """alpha_neg > 0 means repulsion: grad_A gets pushed AWAY from grad_B"""
    torch.manual_seed(seed); np.random.seed(seed)
    env = make_env(T)
    pA = PolicyNetwork(hidden_dim=32); pB = PolicyNetwork(hidden_dim=32)
    prms = list(pA.parameters()) + list(pB.parameters())
    opt = torch.optim.Adam(prms, lr=1e-3)
    eA = list(pA.backbone.parameters()); eB = list(pB.backbone.parameters())
    env.reset()
    for _ in range(T):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        aA = float(pA(x)[0].item()); aB = float(pB(x)[0].item())
        env.step(aA, aB)
        loss = torch.tensor(abs(float(env.state[0])), requires_grad=True)
        opt.zero_grad(); loss.backward(retain_graph=True)
        for pa, pb in zip(eA, eB):
            if pa.grad is not None and pb.grad is not None:
                ga, gb = pa.grad.clone(), pb.grad.clone()
                repulsion = alpha_neg * 0.5 * (ga - gb)
                pa.grad = ga + repulsion
                pb.grad = gb - repulsion
        torch.nn.utils.clip_grad_norm_(prms, max_norm=1.0)
        opt.step()
    er, corr, pc1, pc2 = collect_metrics(env, pA, pB, N)
    return er, corr, pc1, pc2


# =============================================================
# Method 2: Strong orthogonalization
# =============================================================
def train_orthogonal_loss(lam_orth, alpha=0.0, T=1500, N=600, seed=0):
    """Hard penalty: ||Cov(H) - I||² + gradient mixing"""
    torch.manual_seed(seed); np.random.seed(seed)
    env = make_env(T)
    pA = PolicyNetwork(hidden_dim=32); pB = PolicyNetwork(hidden_dim=32)
    prms = list(pA.parameters()) + list(pB.parameters())
    opt = torch.optim.Adam(prms, lr=1e-3)
    eA = list(pA.backbone.parameters()); eB = list(pB.backbone.parameters())
    env.reset()
    for _ in range(T):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        _, _, hA = pA(x); _, _, hB = pB(x)
        aA = float(pA.head_shape(hA).item())
        aB = float(pB.head_shape(hB).item())
        env.step(aA, aB)
        risk = torch.tensor(abs(float(env.state[0])), requires_grad=True)

        Hcat = torch.cat([hA, hB], dim=1)
        Hn = Hcat - Hcat.mean(dim=1, keepdim=True)
        cov = Hn.T @ Hn / (Hn.shape[0] - 1 + 1e-8)
        orth_loss = torch.norm(cov - torch.eye(cov.shape[0]), p='fro')

        loss = risk + lam_orth * orth_loss
        opt.zero_grad(); loss.backward(retain_graph=True)
        for pa, pb in zip(eA, eB):
            if pa.grad is not None and pb.grad is not None:
                ga, gb = pa.grad.clone(), pb.grad.clone()
                mix = alpha * 0.5 * (ga + gb)
                pa.grad = (1.0 - alpha) * ga + mix
                pb.grad = (1.0 - alpha) * gb + mix
        torch.nn.utils.clip_grad_norm_(prms, max_norm=1.0)
        opt.step()
    er, corr, pc1, pc2 = collect_metrics(env, pA, pB, N)
    return er, corr, pc1, pc2


# =============================================================
# Method 3: Nonlinear coupling — multiplicative interaction
# =============================================================
def train_multiplicative(mix_strength=0.5, T=1500, N=600, seed=0):
    """grad_A = grad_A * (1 + mix * sign(dot(grad_A, grad_B)))"""
    torch.manual_seed(seed); np.random.seed(seed)
    env = make_env(T)
    pA = PolicyNetwork(hidden_dim=32); pB = PolicyNetwork(hidden_dim=32)
    prms = list(pA.parameters()) + list(pB.parameters())
    opt = torch.optim.Adam(prms, lr=1e-3)
    eA = list(pA.backbone.parameters()); eB = list(pB.backbone.parameters())
    env.reset()
    for _ in range(T):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        aA = float(pA(x)[0].item()); aB = float(pB(x)[0].item())
        env.step(aA, aB)
        loss = torch.tensor(abs(float(env.state[0])), requires_grad=True)
        opt.zero_grad(); loss.backward(retain_graph=True)
        for pa, pb in zip(eA, eB):
            if pa.grad is not None and pb.grad is not None:
                ga, gb = pa.grad.clone(), pb.grad.clone()
                gfA, gfB = ga.flatten(), gb.flatten()
                dot = torch.dot(gfA, gfB) / (torch.norm(gfA)*torch.norm(gfB) + 1e-12)
                sign = torch.sign(dot)
                pa.grad = ga * (1.0 + mix_strength * sign)
                pb.grad = gb * (1.0 + mix_strength * sign)
        torch.nn.utils.clip_grad_norm_(prms, max_norm=1.0)
        opt.step()
    er, corr, pc1, pc2 = collect_metrics(env, pA, pB, N)
    return er, corr, pc1, pc2


def main():
    print("=" * 60)
    print("  THEOREM BOUNDARY: FINDING er > 1")
    print("=" * 60)
    results = {}

    # =============================================================
    # Method 1: Anti-consensus (negative α)
    # =============================================================
    print("\n  Method 1: Anti-consensus (gradient repulsion)")
    alphas_neg = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    for an in alphas_neg:
        er, corr, pc1, pc2 = train_anti_consensus(an, T=1500, N=600, seed=42)
        status = "✓ BROKEN" if er > 1 else "✗"
        print(f"    α_neg={an:.1f}: er={er} pc1={pc1:.3f} pc2={pc2:.3f} "
              f"corr={corr:+.3f} [{status}]")
        results[f"anti_consensus_{an}"] = {"er": er, "pc1": round(pc1, 3),
                                            "pc2": round(pc2, 3), "corr": round(corr, 3)}

    # =============================================================
    # Method 2: Strong orthogonalization
    # =============================================================
    print("\n  Method 2: Strong orthogonalization penalty")
    for lam in [0.0, 0.001, 0.01, 0.1, 1.0]:
        for alpha in [0.0, 0.4]:
            er, corr, pc1, pc2 = train_orthogonal_loss(lam, alpha, T=1500, N=600, seed=42)
            status = "✓ BROKEN" if er > 1 else "✗"
            print(f"    λ={lam:.3f} α={alpha:.1f}: er={er} pc1={pc1:.3f} "
                  f"pc2={pc2:.3f} corr={corr:+.3f} [{status}]")
            results[f"orth_{lam}_{alpha}"] = {"er": er, "pc1": round(pc1, 3),
                                               "pc2": round(pc2, 3), "corr": round(corr, 3)}

    # =============================================================
    # Method 3: Multiplicative nonlinear coupling
    # =============================================================
    print("\n  Method 3: Multiplicative coupling")
    for ms in [0.0, 0.1, 0.5, 1.0, 2.0]:
        er, corr, pc1, pc2 = train_multiplicative(ms, T=1500, N=600, seed=42)
        status = "✓ BROKEN" if er > 1 else "✗"
        print(f"    mix={ms:.1f}: er={er} pc1={pc1:.3f} pc2={pc2:.3f} "
              f"corr={corr:+.3f} [{status}]")

    out = Path("results_final/theorem_boundary.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
