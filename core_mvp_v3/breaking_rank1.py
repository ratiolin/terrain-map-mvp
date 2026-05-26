"""Breaking the Rank-1 Attractor: Three Architectural Methods.

Method 1: Orthogonal gradient projection (Gram-Schmidt before mixing)
Method 2: Partial coupling (split shared/private hidden dims)
Method 3: Nonlinear gated coupling (state-dependent α(x))

Key metric: effective rank (95% variance threshold) > 1
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


# =============================================================
# Method 1: Orthogonal gradient projection
# =============================================================
def train_orthogonal(alpha=0.4, T=1500, N=600, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = MultiAgentDriftingEnv(schedule=[(375,(0.1,0.3)),(375,(1.0,2.0)),
                                         (375,(0.1,0.3)),(375,(1.0,2.0))],
                                noise=0.05, state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
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
                ga = pa.grad.clone(); gb = pb.grad.clone()
                ga_f = ga.flatten(); gb_f = gb.flatten()
                dot = torch.dot(ga_f, gb_f) / (torch.norm(ga_f) * torch.norm(gb_f) + 1e-12)
                ga_f = ga_f - dot * gb_f
                gb_f = gb_f - dot * ga_f
                mix = alpha * 0.5 * (ga_f + gb_f)
                pa.grad = ((1.0 - alpha) * ga_f + mix).reshape_as(ga)
                pb.grad = ((1.0 - alpha) * gb_f + mix).reshape_as(gb)
        torch.nn.utils.clip_grad_norm_(prms, max_norm=1.0)
        opt.step()

    H, A, B = [], [], []
    env.reset()
    for _ in range(N):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            asA, _, h = pA(x); asB, _, _ = pB(x)
        env.step(float(asA.item()), float(asB.item()))
        H.append(h.numpy().flatten())
        A.append(float(asA.item())); B.append(float(asB.item()))
    _, S, _ = np.linalg.svd(np.array(H), full_matrices=False)
    er = effective_rank(S)
    corr = float(np.corrcoef(np.array(A), np.array(B))[0, 1])
    return er, corr, float(S[0]**2/(S**2).sum())


# =============================================================
# Method 2: Partial coupling (split architecture)
# =============================================================
class SplitPolicyNetwork(nn.Module):
    def __init__(self, state_dim=1, shared_dim=16, private_dim=16):
        super().__init__()
        total = shared_dim + private_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.shared_enc = nn.Sequential(nn.Linear(state_dim, shared_dim), nn.ReLU())
        self.private_enc = nn.Sequential(nn.Linear(state_dim, private_dim), nn.ReLU())
        self.combine = nn.Sequential(
            nn.Linear(total, total), nn.ReLU(),
            nn.Linear(total, total), nn.ReLU(),
        )
        self.head_shape = nn.Linear(total, 1)
        self.head_adapt = nn.Linear(total, 1)
        nn.init.normal_(self.head_shape.weight, 0, 0.1)
        nn.init.normal_(self.head_adapt.weight, 0, 0.1)

    def forward(self, state):
        hs = self.shared_enc(state)
        hp = self.private_enc(state)
        h = self.combine(torch.cat([hs, hp], dim=-1))
        a_s = self.head_shape(h); a_a = self.head_adapt(h)
        return a_s, a_a, h

    def shared_enc_params(self):
        return list(self.shared_enc.parameters())

    def nonshared_params(self):
        return list(self.private_enc.parameters()) + list(self.combine.parameters()) \
               + list(self.head_shape.parameters()) + list(self.head_adapt.parameters())


def train_partial(alpha=0.4, shared_dim=16, T=1500, N=600, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = MultiAgentDriftingEnv(schedule=[(375,(0.1,0.3)),(375,(1.0,2.0)),
                                         (375,(0.1,0.3)),(375,(1.0,2.0))],
                                noise=0.05, state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
    pA = SplitPolicyNetwork(shared_dim=shared_dim, private_dim=32-shared_dim)
    pB = SplitPolicyNetwork(shared_dim=shared_dim, private_dim=32-shared_dim)
    params = list(pA.parameters()) + list(pB.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    env.reset()
    for _ in range(T):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        aA = float(pA(x)[0].item()); aB = float(pB(x)[0].item())
        env.step(aA, aB)
        loss = torch.tensor(abs(float(env.state[0])), requires_grad=True)
        opt.zero_grad(); loss.backward(retain_graph=True)
        for sA, sB in zip(pA.shared_enc_params(), pB.shared_enc_params()):
            if sA.grad is not None and sB.grad is not None:
                ga, gb = sA.grad.clone(), sB.grad.clone()
                mix = alpha * 0.5 * (ga + gb)
                sA.grad = (1.0 - alpha) * ga + mix
                sB.grad = (1.0 - alpha) * gb + mix
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

    H, A, B = [], [], []
    env.reset()
    for _ in range(N):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            asA, _, h = pA(x); asB, _, _ = pB(x)
        env.step(float(asA.item()), float(asB.item()))
        H.append(h.numpy().flatten())
        A.append(float(asA.item())); B.append(float(asB.item()))
    _, S, _ = np.linalg.svd(np.array(H), full_matrices=False)
    er = effective_rank(S)
    corr = float(np.corrcoef(np.array(A), np.array(B))[0, 1])
    return er, corr, float(S[0]**2/(S**2).sum())


# =============================================================
# Method 3: Nonlinear gated coupling
# =============================================================
class GateNetwork(nn.Module):
    def __init__(self, state_dim=1, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


def train_gated(alpha_base=0.4, T=1500, N=600, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = MultiAgentDriftingEnv(schedule=[(375,(0.1,0.3)),(375,(1.0,2.0)),
                                         (375,(0.1,0.3)),(375,(1.0,2.0))],
                                noise=0.05, state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
    pA = PolicyNetwork(hidden_dim=32); pB = PolicyNetwork(hidden_dim=32)
    gate = GateNetwork()
    prms = list(pA.parameters()) + list(pB.parameters()) + list(gate.parameters())
    opt = torch.optim.Adam(prms, lr=1e-3)
    eA = list(pA.backbone.parameters()); eB = list(pB.backbone.parameters())
    env.reset()
    for _ in range(T):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        aA = float(pA(x)[0].item()); aB = float(pB(x)[0].item())
        env.step(aA, aB)
        loss = torch.tensor(abs(float(env.state[0])), requires_grad=True)
        opt.zero_grad(); loss.backward(retain_graph=True)
        g = gate(x).item()
        alpha_eff = alpha_base * g
        for pa, pb in zip(eA, eB):
            if pa.grad is not None and pb.grad is not None:
                ga, gb = pa.grad.clone(), pb.grad.clone()
                mix = alpha_eff * 0.5 * (ga + gb)
                pa.grad = (1.0 - alpha_eff) * ga + mix
                pb.grad = (1.0 - alpha_eff) * gb + mix
        torch.nn.utils.clip_grad_norm_(prms, max_norm=1.0)
        opt.step()

    H, A, B = [], [], []
    env.reset()
    for _ in range(N):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            asA, _, h = pA(x); asB, _, _ = pB(x)
        env.step(float(asA.item()), float(asB.item()))
        H.append(h.numpy().flatten())
        A.append(float(asA.item())); B.append(float(asB.item()))
    _, S, _ = np.linalg.svd(np.array(H), full_matrices=False)
    er = effective_rank(S)
    corr = float(np.corrcoef(np.array(A), np.array(B))[0, 1])
    return er, corr, float(S[0]**2/(S**2).sum())


def main():
    print("=" * 60)
    print("  BREAKING RANK-1: THREE ARCHITECTURAL METHODS")
    print("=" * 60)
    results = {}

    print("\n  Method 1: Orthogonal gradient projection")
    for seed in range(6):
        er, corr, pc1 = train_orthogonal(alpha=0.4, T=1500, N=600, seed=seed)
        status = "✓ BROKEN" if er > 1 else "✗ rank=1"
        print(f"    seed={seed}: er={er} pc1={pc1:.3f} corr={corr:+.3f} [{status}]")

    print("\n  Method 2: Partial coupling (split shared/private)")
    for shared_dim in [4, 8, 16, 24]:
        er, corr, pc1 = train_partial(alpha=0.4, shared_dim=shared_dim, T=1500, N=600, seed=42)
        status = "✓ BROKEN" if er > 1 else "✗ rank=1"
        print(f"    shared={shared_dim:2d} private={32-shared_dim:2d}: "
              f"er={er} pc1={pc1:.3f} corr={corr:+.3f} [{status}]")
        results[f"m2_shared{shared_dim}"] = {"er": er, "pc1_var": round(pc1, 3),
                                              "corr": round(corr, 3)}

    print("\n  Method 3: Nonlinear gated coupling")
    for seed in range(6):
        er, corr, pc1 = train_gated(alpha_base=0.4, T=1500, N=600, seed=seed)
        status = "✓ BROKEN" if er > 1 else "✗ rank=1"
        print(f"    seed={seed}: er={er} pc1={pc1:.3f} corr={corr:+.3f} [{status}]")

    out = Path("results_final/breaking_rank1.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
