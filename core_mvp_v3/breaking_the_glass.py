"""Breaking the Glass: Optimizer Sensitivity + Ergodic Regime Paths.

Exp1: Optimizer sweep — lr, momentum, clip_norm → jump_freq, dwell, modes
Exp2: Path A — spectral entropy bonus in loss
Exp3: Path C — chaotic environment (nonlinear noise)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def count_modes(arr, bins=20):
    hist, edges = np.histogram(arr, bins=bins)
    m = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 2:
            m += 1
    return m


# =============================================================
# Exp1: Optimizer sensitivity
# =============================================================
def train_optim_sweep(alpha, lr, momentum, clip, T=1500, N=600, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = MultiAgentDriftingEnv(schedule=[(375,(0.1,0.3)),(375,(1.0,2.0)),
                                         (375,(0.1,0.3)),(375,(1.0,2.0))],
                                noise=0.05, state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
    pA = PolicyNetwork(hidden_dim=32); pB = PolicyNetwork(hidden_dim=32)
    prms = list(pA.parameters()) + list(pB.parameters())
    opt = torch.optim.SGD(prms, lr=lr, momentum=momentum)
    eA = list(pA.backbone.parameters()); eB = list(pB.backbone.parameters())

    jumps, a_prev, b_prev = 0, 0.0, 0.0
    env.reset()
    for _ in range(T):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        aA = float(pA(x)[0].item()); aB = float(pB(x)[0].item())
        env.step(aA, aB)
        if abs(aA - a_prev) > 0.1: jumps += 1
        if abs(aB - b_prev) > 0.1: jumps += 1
        a_prev, b_prev = aA, aB
        loss = torch.tensor(abs(float(env.state[0])), requires_grad=True)
        opt.zero_grad()
        loss.backward(retain_graph=True)
        for pa, pb in zip(eA, eB):
            if pa.grad is not None and pb.grad is not None:
                ga, gb = pa.grad.clone(), pb.grad.clone()
                mix = alpha * 0.5 * (ga + gb)
                pa.grad = (1.0 - alpha) * ga + mix
                pb.grad = (1.0 - alpha) * gb + mix
        torch.nn.utils.clip_grad_norm_(prms, max_norm=clip)
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
    _, S, Vt = np.linalg.svd(np.array(H), full_matrices=False)
    er = int((np.cumsum(S**2) / (S**2).sum() > 0.95).argmax()) + 1
    return jumps, er, float(np.corrcoef(np.array(A), np.array(B))[0, 1])


# =============================================================
# Exp2: Path A — Spectral entropy bonus
# =============================================================
def train_entropy_bonus(beta, alpha=0.4, T=1500, N=600, seed=0):
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
        _, _, hA = pA(x); _, _, hB = pB(x)
        aA = float(pA.head_shape(hA).item())
        aB = float(pB.head_shape(hB).item())
        env.step(aA, aB)
        risk = torch.tensor(abs(float(env.state[0])), requires_grad=True)
        loss = risk
        if beta > 0:
            H_cat = torch.cat([hA, hB], dim=1)
            _, Sh, _ = torch.linalg.svd(H_cat, full_matrices=False)
            p_s = Sh / (Sh.sum() + 1e-12)
            ent = -torch.sum(p_s * torch.log(p_s + 1e-12))
            loss = risk - beta * ent
        opt.zero_grad()
        loss.backward(retain_graph=True)
        for pa, pb in zip(eA, eB):
            if pa.grad is not None and pb.grad is not None:
                ga, gb = pa.grad.clone(), pb.grad.clone()
                mix = alpha * 0.5 * (ga + gb)
                pa.grad = (1.0 - alpha) * ga + mix
                pb.grad = (1.0 - alpha) * gb + mix
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
    Hnp = np.array(H)
    _, S, Vt = np.linalg.svd(Hnp, full_matrices=False)
    er_95 = int((np.cumsum(S**2) / (S**2).sum() > 0.95).argmax()) + 1
    corr = float(np.corrcoef(np.array(A), np.array(B))[0, 1])
    return er_95, corr, float(S[0]**2/(S**2).sum())


# =============================================================
# Exp3: Path C — Chaotic environment
# =============================================================
class ChaoticDriftingEnv:
    def __init__(self):
        self._env = MultiAgentDriftingEnv(
            schedule=[(375,(0.1,0.3)),(375,(1.0,2.0)),
                      (375,(0.1,0.3)),(375,(1.0,2.0))],
            noise=0.05, state_clip=5.0, force_scale=0.1,
            action_scale=0.1, action_mix=0.5)

    def reset(self): return self._env.reset()

    def step(self, aA, aB):
        s = self._env.step(aA, aB)
        x = float(s[0])
        s[0] = x + 0.02 * np.sin(3.0 * x) + 0.01 * np.random.randn()
        return np.clip(s, -5.0, 5.0)

    @property
    def state(self): return self._env.state


def train_chaotic(alpha=0.4, T=1500, N=600, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = ChaoticDriftingEnv()
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
                mix = alpha * 0.5 * (ga + gb)
                pa.grad = (1.0 - alpha) * ga + mix
                pb.grad = (1.0 - alpha) * gb + mix
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
    _, S, Vt = np.linalg.svd(np.array(H), full_matrices=False)
    er = int((np.cumsum(S**2) / (S**2).sum() > 0.95).argmax()) + 1
    corr = float(np.corrcoef(np.array(A), np.array(B))[0, 1])
    return er, corr, float(S[0]**2/(S**2).sum())


def main():
    print("=" * 60)
    print("  BREAKING THE GLASS: ERGODIC REGIME PATHS")
    print("=" * 60)
    results = {}

    # =============================================================
    # Exp1: Optimizer sensitivity
    # =============================================================
    print("\n  Exp1: Optimizer sensitivity (α=0.4)")

    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    momenta = [0.0, 0.5, 0.9]
    clips = [0.1, 0.5, 1.0, 5.0]
    baseline_jumps, baseline_er, baseline_corr = train_optim_sweep(
        0.4, 1e-3, 0.0, 1.0, T=1500, N=600, seed=42)
    print(f"    baseline (lr=1e-3, m=0, clip=1): jumps={baseline_jumps} "
          f"er={baseline_er} corr={baseline_corr:+.3f}")

    for lr in lrs:
        j, er, corr = train_optim_sweep(0.4, lr, 0.0, 1.0, T=1500, N=600, seed=42)
        print(f"    lr={lr:.0e}: jumps={j:3d} er={er} corr={corr:+.3f}")

    for m in momenta:
        j, er, corr = train_optim_sweep(0.4, 1e-3, m, 1.0, T=1500, N=600, seed=42)
        print(f"    momentum={m:.1f}: jumps={j:3d} er={er} corr={corr:+.3f}")

    for c in clips:
        j, er, corr = train_optim_sweep(0.4, 1e-3, 0.0, c, T=1500, N=600, seed=42)
        print(f"    clip={c:.1f}: jumps={j:3d} er={er} corr={corr:+.3f}")

    results["exp1_optim_baseline"] = {"jumps": baseline_jumps, "er": baseline_er,
                                       "corr": baseline_corr}

    # =============================================================
    # Exp2: Path A — Spectral entropy bonus
    # =============================================================
    print("\n  Exp2: Path A — Entropy bonus sweep")
    betas = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    for beta in betas:
        er, corr, pc1 = train_entropy_bonus(beta, alpha=0.4, T=1500, N=600, seed=42)
        status = "ERGODIC?" if er > 1 else "COLLAPSED"
        print(f"    β={beta:.2f}: er(95%)={er:2d} pc1_var={pc1:.3f} "
              f"corr={corr:+.3f} [{status}]")

    # =============================================================
    # Exp3: Path C — Chaotic environment
    # =============================================================
    print("\n  Exp3: Path C — Chaotic environment")
    for seed in range(5):
        er, corr, pc1 = train_chaotic(alpha=0.4, T=1500, N=600, seed=seed)
        print(f"    seed={seed}: er(95%)={er:2d} pc1_var={pc1:.3f} corr={corr:+.3f}")

    results["exp3_chaotic"] = {"description": "nonlinear sin(x) + extra noise"}

    out = Path("results_final/breaking_the_glass.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
