"""Phase Boundary: Dense α-Sweep with Spectral Entropy and Action Correlation.

Scans 51 α values for hidden_dim ∈ {16, 32, 64} to locate the phase
transition boundary where entropy collapse and sign flips occur.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def spectral_entropy(contribs):
    p = np.abs(contribs) / (np.abs(contribs).sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def train_and_analyze(alpha, hidden_dim=32, T=3000, N=1500, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    schedule = [
        (750, (0.1, 0.3)), (750, (1.0, 2.0)),
        (750, (0.1, 0.3)), (750, (1.0, 2.0)),
    ]
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
    aA_arr = np.array(aA_list)
    aB_arr = np.array(aB_list)

    W_A = pol_A.head_shape.weight.detach().numpy().flatten()
    W_B = pol_B.head_shape.weight.detach().numpy().flatten()

    U, S, Vt = np.linalg.svd(Hnp, full_matrices=False)
    Z = Hnp @ Vt.T
    wA_pc = W_A @ Vt.T
    wB_pc = W_B @ Vt.T
    contribs = np.array([wA_pc[i] * wB_pc[i] * float(np.var(Z[:, i])) for i in range(len(S))])

    entropy = spectral_entropy(contribs)
    action_corr = float(np.corrcoef(aA_arr, aB_arr)[0, 1])
    max_pc = int(np.argmax(np.abs(contribs)))

    return {"alpha": float(alpha), "entropy": round(entropy, 4),
            "corr": round(action_corr, 4), "max_pc": max_pc,
            "pc1_var": round(float(S[0]**2 / (S**2).sum()), 3)}


def main():
    print("=" * 60)
    print("  PHASE BOUNDARY: DENSE α-SWEEP")
    print("=" * 60)

    alphas = np.linspace(0, 1, 51)
    dims = [16, 32, 64]

    all_data = {}

    for hidden_dim in dims:
        print(f"\n  hidden_dim={hidden_dim}")
        sweep = []
        for i, alpha in enumerate(alphas):
            r = train_and_analyze(alpha, hidden_dim=hidden_dim, T=3000, N=1500)
            sweep.append(r)
            marker = ""
            if r["entropy"] < 0.01:
                marker = " ← COLLAPSE"
            elif abs(r["corr"]) < 0.1:
                marker = " ← DECOUPLED"
            elif r["corr"] < -0.3:
                marker = " ← OPPOSED"
            if i % 5 == 0 or marker:
                print(f"    α={alpha:.3f}: entropy={r['entropy']:.4f}  "
                      f"corr={r['corr']:+.4f}  max_pc={r['max_pc']}  "
                      f"pc1_var={r['pc1_var']:.3f}{marker}")
        all_data[f"dim_{hidden_dim}"] = sweep

    out = Path("results_final/phase_boundary.json")
    with open(out, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n  saved → {out}")

    for hidden_dim in dims:
        sweep = all_data[f"dim_{hidden_dim}"]
        ents = [r["entropy"] for r in sweep]
        corrs = [r["corr"] for r in sweep]
        collapses = [r["alpha"] for r in sweep if r["entropy"] < 0.01]
        sign_flips = []
        for i in range(len(corrs) - 1):
            if corrs[i] * corrs[i+1] < 0:
                sign_flips.append(round(float(alphas[i]), 3))
        print(f"\n  dim={hidden_dim}: entropy∈[{min(ents):.4f},{max(ents):.4f}]  "
              f"collapses@α={collapses}  sign_flips@α={sign_flips}")


if __name__ == "__main__":
    main()
