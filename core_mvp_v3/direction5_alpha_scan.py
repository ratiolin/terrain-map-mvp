"""Direction 5: CKA Alpha Scan — Gradient Coupling Experiment.

Trains two agents from scratch with a controlled gradient-mixing
coefficient α. Maps the relationship between encoder gradient sharing
and emergent coordination (CKA, action correlation, stability).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def linear_CKA(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K = X @ X.T
    L = Y @ Y.T
    return float(np.sum(K * L) / (np.linalg.norm(K) * np.linalg.norm(L) + 1e-8))


def train_with_alpha(alpha, T=4000, seed=0):
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

    params = (list(policy_A.parameters()) + list(policy_B.parameters()))
    opt = torch.optim.Adam(params, lr=1e-3)

    enc_params_A = list(policy_A.backbone.parameters())
    enc_params_B = list(policy_B.backbone.parameters())

    h_buf_A, h_buf_B = [], []
    a_buf_A, a_buf_B = [], []
    loss_history = []

    env.reset()
    for step in range(T):
        state = env.state.copy()
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        a_s_A, _, h_A = policy_A(x)
        a_s_B, _, h_B = policy_B(x)

        a_A = float(a_s_A.item())
        a_B = float(a_s_B.item())
        next_state = env.step(a_A, a_B)
        risk = abs(float(next_state[0]))

        loss = torch.tensor(risk, dtype=torch.float32, requires_grad=True)
        opt.zero_grad()
        loss.backward(retain_graph=True)

        for pA, pB in zip(enc_params_A, enc_params_B):
            if pA.grad is not None and pB.grad is not None:
                gA, gB = pA.grad.clone(), pB.grad.clone()
                mixed = alpha * 0.5 * (gA + gB)
                pA.grad = (1.0 - alpha) * gA + mixed
                pB.grad = (1.0 - alpha) * gB + mixed

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        h_buf_A.append(h_A.detach().cpu().numpy().flatten())
        h_buf_B.append(h_B.detach().cpu().numpy().flatten())
        a_buf_A.append(a_A)
        a_buf_B.append(a_B)
        loss_history.append(risk)

    H_A = np.array(h_buf_A)
    H_B = np.array(h_buf_B)
    aA = np.array(a_buf_A)
    aB = np.array(a_buf_B)

    cka = linear_CKA(H_A, H_B)
    corr = float(np.corrcoef(aA, aB)[0, 1])
    avg_loss = float(np.mean(loss_history))

    collapse_sigma = None
    for sigma in np.linspace(0, 0.2, 40):
        noise = np.random.normal(0, sigma, size=aB.shape)
        c_noisy = float(np.corrcoef(aA, aB + noise)[0, 1])
        if abs(c_noisy) < 0.2 and collapse_sigma is None:
            collapse_sigma = float(sigma)
            break

    return {"alpha": float(alpha), "cka": round(cka, 4), "corr": round(corr, 4),
            "avg_loss": round(avg_loss, 6), "collapse_sigma": collapse_sigma}


def main():
    print("=" * 60)
    print("  DIRECTION 5: CKA ALPHA SCAN")
    print("  Gradient Coupling Sweep")
    print("=" * 60)

    alphas = np.linspace(0, 1, 11)
    results = []

    for alpha in alphas:
        r = train_with_alpha(alpha, T=4000, seed=42)
        results.append(r)
        print(f"  α={alpha:.1f}: CKA={r['cka']:.4f}  corr={r['corr']:+.4f}  "
              f"loss={r['avg_loss']:.6f}  collapse_σ={r['collapse_sigma']}")

    out_path = Path("results_final/direction5_alpha_scan.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out_path}")

    corrs = [r["corr"] for r in results]
    print(f"  corr range: [{min(corrs):+.4f}, {max(corrs):+.4f}]")


if __name__ == "__main__":
    main()
