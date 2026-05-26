"""Direction: Frozen Encoder Control Experiment.

Freezes the backbone (encoder) of both pre-trained agents and trains
only the policy heads in a shared environment. Tests whether the
anti-correlated behavior emerges from head plasticity alone.
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
    hsic = np.sum(K * L)
    norm = np.linalg.norm(K) * np.linalg.norm(L)
    return float(hsic / (norm + 1e-8))


def main():
    print("=" * 60)
    print("  FROZEN ENCODER CONTROL EXPERIMENT")
    print("=" * 60)

    net_A = PolicyNetwork(hidden_dim=32)
    net_B = PolicyNetwork(hidden_dim=32)
    net_A.load_state_dict(torch.load(
        Path("results_final/phase0_policy_net.pt"), map_location="cpu", weights_only=True))
    net_B.load_state_dict(torch.load(
        Path("results_final/phase0_policy_net_seed1.pt"), map_location="cpu", weights_only=True))

    for p in net_A.backbone.parameters():
        p.requires_grad = False
    for p in net_B.backbone.parameters():
        p.requires_grad = False

    trainable = (
        list(net_A.head_shape.parameters()) + list(net_A.head_adapt.parameters())
        + list(net_B.head_shape.parameters()) + list(net_B.head_adapt.parameters())
    )
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    schedule = [
        (2000, (0.1, 0.3)),
        (2000, (1.0, 2.0)),
        (2000, (0.1, 0.3)),
        (2000, (1.0, 2.0)),
    ]
    env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)

    N = 8000
    log_every = 200
    metrics = []
    h_buf_A, h_buf_B = [], []
    a_buf_A, a_buf_B = [], []

    env.reset()
    print(f"  training {N} steps, heads only, logging every {log_every}")

    for step in range(N):
        state = env.state.copy()
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        a_shape_A, a_adapt_A, h_A = net_A(x)
        a_shape_B, a_adapt_B, h_B = net_B(x)

        a_A_val = float(a_shape_A.item())
        a_B_val = float(a_shape_B.item())
        next_state = env.step(a_A_val, a_B_val)

        risk = abs(float(next_state[0]))
        loss = torch.tensor(risk, dtype=torch.float32, requires_grad=True)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        h_buf_A.append(h_A.detach().cpu().numpy().flatten())
        h_buf_B.append(h_B.detach().cpu().numpy().flatten())
        a_buf_A.append(a_A_val)
        a_buf_B.append(a_B_val)

        if (step + 1) % log_every == 0:
            H_A = np.array(h_buf_A)
            H_B = np.array(h_buf_B)
            cka = linear_CKA(H_A, H_B)
            a_arr = np.array(a_buf_A)
            b_arr = np.array(a_buf_B)
            corr = float(np.corrcoef(a_arr, b_arr)[0, 1])
            metrics.append({"step": step + 1, "cka": round(cka, 4),
                            "corr": round(corr, 4), "risk": round(float(risk), 4)})
            h_buf_A, h_buf_B = [], []
            a_buf_A, a_buf_B = [], []

            if (step + 1) % 2000 == 0:
                print(f"  step {step+1:5d}: CKA={cka:.4f} corr={corr:+.4f} risk={risk:.4f}")

    early_corr = np.mean([m["corr"] for m in metrics[:len(metrics)//4]])
    late_corr = np.mean([m["corr"] for m in metrics[3*len(metrics)//4:]])
    early_cka = np.mean([m["cka"] for m in metrics[:len(metrics)//4]])
    late_cka = np.mean([m["cka"] for m in metrics[3*len(metrics)//4:]])

    print(f"\n  CKA: {early_cka:.4f} → {late_cka:.4f}")
    print(f"  corr: {early_corr:+.4f} → {late_corr:+.4f}")

    results = {
        "metrics": metrics,
        "summary": {
            "early_cka": float(early_cka), "late_cka": float(late_cka),
            "early_corr": float(early_corr), "late_corr": float(late_corr),
            "trend_corr": "converging" if abs(late_corr) > abs(early_corr) else "diverging",
        },
    }

    out_path = Path("results_final/frozen_encoder.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
