"""Direction 4d: Stability Threshold.

Gradually degrades agent B with increasing Gaussian noise and measures
the coordination collapse point — the sigma where action correlation
drops below threshold.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def load_policy(tag=""):
    suffix = f"_{tag}" if tag else ""
    net = PolicyNetwork(hidden_dim=32)
    state_dict = torch.load(Path(f"results_final/phase0_policy_net{suffix}.pt"),
                            map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def rollout_with_noise(net_A, net_B, sigma, T=4000, seed=0):
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
    env.reset()
    a_A_list, a_B_list = [], []
    for _ in range(T):
        state = env.state.copy()
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            as_A, _, _ = net_A(x)
            as_B, _, _ = net_B(x)
        a_A = float(as_A.item())
        a_B = float(as_B.item()) + np.random.normal(0, sigma)
        env.step(a_A, a_B)
        a_A_list.append(a_A)
        a_B_list.append(a_B)
    return np.array(a_A_list), np.array(a_B_list)


def main():
    print("=" * 60)
    print("  DIRECTION 4d: STABILITY THRESHOLD")
    print("=" * 60)

    print("\n--- loading agents ---")
    net_A = load_policy()
    net_B = load_policy("seed1")

    sigmas = np.linspace(0, 0.2, 21)
    print(f"  testing σ ∈ [0, 0.2] in {len(sigmas)} steps")

    sweep = []
    collapse_sigma = None
    for sigma in sigmas:
        a_A, a_B = rollout_with_noise(net_A, net_B, sigma, T=4000, seed=42)
        corr = float(np.corrcoef(a_A, a_B)[0, 1])
        diff = float(np.mean(np.abs(a_A - a_B)))
        sweep.append({"sigma": float(sigma), "corr": corr, "diff": diff})

        if collapse_sigma is None and corr < 0.2:
            collapse_sigma = float(sigma)
            print(f"  σ={sigma:.3f}: corr={corr:.4f} diff={diff:.4f} ← COLLAPSE")
        elif abs(sigma - 0.0) < 1e-10 or abs(sigma - 0.1) < 1e-10 or abs(sigma - 0.2) < 1e-10:
            print(f"  σ={sigma:.3f}: corr={corr:.4f} diff={diff:.4f}")

    if collapse_sigma is None:
        collapse_sigma = 0.2
        print(f"  no collapse below σ=0.2")

    print(f"\n  collapse threshold: σ* = {collapse_sigma:.3f}")

    results = {"sweep": sweep, "collapse_sigma": collapse_sigma}

    out_path = Path("results_final/direction4d_stability.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
