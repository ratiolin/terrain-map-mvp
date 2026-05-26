"""Direction 4c-robust: Anti-Correlation Stability Under Noise.

Tests whether the anti-correlated action pattern from co-trained agents
is robust to noise injection on agent B.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def load_cotrained_policy(tag):
    net = PolicyNetwork(hidden_dim=32)
    state_dict = torch.load(Path(f"results_final/direction4c_policy_{tag}.pt"),
                            map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def main():
    print("=" * 60)
    print("  DIRECTION 4c-robust: ANTI-CORRELATION STABILITY")
    print("=" * 60)

    net_A = load_cotrained_policy("A")
    net_B = load_cotrained_policy("B")

    schedule = [
        (1000, (0.1, 0.3)),
        (1000, (1.0, 2.0)),
        (1000, (0.1, 0.3)),
        (1000, (1.0, 2.0)),
    ]

    sigmas = np.linspace(0, 0.2, 12)
    T = 4000
    results = []

    print(f"  testing σ ∈ [0, 0.2], {len(sigmas)} steps, T={T}")

    for sigma in sigmas:
        env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                    state_clip=5.0, force_scale=0.1,
                                    action_scale=0.1, action_mix=0.5)
        env.reset()
        a_A_seq, a_B_seq = [], []

        for _ in range(T):
            state = env.state.copy()
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a_A_t, _, _ = net_A(x)
                a_B_t, _, _ = net_B(x)

            a_A = float(a_A_t.item())
            a_B = float(a_B_t.item())
            noise = np.random.normal(0, sigma)
            a_B_noisy = a_B + noise

            env.step(a_A, a_B_noisy)
            a_A_seq.append(a_A)
            a_B_seq.append(a_B_noisy)

        a_A_arr = np.array(a_A_seq)
        a_B_arr = np.array(a_B_seq)
        corr = float(np.corrcoef(a_A_arr, a_B_arr)[0, 1])
        diff = float(np.mean(np.abs(a_A_arr - a_B_arr)))

        results.append({"sigma": float(sigma), "corr": corr, "diff": diff})

        if abs(corr) < 0.2 and sigma > 0:
            print(f"  σ={sigma:.3f}: corr={corr:+.4f} diff={diff:.4f} ← broken")
        else:
            print(f"  σ={sigma:.3f}: corr={corr:+.4f} diff={diff:.4f}")

    retained = [r for r in results if r["corr"] < -0.8]
    collapse_sigma = retained[-1]["sigma"] if retained else None

    if collapse_sigma:
        print(f"\n  anti-correlation retained up to σ={collapse_sigma:.3f}")
    else:
        print(f"\n  anti-correlation collapsed immediately")

    out_path = Path("results_final/direction4c_robust.json")
    with open(out_path, "w") as f:
        json.dump({"sweep": results, "collapse_sigma": collapse_sigma}, f, indent=2)
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
