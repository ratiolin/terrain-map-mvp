"""Direction: Environment Rule Rewriting — Symmetry Test.

Tests how different action combination rules (mean, max, min, alternating)
affect the coordination between independently trained agents.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import DriftingDoubleWellSchedule
from core_mvp_v3.models import PolicyNetwork


def combine_mean(a_A, a_B):
    return 0.5 * (a_A + a_B)


def combine_max(a_A, a_B):
    return max(float(a_A), float(a_B))


def combine_min(a_A, a_B):
    return min(float(a_A), float(a_B))


def combine_alt(a_A, a_B, t):
    return float(a_A) if t % 2 == 0 else float(a_B)


class MultiEnv:
    def __init__(self, base_env, combine_fn):
        self.env = base_env
        self.combine_fn = combine_fn
        self.t = 0

    def reset(self):
        self.t = 0
        return self.env.reset()

    def step(self, a_A, a_B):
        if self.combine_fn == combine_alt:
            action = self.combine_fn(a_A, a_B, self.t)
        else:
            action = self.combine_fn(a_A, a_B)
        self.t += 1
        return self.env.step(action)

    @property
    def state(self):
        return self.env.state


def load_policy(tag=""):
    suffix = f"_{tag}" if tag else ""
    net = PolicyNetwork(hidden_dim=32)
    path = Path(f"results_final/phase0_policy_net{suffix}.pt")
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def main():
    print("=" * 60)
    print("  ENV RULE SYMMETRY TEST")
    print("=" * 60)

    net_A = load_policy()
    net_B = load_policy("seed1")

    schedule = [
        (1000, (0.1, 0.3)),
        (1000, (1.0, 2.0)),
        (1000, (0.1, 0.3)),
        (1000, (1.0, 2.0)),
    ]
    base_env = DriftingDoubleWellSchedule(
        schedule=schedule, noise=0.05, state_clip=5.0,
        force_scale=0.1, action_scale=0.1,
    )

    rules = {
        "mean": combine_mean,
        "max": combine_max,
        "min": combine_min,
        "alt": combine_alt,
    }

    T = 4000
    results = {}

    for name, fn in rules.items():
        env = MultiEnv(base_env.clone(), fn)
        env.reset()
        a_A_seq, a_B_seq, risk_seq = [], [], []

        for _ in range(T):
            state = env.state.copy()
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a_A_t, _, _ = net_A(x)
                a_B_t, _, _ = net_B(x)

            a_A = float(a_A_t.item())
            a_B = float(a_B_t.item())
            next_state = env.step(a_A, a_B)

            a_A_seq.append(a_A)
            a_B_seq.append(a_B)
            risk_seq.append(abs(float(next_state[0])))

        a_A_arr = np.array(a_A_seq)
        a_B_arr = np.array(a_B_seq)
        corr = float(np.corrcoef(a_A_arr, a_B_arr)[0, 1])
        diff = float(np.mean(np.abs(a_A_arr - a_B_arr)))
        mean_risk = float(np.mean(risk_seq))

        results[name] = {"corr": corr, "diff": diff, "mean_risk": mean_risk}
        print(f"  {name:6s}: corr={corr:+.4f}  diff={diff:.4f}  risk={mean_risk:.4f}")

    out_path = Path("results_final/env_rules.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
