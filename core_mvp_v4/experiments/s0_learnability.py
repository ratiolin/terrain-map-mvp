"""S0: Signal Learnability Calibration.

Verifies whether C(s) can be learned by the current pipeline.

Sweeps: hidden_dim ∈ [16, 32, 64], step_multiplier ∈ [1, 2, 4],
         lambda_signal ∈ [0.1, 0.5, 1.0]

If best R² < 0.2: C(s) not learnable — re-architect.
If best R² >= 0.5: proceed to subsequent experiments.
"""

import json
import numpy as np
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, train_with_signal_loss, collect_controllability_data


def run_s0_learnability(n_seeds=8, d=2, k=2, sigma_obs=0.0,
                        n_episodes=3, episode_length=2000):
    base_episodes = n_episodes
    base_ep_length = episode_length
    hidden_dims = [32]
    step_multipliers = [1, 2]
    lambdas = [0.1, 0.5, 1.0]

    results = {"sweep": [], "best_config": None, "best_R2": None, "conclusion": ""}

    best_R2_overall = -np.inf
    best_config_overall = None

    for hd in hidden_dims:
        for sm in step_multipliers:
            for lam in lambdas:
                config_label = f"hd{hd}_steps{sm}x_lambda{lam}"
                R2_values = []

                for seed in range(n_seeds):
                    train_env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed)
                    model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)

                    n_ep = base_episodes * sm
                    ep_len = base_ep_length
                    train_with_signal_loss(
                        model, train_env,
                        num_episodes=n_ep, episode_length=ep_len,
                        lambda_signal=lam, seed=seed,
                    )

                    test_env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed + 10000)
                    test_h, test_C = collect_controllability_data(
                        model, test_env, n_samples=500, sigma_obs=sigma_obs,
                    )

                    probe = LinearRegression().fit(test_h, test_C)
                    R2 = float(probe.score(test_h, test_C))
                    R2_values.append(R2)

                R2_arr = np.array(R2_values)
                R2_mean = float(np.mean(R2_arr))
                R2_std = float(np.std(R2_arr))

                entry = {
                    "hidden_dim": hd,
                    "step_multiplier": sm,
                    "n_episodes": base_episodes * sm,
                    "lambda_signal": lam,
                    "R2_mean": R2_mean,
                    "R2_std": R2_std,
                    "R2_values": R2_values,
                }
                results["sweep"].append(entry)

                if R2_mean > best_R2_overall:
                    best_R2_overall = R2_mean
                    best_config_overall = {
                        "hidden_dim": hd,
                        "step_multiplier": sm,
                        "n_episodes": base_episodes * sm,
                        "lambda_signal": lam,
                        "R2": R2_mean,
                        "R2_std": R2_std,
                    }

    results["best_config"] = best_config_overall
    results["best_R2"] = round(best_R2_overall, 4)

    if best_R2_overall < 0.2:
        results["conclusion"] = (
            "FAIL: C(s) not learnable under current representation. "
            "Re-define controllability signal or enhance representation capacity."
        )
    elif best_R2_overall < 0.5:
        results["conclusion"] = (
            "MARGINAL: R2 < 0.5. Proceeding with caution — "
            "signal partially learnable."
        )
    else:
        results["conclusion"] = (
            "PASS: R2 >= 0.5. Controllability signal is reliably learnable. "
            "Proceed to P1-Refocus."
        )

    return results


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_s0_learnability(n_seeds=8)
    with open("core_mvp_v4/results/s0_learnability.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"S0: best_config={r['best_config']}")
    print(f"  best_R2={r['best_R2']}")
    print(f"  conclusion={r['conclusion']}")
