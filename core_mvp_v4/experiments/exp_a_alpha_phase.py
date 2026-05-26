"""Experiment A: P2 Reward-Semantic Phase Transition (Refined).

d=2, k=2. Uses best config from S0.
reward = alpha * C(s) + (1-alpha) * random_noise
Scans alpha in [0, 0.1, ..., 1.0], 8+ seeds per alpha.

Determines whether semantic encoding shows a phase transition or gradual degradation.
"""

import json
import numpy as np
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, train_with_signal_loss, collect_controllability_data, compute_jacobian
from core_mvp_v4.metrics import compute_k80, alignment


def run_exp_a_alpha_phase(n_seeds=8, d=2, k=2, hd=32,
                          n_episodes=3, episode_length=2000, lambda_signal=1.0):
    alpha_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}

    for alpha in alpha_list:
        r2_list = []
        align_gt_list = []
        k80_list = []

        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)

            train_with_signal_loss(
                model, env,
                num_episodes=n_episodes, episode_length=episode_length,
                lambda_signal=lambda_signal, seed=seed, alpha=alpha,
            )

            test_h, test_C = collect_controllability_data(
                model, env, n_samples=300,
            )
            probe = LinearRegression().fit(test_h, test_C)
            r2_list.append(float(probe.score(test_h, test_C)))

            states_after = []
            env.reset()
            for _ in range(100):
                states_after.append(env.get_state())
                a = model.act_numpy(env.get_state())
                env.step(a)
            J_mean = np.mean([compute_jacobian(model, s) for s in states_after[::3]], axis=0)
            _, S_mean, Vt_mean = np.linalg.svd(J_mean, full_matrices=False)
            V = Vt_mean.T

            k80_list.append(compute_k80(S_mean))
            U_true = np.eye(d)[:, :k]
            k_use = min(k, V.shape[1])
            align_gt_list.append(float(alignment(V[:, :k_use], U_true[:, :k_use], k=k_use)))

        results[f"{alpha}"] = {
            "alpha": alpha,
            "R2_mean": float(np.mean(r2_list)),
            "R2_std": float(np.std(r2_list)),
            "alignment_gt_mean": float(np.mean(align_gt_list)),
            "alignment_gt_std": float(np.std(align_gt_list)),
            "k80_mean": float(np.mean(k80_list)),
            "k80_std": float(np.std(k80_list)),
            "raw_R2": r2_list,
            "raw_align": align_gt_list,
        }

    phase = _analyze_phase(results, alpha_list)
    results["phase_analysis"] = phase
    return results


def _analyze_phase(results, alpha_list):
    r2_curve = [results[f"{a}"]["R2_mean"] for a in alpha_list]
    align_curve = [results[f"{a}"]["alignment_gt_mean"] for a in alpha_list]
    k80_curve = [results[f"{a}"]["k80_mean"] for a in alpha_list]

    r2_base = r2_curve[-1] if r2_curve[-1] > 0 else 1e-6
    r2_ratios = [r / r2_base for r in r2_curve]

    alpha_crit = None
    for i in range(1, len(alpha_list)):
        if r2_ratios[i] < 0.5 and r2_ratios[i - 1] >= 0.5:
            alpha_crit = alpha_list[i]
            break

    if alpha_crit is None:
        r2_drop = (r2_curve[0] - r2_curve[-1]) / max(r2_base, 1e-6)
        if abs(r2_drop) < 0.2:
            mechanism = "GRADUAL: no significant R2 drop across alpha. Semantic encoding is robust to reward noise."
        else:
            mechanism = "CONTINUOUS: steady R2 decline without sharp cliff. Semantic encoding degrades smoothly."
    else:
        mechanism = (
            f"PHASE_TRANSITION at alpha_crit={alpha_crit}. "
            f"R2 collapse onset: semantic encoding requires minimum reward correlation threshold."
        )

    return {
        "alpha_crit": alpha_crit,
        "R2_curve": r2_curve,
        "alignment_curve": align_curve,
        "k80_curve": k80_curve,
        "mechanism": mechanism,
    }


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_exp_a_alpha_phase(n_seeds=8)
    with open("core_mvp_v4/results/p2_alpha_phase.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    pa = r["phase_analysis"]
    print(f"Experiment A: alpha_crit={pa['alpha_crit']}")
    print(f"  {pa['mechanism']}")
