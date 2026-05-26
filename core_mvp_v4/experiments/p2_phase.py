"""P2-Continuous: Reward-Semantic Phase Transition (3C-continuized).

Precondition: S0 passed. Uses d=2, k=2. Fixed lambda from S0.

reward = alpha * C(s) + (1-alpha) * random_noise
alpha ∈ [0.0, 0.1, 0.2, ..., 1.0]

Goal: phase transition curve of structure vs reward correlation strength.
"""

import json
import numpy as np
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, train_with_signal_loss, collect_controllability_data, compute_jacobian
from core_mvp_v4.metrics import compute_k80, R2_probe, alignment


def run_p2_phase(n_seeds=8, d=2, k=2, sigma_obs=0.0,
                 hd=16, n_episodes=3, episode_length=2000, lambda_signal=0.5):
    alpha_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}

    for alpha in alpha_list:
        a_results = {
            "R2_list": [],
            "alignment_gt_list": [],
            "k80_list": [],
        }

        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)

            train_with_signal_loss(
                model, env,
                num_episodes=n_episodes, episode_length=episode_length,
                lambda_signal=lambda_signal, seed=seed, alpha=alpha,
            )

            env.reset()
            test_h, test_C = collect_controllability_data(
                model, env, n_samples=500, sigma_obs=sigma_obs,
            )
            probe = LinearRegression().fit(test_h, test_C)
            R2_val = float(probe.score(test_h, test_C))
            a_results["R2_list"].append(R2_val)

            states = [env.get_state()]
            for _ in range(100):
                s = env.get_state()
                a = model.act_numpy(s)
                ns, _, _, _ = env.step(a)
                states.append(ns)
            J_mean = np.mean([compute_jacobian(model, s) for s in states[::3]], axis=0)
            _, S_mean, Vt_mean = np.linalg.svd(J_mean, full_matrices=False)
            V = Vt_mean.T
            k80_val = compute_k80(S_mean)
            a_results["k80_list"].append(k80_val)

            U_true = np.eye(d)[:, :k]
            k_use = min(k, V.shape[1])
            align_gt = alignment(V[:, :k_use], U_true[:, :k_use], k=k_use)
            a_results["alignment_gt_list"].append(float(align_gt))

        for key in ["R2_list", "alignment_gt_list", "k80_list"]:
            arr = np.array(a_results[key])
            a_results[f"{key}_mean"] = float(np.mean(arr))
            a_results[f"{key}_std"] = float(np.std(arr))

        a_results["summary"] = {
            "alpha": alpha,
            "R2": f"{a_results['R2_list_mean']:.4f} ± {a_results['R2_list_std']:.4f}",
            "alignment_gt": f"{a_results['alignment_gt_list_mean']:.4f} ± {a_results['alignment_gt_list_std']:.4f}",
            "k80": f"{a_results['k80_list_mean']:.2f} ± {a_results['k80_list_std']:.2f}",
        }
        results[f"alpha_{alpha}"] = a_results

    results["phase_analysis"] = _analyze_phase(results, alpha_list)
    return results


def _analyze_phase(results, alpha_list):
    r2_means = [results[f"alpha_{a}"]["R2_list_mean"] for a in alpha_list]
    align_means = [results[f"alpha_{a}"]["alignment_gt_list_mean"] for a in alpha_list]
    k80_means = [results[f"alpha_{a}"]["k80_list_mean"] for a in alpha_list]

    alpha_crit = None
    for i in range(1, len(alpha_list)):
        if r2_means[i] < r2_means[0] * 0.5 and align_means[i] < align_means[0] * 0.5:
            alpha_crit = alpha_list[i]
            break

    if alpha_crit is not None:
        conclusion = (
            f"Phase transition at alpha_crit={alpha_crit}. "
            "Subspace semantics derive from reward signal controllability component; "
            "minimum correlation threshold exists."
        )
    else:
        r2_slope = (r2_means[-1] - r2_means[0]) / alpha_list[-1] if alpha_list[-1] != 0 else 0
        if abs(r2_slope) < 0.1:
            conclusion = "No clear phase transition. Semantic encoding is gradual."
        else:
            conclusion = "Continuous degradation: no critical threshold detected."

    return {
        "alpha_list": alpha_list,
        "R2_curve": r2_means,
        "alignment_gt_curve": align_means,
        "k80_curve": k80_means,
        "alpha_crit": alpha_crit,
        "conclusion": conclusion,
    }


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_p2_phase(n_seeds=8)
    with open("core_mvp_v4/results/p2_reward_semantic_phase.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    pa = r.get("phase_analysis", {})
    print(f"P2-Continuous: alpha_crit={pa.get('alpha_crit')}")
    print(f"  {pa.get('conclusion')}")
