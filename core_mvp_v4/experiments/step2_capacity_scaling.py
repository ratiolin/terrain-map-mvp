"""Step 2: Capacity Scaling Law.

Fixed d=8, k=2. Sweeps hidden_dim/d ratio.
Ratios: [2, 4, 6, 8, 10, 12] -> hd ∈ [16, 32, 48, 64, 80, 96].
Finds minimum ratio for alignment_gt > 0.9.
"""

import json
import numpy as np
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, train_with_signal_loss, collect_controllability_data, compute_jacobian
from core_mvp_v4.metrics import compute_k80, alignment


def run_step2_capacity_scaling(n_seeds=8, d=8, k=2,
                               n_episodes=3, episode_length=2000, lambda_signal=1.0):
    ratios = [2, 4, 6, 8, 10, 12]
    hidden_dims = [int(d * r) for r in ratios]
    results = {}

    for r, hd in zip(ratios, hidden_dims):
        r2_list = []; align_list = []; k80_list = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)

            train_with_signal_loss(model, env, num_episodes=n_episodes,
                                   episode_length=episode_length,
                                   lambda_signal=lambda_signal, seed=seed, alpha=1.0)

            test_h, test_C = collect_controllability_data(model, env, n_samples=300)
            probe = LinearRegression().fit(test_h, test_C)
            r2_list.append(float(probe.score(test_h, test_C)))

            env.reset()
            jac_states = [env.get_state()]
            for _ in range(100):
                a = model.act_numpy(env.get_state())
                env.step(a)
                jac_states.append(env.get_state())
            J_mean = np.mean([compute_jacobian(model, s) for s in jac_states[::3]], axis=0)
            _, S_mean, Vt_mean = np.linalg.svd(J_mean, full_matrices=False)
            V = Vt_mean.T

            k80_list.append(compute_k80(S_mean))
            U_true = np.eye(d)[:, :k]
            k_use = min(k, V.shape[1])
            align_list.append(float(alignment(V[:, :k_use], U_true[:, :k_use], k=k_use)))

        results[str(hd)] = {
            "hidden_dim": hd, "ratio": r,
            "R2_mean": float(np.mean(r2_list)), "R2_std": float(np.std(r2_list)),
            "alignment_gt_mean": float(np.mean(align_list)), "alignment_gt_std": float(np.std(align_list)),
            "k80_mean": float(np.mean(k80_list)), "k80_std": float(np.std(k80_list)),
        }

    scaling = _analyze_scaling(results)
    results["scaling_analysis"] = scaling
    return results


def _analyze_scaling(results):
    hd_sorted = sorted([int(k) for k in results.keys() if k.isdigit()], key=int)
    aligns = [results[str(h)]["alignment_gt_mean"] for h in hd_sorted]
    r2s = [results[str(h)]["R2_mean"] for h in hd_sorted]
    ratios = [results[str(h)]["ratio"] for h in hd_sorted]

    threshold_ratio = None
    for i, align in enumerate(aligns):
        if align > 0.9:
            threshold_ratio = ratios[i]
            break

    if threshold_ratio is None:
        conclusion = "No ratio achieves alignment>0.9 within tested range. Need larger ratios."
    else:
        conclusion = (f"Minimum ratio for align>0.9: hidden_dim/d = {threshold_ratio} "
                      f"(hidden_dim={int(threshold_ratio * 8)})")

    return {
        "ratios": ratios, "hidden_dims": hd_sorted,
        "alignment_curve": aligns, "R2_curve": r2s,
        "threshold_ratio": threshold_ratio,
        "conclusion": conclusion,
    }


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_step2_capacity_scaling(n_seeds=8)
    with open("core_mvp_v4/results/capacity_scaling.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    sa = r["scaling_analysis"]
    print(f"Step 2: threshold_ratio={sa['threshold_ratio']}")
    print(f"  {sa['conclusion']}")
