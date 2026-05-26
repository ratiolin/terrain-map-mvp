"""Step 1: P2 High-dimensional Alpha Scan (d=8).

Precondition: S0 passed. d=8, hidden_dim=64 (sufficient capacity), k=2.
reward = alpha * C(s) + (1-alpha) * random_noise.
Scans alpha in [0, 0.1, ..., 1.0], 8+ seeds each.
Determines if alpha_crit exists triggering semantic collapse at d=8.
"""

import json
import numpy as np
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, train_with_signal_loss, collect_controllability_data, compute_jacobian
from core_mvp_v4.metrics import compute_k80, alignment


def run_step1_d8_alpha_scan(n_seeds=8, d=8, k=2, hd=64,
                            n_episodes=3, episode_length=2000, lambda_signal=1.0):
    alpha_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}

    for alpha in alpha_list:
        r2_list = []; align_list = []; k80_list = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)

            train_with_signal_loss(model, env, num_episodes=n_episodes,
                                   episode_length=episode_length,
                                   lambda_signal=lambda_signal, seed=seed, alpha=alpha)

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

        results[f"{alpha}"] = {
            "alpha": alpha,
            "R2_mean": float(np.mean(r2_list)), "R2_std": float(np.std(r2_list)),
            "alignment_gt_mean": float(np.mean(align_list)), "alignment_gt_std": float(np.std(align_list)),
            "k80_mean": float(np.mean(k80_list)), "k80_std": float(np.std(k80_list)),
        }

    phase = _analyze(results, alpha_list)
    results["phase_analysis"] = phase
    return results


def _analyze(results, alpha_list):
    r2_curve = [results[f"{a}"]["R2_mean"] for a in alpha_list]
    align_curve = [results[f"{a}"]["alignment_gt_mean"] for a in alpha_list]
    k80_curve = [results[f"{a}"]["k80_mean"] for a in alpha_list]

    r2_max = max(r2_curve)
    r2_norm = [r / r2_max for r in r2_curve]

    alpha_crit = None
    for i in range(1, len(alpha_list)):
        if r2_norm[i] < 0.5 and r2_norm[i - 1] >= 0.5:
            alpha_crit = alpha_list[i]
            break

    if alpha_crit is not None:
        mechanism = f"PHASE_TRANSITION at alpha_crit={alpha_crit}: R2 collapses at this reward correlation threshold."
    else:
        r2_drop_pct = abs(r2_curve[0] - r2_curve[-1]) / max(r2_max, 1e-6)
        if r2_drop_pct < 0.15:
            mechanism = "GRADUAL: no significant R2 drop. Semantics robust to reward noise even at d=8."
        else:
            mechanism = "CONTINUOUS: steady R2 decline without sharp cliff. Gradual semantic degradation."

    return {
        "alpha_crit": alpha_crit,
        "R2_curve": r2_curve, "alignment_curve": align_curve, "k80_curve": k80_curve,
        "mechanism": mechanism,
    }


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_step1_d8_alpha_scan(n_seeds=8)
    with open("core_mvp_v4/results/p2_d8_alpha_scan.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    pa = r["phase_analysis"]
    print(f"Step 1: alpha_crit={pa['alpha_crit']}")
    print(f"  {pa['mechanism']}")
