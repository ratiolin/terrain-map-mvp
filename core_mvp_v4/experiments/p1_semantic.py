"""P1-Refocus: Semantic Degradation Boundary (Part 1 revision).

Precondition: S0 found any config with R² >= 0.3. Fix that config.

Sweeps: d ∈ [4, 8, 16], k=2, sigma_obs=0.

Goal: find d where alignment_gt drops significantly — the semantic boundary.
"""

import json
import numpy as np
from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, train_with_signal_loss, collect_controllability_data, compute_jacobian
from core_mvp_v4.metrics import compute_k80, R2_probe, alignment
from sklearn.linear_model import LinearRegression


def run_p1_semantic(n_seeds=8, k=2, sigma_obs=0.0, d=4,
                    hd=16, n_episodes=3, episode_length=2000, lambda_signal=0.5):
    d_list = [4, 8, 16]
    results = {}

    for d in d_list:
        if d < k:
            continue
        d_results = {
            "alignment_gt_list": [],
            "R2_list": [],
            "k80_list": [],
        }

        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)

            train_with_signal_loss(
                model, env,
                num_episodes=n_episodes, episode_length=episode_length,
                lambda_signal=lambda_signal, seed=seed,
            )

            env.reset()
            states = []
            hidden_vals = []
            C_vals = []
            for _ in range(500):
                s = env.get_state()
                if sigma_obs > 0:
                    s_obs = s + np.random.normal(0, sigma_obs, size=s.shape)
                else:
                    s_obs = s
                states.append(s_obs)
                h = model.f_numpy(s_obs)
                hidden_vals.append(h)

                a = model.act_numpy(s_obs)
                s_next_actual = env.forward_static(s, a)
                s_next_zero = env.forward_static(s, np.zeros(env.k))
                delta = s_next_actual[:env.k] - s_next_zero[:env.k]
                C_val = float(np.linalg.norm(delta))
                C_vals.append(C_val)

                ns, _, _, _ = env.step(a)
                s = ns

            h_all = np.array(hidden_vals)
            C_all = np.array(C_vals).reshape(-1, 1)
            probe = LinearRegression().fit(h_all, C_all)
            R2_val = float(probe.score(h_all, C_all))
            d_results["R2_list"].append(R2_val)

            J_mean = np.mean([
                compute_jacobian(model, s) for s in states[::5]
            ], axis=0)
            _, S_mean, Vt_mean = np.linalg.svd(J_mean, full_matrices=False)
            V = Vt_mean.T

            k80_val = compute_k80(S_mean)
            d_results["k80_list"].append(k80_val)

            U_true = np.eye(d)[:, :k]
            k_use = min(k, V.shape[1])
            align_gt = alignment(V[:, :k_use], U_true[:, :k_use], k=k_use)
            d_results["alignment_gt_list"].append(float(align_gt))

        for key in ["alignment_gt_list", "R2_list", "k80_list"]:
            arr = np.array(d_results[key])
            d_results[f"{key}_mean"] = float(np.mean(arr))
            d_results[f"{key}_std"] = float(np.std(arr))

        d_results["summary"] = {
            "d": d,
            "k": k,
            "alignment_gt": f"{d_results['alignment_gt_list_mean']:.4f} ± {d_results['alignment_gt_list_std']:.4f}",
            "R2": f"{d_results['R2_list_mean']:.4f} ± {d_results['R2_list_std']:.4f}",
            "k80": f"{d_results['k80_list_mean']:.2f} ± {d_results['k80_list_std']:.2f}",
        }
        results[str(d)] = d_results

    results["boundary_analysis"] = _analyze_boundary(results)
    return results


def _analyze_boundary(results):
    d_keys = sorted([int(k) for k in results.keys() if k.isdigit()])
    align_means = [results[str(d)]["alignment_gt_list_mean"] for d in d_keys]
    r2_means = [results[str(d)]["R2_list_mean"] for d in d_keys]

    boundary = None
    for i in range(1, len(d_keys)):
        if align_means[i] < 0.6:
            prev_d = d_keys[i - 1]
            curr_d = d_keys[i]
            boundary = f"suggested boundary between d={prev_d} and d={curr_d} "
            boundary += f"(alignment dropped from {align_means[i-1]:.3f} to {align_means[i]:.3f})"
            break

    if boundary is None:
        all_above = all(a > 0.9 for a in align_means)
        if all_above:
            boundary = "No degradation: all align_gt > 0.9. Dimension alone does not cause semantic degradation. Need noise coupling."
        else:
            boundary = "Partial degradation seen; no clear boundary identified."

    return {"d_list": d_keys, "alignment_means": align_means, "R2_means": r2_means, "boundary": boundary}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_p1_semantic(n_seeds=8)
    with open("core_mvp_v4/results/p1_semantic_degradation.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    ba = r.get("boundary_analysis", {})
    print(f"P1-Refocus: {ba.get('boundary')}")
    for dk, data in r.items():
        s = data.get("summary")
        if s:
            print(f"  d={s['d']}: align_gt={s['alignment_gt']}, R2={s['R2']}")
