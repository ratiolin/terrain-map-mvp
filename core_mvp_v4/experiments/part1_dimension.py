import json
import numpy as np
from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, train_model, compute_jacobian
from core_mvp_v4.metrics import (
    compute_k80, effective_rank, R2_probe, alignment_time, alignment,
    compute_silhouette_scores, analyze_jacobian,
)


def run_part1_dimension_pressure(n_seeds=8, k_base=2,
                                 n_episodes=3, episode_length=2000,
                                 hidden_dim=32, coupling=0.05):
    """Part 1: Dimension pressure.

    Scans total state dimension d, measuring how the controllability
    subspace degrades as irrelevant dimensions are added.

    Uses ground truth: U_true = I[:, :k] — the first k basis vectors.
    """
    d_list = [2, 4, 8, 16, 32]
    results = {}

    for d in d_list:
        k = min(k_base, d - 1) if d > k_base else d
        if k <= 0:
            k = 1

        d_results = {
            "k80_list": [],
            "R2_list": [],
            "alignment_time_mean_list": [],
            "silhouette_list": [],
            "alignment_gt_list": [],
            "effective_rank_list": [],
        }

        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, coupling=coupling, seed=seed)
            model = V4Model(state_dim=d, hidden_dim=hidden_dim, action_dim=k)

            train_model(model, env, num_episodes=n_episodes,
                        episode_length=episode_length, seed=seed)

            states = []
            hidden_vals = []
            risk_vals = []
            env.reset()
            for _ in range(500):
                s = env.get_state()
                states.append(s)
                h = model.f_numpy(s)
                hidden_vals.append(h)
                a = model.act_numpy(s)
                ns, risk, _, _ = env.step(a)
                risk_vals.append(risk)

            J_results = analyze_jacobian(model, states)

            k80_vals = [r["k80"] for r in J_results]
            d_results["k80_list"].append(float(np.mean(k80_vals)))

            er_vals = [r["effective_rank_val"] for r in J_results]
            d_results["effective_rank_list"].append(float(np.mean(er_vals)))

            h_all = np.array(hidden_vals)
            r_all = np.array(risk_vals).reshape(-1, 1)
            r2 = R2_probe(h_all, r_all)
            d_results["R2_list"].append(r2)

            a_t = alignment_time(model, env, states, delta=100)
            if a_t:
                d_results["alignment_time_mean_list"].append(float(np.mean(a_t)))

            U_true = np.eye(d)[:, :k]
            J_mean = None
            for s_sample in states[::10]:
                J = compute_jacobian(model, s_sample)
                if J_mean is None:
                    J_mean = J
                else:
                    J_mean += J
            J_mean /= (len(states[::10]))

            _, S_mean, Vt_mean = np.linalg.svd(J_mean, full_matrices=False)
            k_align = min(k, S_mean.shape[0])
            V_mean = Vt_mean.T[:, :k_align]

            if V_mean.shape[1] >= k:
                align_gt = alignment(V_mean, U_true, k=k)
            else:
                k_adj = min(V_mean.shape[1], k)
                align_gt = alignment(V_mean[:, :k_adj], U_true[:, :k_adj], k=k_adj)
            d_results["alignment_gt_list"].append(float(align_gt))

            silhouette = compute_silhouette_scores(hidden_vals, risk_vals)
            d_results["silhouette_list"].append(float(silhouette))

        for key in ["k80_list", "R2_list", "alignment_time_mean_list",
                     "silhouette_list", "alignment_gt_list", "effective_rank_list"]:
            arr = np.array(d_results[key]) if d_results[key] else np.array([0.0])
            d_results[f"{key}_mean"] = float(np.mean(arr))
            d_results[f"{key}_std"] = float(np.std(arr))

        if d_results["k80_list"]:
            d_results["summary"] = {
                "d": d,
                "k": k,
                "k80": f"{d_results['k80_list_mean']:.2f} ± {d_results['k80_list_std']:.2f}",
                "R2": f"{d_results['R2_list_mean']:.4f} ± {d_results['R2_list_std']:.4f}",
                "alignment_time": f"{d_results['alignment_time_mean_list_mean']:.4f} ± {d_results['alignment_time_mean_list_std']:.4f}",
                "silhouette": f"{d_results['silhouette_list_mean']:.4f} ± {d_results['silhouette_list_std']:.4f}",
                "alignment_gt": f"{d_results['alignment_gt_list_mean']:.4f} ± {d_results['alignment_gt_list_std']:.4f}",
            }

        results[str(d)] = d_results

    return results


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_part1_dimension_pressure(n_seeds=8)
    with open("core_mvp_v4/results/part1_dimension_pressure.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("Part 1 complete.")
    for dk, data in r.items():
        s = data.get("summary", {})
        if s:
            print(f"  d={s.get('d')}: k80={s.get('k80')}, R2={s.get('R2')}, "
                  f"silhouette={s.get('silhouette')}, align_gt={s.get('alignment_gt')}")
