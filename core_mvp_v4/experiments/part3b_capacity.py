import json
import numpy as np
from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, train_model, compute_jacobian
from core_mvp_v4.metrics import (
    compute_k80, effective_rank, R2_probe, alignment_time, rank_J,
    analyze_jacobian,
)


def run_part3b_capacity(n_seeds=8, d=16, k=2,
                        n_episodes=3, episode_length=2000):
    """Part 3B: Capacity boundary.

    Varies hidden_dim to find the capacity threshold where
    alignment or R2 collapses.
    """
    hidden_dim_list = [4, 8, 16, 32, 64]
    results = {}

    for hd in hidden_dim_list:
        hd_results = {
            "k80_list": [],
            "R2_list": [],
            "alignment_time_mean_list": [],
            "rank_J_list": [],
            "effective_rank_list": [],
        }

        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)

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
            hd_results["k80_list"].append(float(np.mean(k80_vals)))

            er_vals = [r["effective_rank_val"] for r in J_results]
            hd_results["effective_rank_list"].append(float(np.mean(er_vals)))

            rank_vals = [r["rank_J_val"] for r in J_results]
            hd_results["rank_J_list"].append(float(np.mean(rank_vals)))

            h_all = np.array(hidden_vals)
            r_all = np.array(risk_vals).reshape(-1, 1)
            r2 = R2_probe(h_all, r_all)
            hd_results["R2_list"].append(r2)

            a_t = alignment_time(model, env, states, delta=100)
            if a_t:
                hd_results["alignment_time_mean_list"].append(float(np.mean(a_t)))

        for key in ["k80_list", "R2_list", "alignment_time_mean_list",
                     "rank_J_list", "effective_rank_list"]:
            arr = np.array(hd_results[key]) if hd_results[key] else np.array([0.0])
            hd_results[f"{key}_mean"] = float(np.mean(arr))
            hd_results[f"{key}_std"] = float(np.std(arr))

        if hd_results["k80_list"]:
            hd_results["summary"] = {
                "hidden_dim": hd,
                "k80": f"{hd_results['k80_list_mean']:.2f} ± {hd_results['k80_list_std']:.2f}",
                "R2": f"{hd_results['R2_list_mean']:.4f} ± {hd_results['R2_list_std']:.4f}",
                "alignment_time": f"{hd_results['alignment_time_mean_list_mean']:.4f} ± {hd_results['alignment_time_mean_list_std']:.4f}",
                "rank_J": f"{hd_results['rank_J_list_mean']:.2f} ± {hd_results['rank_J_list_std']:.2f}",
            }

        results[str(hd)] = hd_results

    hidden_dim_crit = _find_capacity_threshold(results)
    results["hidden_dim_critical"] = hidden_dim_crit

    return results


def _find_capacity_threshold(results):
    dim_sorted = sorted(
        [k for k in results.keys() if k != "hidden_dim_critical"],
        key=lambda x: float(x)
    )
    if len(dim_sorted) < 2:
        return {"hidden_dim_crit": None, "reason": "Insufficient data"}

    alignment_means = []
    r2_means = []
    for k in dim_sorted:
        alignment_means.append(results[k].get("alignment_time_mean_list_mean", 0))
        r2_means.append(results[k].get("R2_list_mean", 0))

    base_align = alignment_means[0]
    base_r2 = r2_means[0]

    hidden_dim_crit = None
    for i, k in enumerate(dim_sorted):
        if i == 0:
            continue
        if (alignment_means[i] < base_align * 0.5) or (r2_means[i] < base_r2 * 0.5):
            hidden_dim_crit = int(float(k))
            break

    return {
        "hidden_dim_crit": hidden_dim_crit,
        "alignment_means": alignment_means,
        "r2_means": r2_means,
    }


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_part3b_capacity(n_seeds=8)
    with open("core_mvp_v4/results/part3b_capacity_boundary.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"Part 3B complete. hidden_dim_crit = {r.get('hidden_dim_critical', {}).get('hidden_dim_crit')}")
