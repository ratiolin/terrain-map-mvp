import json
import numpy as np
from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, train_with_panic, compute_jacobian
from core_mvp_v4.metrics import (
    compute_k80, effective_rank, R2_probe, alignment_time,
    cross_seed_consistency, analyze_jacobian, failure_criteria,
)


def run_part3c_mechanism(n_seeds=8, d=16, k=2, hidden_dim=32,
                         n_episodes=3, episode_length=2000):
    """Part 3C: Mechanism destruction experiments.

    Tests three destruction modes:
    3C(1) Random panic - panic signal replaced with random noise
    3C(2) No rollout - panic set to zero
    3C(3) Random reward - reward signal replaced with random noise
    """
    modes = ["baseline", "random", "none", "random_reward"]
    mode_labels = {
        "baseline": "Normal (baseline)",
        "random": "3C(1) Random panic",
        "none": "3C(2) No rollout (panic=0)",
        "random_reward": "3C(3) Random reward",
    }

    results = {}

    for mode in modes:
        mode_results = {
            "k80_list": [],
            "R2_list": [],
            "effective_rank_list": [],
            "alignment_time_mean_list": [],
            "cross_seed_consistency_list": [],
            "failure_criteria_list": [],
            "k80_values": [],
        }

        models_list = []
        all_hidden_states = []
        all_risks = []
        all_alignment_time = []

        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed)
            model = V4Model(state_dim=d, hidden_dim=hidden_dim, action_dim=k)
            train_with_panic(model, env, num_episodes=n_episodes,
                             episode_length=episode_length, panic_mode=mode, seed=seed)

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

            all_hidden_states.append(np.array(hidden_vals))
            all_risks.append(np.array(risk_vals))
            models_list.append(model)

            J_results = analyze_jacobian(model, states)
            k80_vals = [r["k80"] for r in J_results]
            k80_mean = float(np.mean(k80_vals))
            mode_results["k80_list"].append(k80_mean)
            mode_results["k80_values"].append(k80_vals)

            er_vals = [r["effective_rank_val"] for r in J_results]
            mode_results["effective_rank_list"].append(float(np.mean(er_vals)))

            h_all = np.array(hidden_vals)
            r_all = np.array(risk_vals).reshape(-1, 1)
            r2 = R2_probe(h_all, r_all)
            mode_results["R2_list"].append(r2)

            a_t = alignment_time(model, env, states, delta=100)
            if a_t:
                mode_results["alignment_time_mean_list"].append(float(np.mean(a_t)))
                all_alignment_time.extend(a_t)

        mode_results["cross_seed_consistency"] = cross_seed_consistency(
            models_list, states[:50])

        if mode_results["k80_list"]:
            k80_arr = np.array(mode_results["k80_list"])
            R2_arr = np.array(mode_results["R2_list"])
            alignment_arr = np.array(mode_results["alignment_time_mean_list"]) if mode_results["alignment_time_mean_list"] else np.array([0.0])

            k80_mean = float(np.mean(k80_arr))
            k80_std = float(np.std(k80_arr))
            R2_mean = float(np.mean(R2_arr))
            R2_std = float(np.std(R2_arr))
            align_mean = float(np.mean(alignment_arr))
            align_std = float(np.std(alignment_arr))

            avg_d = d
            fail = failure_criteria(k80_mean, avg_d, R2_mean, align_mean)

            mode_results["summary"] = {
                "k80": f"{k80_mean:.2f} ± {k80_std:.2f}",
                "R2": f"{R2_mean:.4f} ± {R2_std:.4f}",
                "alignment_time": f"{align_mean:.4f} ± {align_std:.4f}" if mode_results["alignment_time_mean_list"] else "N/A",
                "cross_seed_consistency": f"{mode_results['cross_seed_consistency']:.4f}",
                "failures": fail,
            }
        else:
            mode_results["summary"] = {"error": "No data"}

        results[mode] = mode_results

    return results


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_part3c_mechanism(n_seeds=8)
    with open("core_mvp_v4/results/part3c_mechanism.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("Part 3C complete. Summary:")
    for mode, data in r.items():
        s = data.get("summary", {})
        print(f"  {mode}: k80={s.get('k80')}, R2={s.get('R2')}, "
              f"fail={s.get('failures', {}).get('any_fail')}")
