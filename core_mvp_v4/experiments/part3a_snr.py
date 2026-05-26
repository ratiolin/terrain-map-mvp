import json
import numpy as np
from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, train_model, compute_jacobian
from core_mvp_v4.metrics import (
    compute_k80, effective_rank, R2_probe, spectral_entropy, analyze_jacobian,
)


def run_part3a_snr(n_seeds=8, d=16, k=2, hidden_dim=32,
                   n_episodes=3, episode_length=2000):
    """Part 3A: SNR boundary.

    Injects observation noise with varying sigma. Finds the critical sigma
    where R2 drops AND k80 rises (crossover point).
    """
    sigma_list = [0, 0.05, 0.1, 0.2, 0.5]
    results = {}

    for sigma in sigma_list:
        sigma_results = {
            "k80_list": [],
            "R2_list": [],
            "spectral_entropy_list": [],
            "effective_rank_list": [],
        }

        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed)
            model = V4Model(state_dim=d, hidden_dim=hidden_dim, action_dim=k)

            train_model(model, env, num_episodes=n_episodes,
                        episode_length=episode_length, seed=seed)

            noiseless_states = []
            hidden_states = []
            risk_vals = []
            env.reset()
            for _ in range(500):
                s = env.get_state()
                s_obs = s + np.random.normal(0, sigma, size=s.shape)
                noiseless_states.append(s)
                h = model.f_numpy(s_obs)
                hidden_states.append(h)
                a = model.act_numpy(s_obs)
                ns, risk, _, _ = env.step(a)
                risk_vals.append(risk)

            J_results = analyze_jacobian(model, noiseless_states)
            k80_vals = [r["k80"] for r in J_results]
            sigma_results["k80_list"].append(float(np.mean(k80_vals)))

            se_vals = [r["spectral_entropy_val"] for r in J_results]
            sigma_results["spectral_entropy_list"].append(float(np.mean(se_vals)))

            er_vals = [r["effective_rank_val"] for r in J_results]
            sigma_results["effective_rank_list"].append(float(np.mean(er_vals)))

            h_all = np.array(hidden_states)
            r_all = np.array(risk_vals).reshape(-1, 1)
            r2 = R2_probe(h_all, r_all)
            sigma_results["R2_list"].append(r2)

        for key in ["k80_list", "R2_list", "spectral_entropy_list", "effective_rank_list"]:
            arr = np.array(sigma_results[key])
            sigma_results[f"{key}_mean"] = float(np.mean(arr))
            sigma_results[f"{key}_std"] = float(np.std(arr))

        if sigma_results["k80_list"]:
            sigma_results["summary"] = {
                "sigma": sigma,
                "k80": f"{sigma_results['k80_list_mean']:.2f} ± {sigma_results['k80_list_std']:.2f}",
                "R2": f"{sigma_results['R2_list_mean']:.4f} ± {sigma_results['R2_list_std']:.4f}",
                "spectral_entropy": f"{sigma_results['spectral_entropy_list_mean']:.4f} ± {sigma_results['spectral_entropy_list_std']:.4f}",
            }

        results[str(sigma)] = sigma_results

    sigma_crit = _find_sigma_crit(results)
    results["sigma_critical"] = sigma_crit

    return results


def _find_sigma_crit(results):
    sigma_sorted = sorted(
        [k for k in results.keys() if k != "sigma_critical"],
        key=lambda x: float(x)
    )
    if len(sigma_sorted) < 2:
        return {"sigma_crit": None, "reason": "Insufficient data"}

    r2_means = []
    k80_means = []
    for k in sigma_sorted:
        r2_means.append(results[k].get("R2_list_mean", 0))
        k80_means.append(results[k].get("k80_list_mean", 0))

    base_r2 = r2_means[0]
    base_k80 = k80_means[0]

    sigma_crit = None
    for i, k in enumerate(sigma_sorted):
        if i == 0:
            continue
        if r2_means[i] < base_r2 * 0.7 and k80_means[i] > base_k80 * 1.3:
            sigma_crit = float(k)
            break

    return {"sigma_crit": sigma_crit, "r2_means": r2_means, "k80_means": k80_means}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_part3a_snr(n_seeds=8)
    with open("core_mvp_v4/results/part3a_snr_boundary.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"Part 3A complete. sigma_crit = {r.get('sigma_critical', {}).get('sigma_crit')}")
