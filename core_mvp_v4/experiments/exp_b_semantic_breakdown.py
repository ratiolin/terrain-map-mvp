"""Experiment B: Semantic Breakdown — SNR vs. Representation Competition.

d=8, k=2. Three sub-experiments to diagnose why semantics degrade in high dims.

B1 (SNR): boost control signal via larger force_scale / action_scale.
B2 (Competition): train with noise dimension dropout to force focus on control.
B3 (Capacity): increase hidden_dim to 128, 256 to test capacity bottleneck.
"""

import json
import numpy as np
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import (
    V4Model, train_with_signal_loss, train_with_noise_dropout,
    collect_controllability_data, compute_jacobian,
)
from core_mvp_v4.metrics import compute_k80, alignment


def _measure(model, env, d, k):
    test_h, test_C = collect_controllability_data(model, env, n_samples=300)
    probe = LinearRegression().fit(test_h, test_C)
    R2 = float(probe.score(test_h, test_C))

    states = []
    env.reset()
    for _ in range(100):
        states.append(env.get_state())
        a = model.act_numpy(env.get_state())
        env.step(a)
    J_mean = np.mean([compute_jacobian(model, s) for s in states[::3]], axis=0)
    _, S_mean, Vt_mean = np.linalg.svd(J_mean, full_matrices=False)
    V = Vt_mean.T
    k80 = compute_k80(S_mean)
    U_true = np.eye(d)[:, :k]
    k_use = min(k, V.shape[1])
    align_gt = float(alignment(V[:, :k_use], U_true[:, :k_use], k=k_use))
    return {"R2": R2, "alignment_gt": align_gt, "k80": k80}


def run_b1_snr(n_seeds=8, d=8, k=2, hd=32, n_episodes=3, episode_length=2000, lambda_signal=1.0):
    boosts = {
        "baseline": {"force_scale": 0.1, "action_scale": 0.1},
        "boost_x2":  {"force_scale": 0.2, "action_scale": 0.2},
        "boost_x3":  {"force_scale": 0.3, "action_scale": 0.3},
        "boost_x5":  {"force_scale": 0.5, "action_scale": 0.5},
    }
    results = {}

    for label, cfg in boosts.items():
        r2_vals = []; align_vals = []; k80_vals = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0,
                                     force_scale=cfg["force_scale"],
                                     action_scale=cfg["action_scale"])
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
            train_with_signal_loss(model, env, num_episodes=n_episodes,
                                   episode_length=episode_length,
                                   lambda_signal=lambda_signal, seed=seed)
            m = _measure(model, env, d, k)
            r2_vals.append(m["R2"]); align_vals.append(m["alignment_gt"]); k80_vals.append(m["k80"])
        results[label] = {
            "R2_mean": float(np.mean(r2_vals)), "R2_std": float(np.std(r2_vals)),
            "align_mean": float(np.mean(align_vals)), "align_std": float(np.std(align_vals)),
            "k80_mean": float(np.mean(k80_vals)), "k80_std": float(np.std(k80_vals)),
        }
    baseline = results["baseline"]
    best = max(results.keys(), key=lambda x: results[x]["align_mean"])
    recovered = results[best]["align_mean"] > baseline["align_mean"] * 1.15
    results["diagnosis"] = {
        "SNR_issue": not recovered,
        "best_boost": best,
        "note": "If boosting signal recovers alignment → SNR bottleneck. Else → competition/capacity."
    }
    return results


def run_b2_competition(n_seeds=8, d=8, k=2, hd=32, n_episodes=3, episode_length=2000, lambda_signal=1.0):
    dropout_levels = [0.0, 0.3, 0.5, 0.7]
    results = {}

    for dp in dropout_levels:
        r2_vals = []; align_vals = []; k80_vals = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
            train_with_noise_dropout(model, env, num_episodes=n_episodes,
                                     episode_length=episode_length,
                                     lambda_signal=lambda_signal, seed=seed,
                                     noise_dropout=dp)
            env.reset()
            m = _measure(model, env, d, k)
            r2_vals.append(m["R2"]); align_vals.append(m["alignment_gt"]); k80_vals.append(m["k80"])
        results[f"dropout_{dp}"] = {
            "R2_mean": float(np.mean(r2_vals)), "R2_std": float(np.std(r2_vals)),
            "align_mean": float(np.mean(align_vals)), "align_std": float(np.std(align_vals)),
            "k80_mean": float(np.mean(k80_vals)), "k80_std": float(np.std(k80_vals)),
        }
    baseline = results["dropout_0.0"]
    best = max(results.keys(), key=lambda x: results[x]["align_mean"])
    recovered = results[best]["align_mean"] > baseline["align_mean"] * 1.15
    results["diagnosis"] = {
        "competition_issue": not recovered,
        "best_dropout": best,
        "note": "If dropout recovers alignment → competition problem. Else → SNR or capacity."
    }
    return results


def run_b3_capacity(n_seeds=8, d=8, k=2, n_episodes=3, episode_length=2000, lambda_signal=1.0):
    hidden_dims = [32, 64, 128, 256]
    results = {}

    for hd in hidden_dims:
        r2_vals = []; align_vals = []; k80_vals = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
            train_with_signal_loss(model, env, num_episodes=n_episodes,
                                   episode_length=episode_length,
                                   lambda_signal=lambda_signal, seed=seed)
            m = _measure(model, env, d, k)
            r2_vals.append(m["R2"]); align_vals.append(m["alignment_gt"]); k80_vals.append(m["k80"])
        results[str(hd)] = {
            "R2_mean": float(np.mean(r2_vals)), "R2_std": float(np.std(r2_vals)),
            "align_mean": float(np.mean(align_vals)), "align_std": float(np.std(align_vals)),
            "k80_mean": float(np.mean(k80_vals)), "k80_std": float(np.std(k80_vals)),
        }
    baseline = results["32"]
    best = max(results.keys(), key=lambda x: results[x]["align_mean"])
    recovered = results[best]["align_mean"] > baseline["align_mean"] * 1.15
    results["diagnosis"] = {
        "capacity_issue": not recovered,
        "best_hd": int(best),
        "note": "If larger hd recovers alignment → capacity bottleneck. Else → SNR or optimization bias."
    }
    return results


def run_exp_b_semantic_breakdown(n_seeds=8, d=8, k=2, hd=32,
                                 n_episodes=3, episode_length=2000, lambda_signal=1.0):
    results = {}

    print("  B1: SNR test...")
    results["B1_SNR"] = run_b1_snr(n_seeds=n_seeds, d=d, k=k, hd=hd,
                                   n_episodes=n_episodes, episode_length=episode_length,
                                   lambda_signal=lambda_signal)

    print("  B2: Competition test...")
    results["B2_competition"] = run_b2_competition(n_seeds=n_seeds, d=d, k=k, hd=hd,
                                                   n_episodes=n_episodes, episode_length=episode_length,
                                                   lambda_signal=lambda_signal)

    print("  B3: Capacity test...")
    results["B3_capacity"] = run_b3_capacity(n_seeds=n_seeds, d=d, k=k,
                                             n_episodes=n_episodes, episode_length=episode_length,
                                             lambda_signal=lambda_signal)

    results["overall_diagnosis"] = _synthesize(results)
    return results


def _synthesize(results):
    b1_diag = results["B1_SNR"]["diagnosis"]
    b2_diag = results["B2_competition"]["diagnosis"]
    b3_diag = results["B3_capacity"]["diagnosis"]

    causes = []
    if b1_diag["SNR_issue"]:
        causes.append("SNR")
    if b2_diag["competition_issue"]:
        causes.append("competition")
    if b3_diag["capacity_issue"]:
        causes.append("capacity")

    if not causes:
        return "ALL MITIGATED: all interventions recovered semantics. Root cause ambiguous (multi-factor)."
    elif len(causes) == 3:
        return "DEEP ISSUE: SNR + competition + capacity all fail. Representation fundamentally unable to encode control signal at this dimensionality."
    else:
        return f"ROOT CAUSE: {' + '.join(causes)}. The indicated mechanisms are the primary bottlenecks."


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_exp_b_semantic_breakdown(n_seeds=8)
    with open("core_mvp_v4/results/b_semantic_breakdown.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"Experiment B: {r['overall_diagnosis']}")
