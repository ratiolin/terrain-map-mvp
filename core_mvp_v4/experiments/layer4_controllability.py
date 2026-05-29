"""Layer 4: Controllability Structure.

Steps 14-17:
  Step 14: Jacobian J = da/dh computation and SVD.
  Step 15: Representation invariance test (random orthogonal transformation).
  Step 16: Local controllability test (F-B along top singular vectors).
  Step 17: Scaling analysis (k80 vs d, monotonicity ratio vs d).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.core_env import MultiDimEnv
from core_mvp_v4.core_models import V4Model, get_designed_hidden_dim
from core_mvp_v4.core_training import train_closed_loop
from core_mvp_v4.core_metrics import compute_k80, effective_rank, spectral_entropy, alignment
from core_mvp_v4.core_logger import ExperimentLogger

D_DIMS = [4, 8, 16, 32]
K = 2
SEEDS = list(range(8))
DEFAULT_LR = 1e-3
LAMBDA_CTRL = 0.1


def run_layer4(results_dir, d_dims=None, seeds=None, episodes=5, steps=2000,
              n_sample_states=100):
    if d_dims is None:
        d_dims = D_DIMS
    if seeds is None:
        seeds = SEEDS

    layer_dir = os.path.join(results_dir, "layer4")
    os.makedirs(layer_dir, exist_ok=True)
    all_results = {}

    for d in d_dims:
        seed_results = {}

        for seed in seeds:
            print(f"  Layer 4 - d={d}, seed={seed}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            env = MultiDimEnv(d=d, k=K, drift=0.5, seed=seed)
            env.calibrate_noise_scales()

            hidden_dim = get_designed_hidden_dim(d)
            model = V4Model(state_dim=d, hidden_dim=hidden_dim, action_dim=K)

            train_closed_loop(model, env, num_episodes=episodes,
                              episode_length=steps, lr=DEFAULT_LR,
                              lambda_ctrl=LAMBDA_CTRL, seed=seed)
            model.eval()

            sample_states = _collect_sample_states(env, model, n_sample_states,
                                                    seed=seed + 1000)
            jacobian_results = _analyze_jacobians(model, sample_states, d)

            invariance_results = _test_representation_invariance(model, sample_states, seed)

            monotonicity_ratio = _test_local_controllability(env, model, sample_states,
                                                              jacobian_results, seed)

            k80_vals = [jr["k80"] for jr in jacobian_results]
            er_vals = [jr["effective_rank"] for jr in jacobian_results]

            seed_results[seed] = {
                "k80_mean": float(np.mean(k80_vals)) if k80_vals else 0,
                "k80_std": float(np.std(k80_vals)) if k80_vals else 0,
                "k80_over_d": float(np.mean(k80_vals) / d) if k80_vals else 0,
                "effective_rank_mean": float(np.mean(er_vals)) if er_vals else 0,
                "effective_rank_std": float(np.std(er_vals)) if er_vals else 0,
                "alignment_before_after": invariance_results.get("alignment", 0.0),
                "sv_entropy_before": float(invariance_results.get("sv_entropy_before", 0)),
                "sv_entropy_after": float(invariance_results.get("sv_entropy_after", 0)),
                "monotonicity_ratio": monotonicity_ratio,
            }

            print(f"    d={d} seed={seed}: k80={np.mean(k80_vals):.2f}, "
                  f"k80/d={np.mean(k80_vals)/d:.3f}, mono={monotonicity_ratio:.3f}")

        agg = {}
        for key in ["k80_mean", "k80_over_d", "effective_rank_mean",
                     "alignment_before_after", "monotonicity_ratio"]:
            vals = [s[key] for s in seed_results.values() if key in s]
            if vals:
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        all_results[f"d{d}"] = agg

        with open(os.path.join(layer_dir, f"d{d}_aggregated.json"), "w") as f:
            json.dump({"per_seed": {str(k): v for k, v in seed_results.items()},
                        "aggregated": agg}, f, indent=2)

    k80_over_d_values = [all_results[f"d{d}"].get("k80_over_d", {}).get("mean", 0) for d in d_dims]
    mono_values = [all_results[f"d{d}"].get("monotonicity_ratio", {}).get("mean", 0) for d in d_dims]

    scaling_fits = _fit_scaling_laws(d_dims, k80_over_d_values, mono_values)

    summary = {
        "k80_over_d_vs_d": {f"d{d}": all_results[f"d{d}"].get("k80_over_d", {}) for d in d_dims},
        "monotonicity_vs_d": {f"d{d}": all_results[f"d{d}"].get("monotonicity_ratio", {}) for d in d_dims},
        "scaling_fits": scaling_fits,
    }
    with open(os.path.join(layer_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Layer 4 complete. Results in {layer_dir}")
    return all_results


def _collect_sample_states(env, model, n_samples, seed):
    np.random.seed(seed)
    env.set_seed(seed)
    env.reset()
    states = []
    state = env.get_state()
    for t in range(n_samples * 5):
        a = model.act_numpy(state)
        ns, risk, done, info = env.step(a)
        if len(states) < n_samples and t % 5 == 0:
            states.append(state.copy())
        state = ns
    return states[:n_samples]


def _analyze_jacobians(model, states, d):
    results = []
    for s in states:
        J = model.compute_jacobian_a_h(s)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        results.append({
            "S": S.tolist(),
            "k80": compute_k80(S),
            "effective_rank": effective_rank(S),
            "spectral_entropy": spectral_entropy(S),
            "Vt": Vt.tolist(),
        })
    return results


def _test_representation_invariance(model, states, seed):
    np.random.seed(seed)
    h_dim = model.hidden_dim

    Q = np.linalg.qr(np.random.randn(h_dim, h_dim))[0]

    jacobians_before = []
    jacobians_after = []
    for s in states[:min(20, len(states))]:
        J = model.compute_jacobian_a_h(s)
        J_transformed = J @ Q.T
        jacobians_before.append(J)
        jacobians_after.append(J_transformed)

    J_avg_before = np.mean(jacobians_before, axis=0)
    J_avg_after = np.mean(jacobians_after, axis=0)

    _, S_before, Vt_before = np.linalg.svd(J_avg_before, full_matrices=False)
    _, S_after, Vt_after = np.linalg.svd(J_avg_after, full_matrices=False)

    k_before = compute_k80(S_before)
    k_after = compute_k80(S_after)
    k_alignment = min(k_before, k_after)
    if k_alignment > 0:
        align_val = alignment(Vt_before.T[:, :k_alignment], Vt_after.T[:, :k_alignment], k=k_alignment)
    else:
        align_val = 0.0

    return {
        "alignment": align_val,
        "sv_entropy_before": spectral_entropy(S_before),
        "sv_entropy_after": spectral_entropy(S_after),
    }


def _test_local_controllability(env, model, states, jacobian_results, seed):
    np.random.seed(seed)
    n_test = min(30, len(states))
    idx = np.random.choice(len(states), n_test, replace=False)

    cost_decreasing = 0
    for i, si in enumerate(idx):
        s = states[si]
        J = model.compute_jacobian_a_h(s)
        _, S, Vt = np.linalg.svd(J, full_matrices=False)

        if S.shape[0] < 1:
            continue

        top_direction = Vt[0, :]

        saved = env.save_state()
        env.state[:] = s.copy()
        _, r_baseline, _, _ = env.step(model.act_numpy(s))

        env.restore_state(saved)
        h0 = model.f_numpy(s)
        eps_vals = [0.01, 0.05]
        r_perturbed = []
        for eps in eps_vals:
            env.restore_state(saved)
            env.state[:] = s.copy()
            perturbed_state = _move_along_direction(model, s, top_direction, eps)
            ns, r, _, _ = env.step(model.act_numpy(perturbed_state))
            r_perturbed.append(r)

        r_best = min(r_perturbed)
        if r_best < r_baseline * 0.99:
            cost_decreasing += 1

        env.restore_state(saved)

    return float(cost_decreasing) / max(n_test, 1)


def _move_along_direction(model, s, direction, step):
    """Move state along direction in input space that aligns with top singular vector in h-space."""
    J = model.compute_jacobian_h_s(s)
    U_hs, S_hs, Vt_hs = np.linalg.svd(J, full_matrices=False)
    input_dir = Vt_hs[0, :]
    return s + step * input_dir


def _fit_scaling_laws(d_dims, k80_over_d_vals, mono_vals):
    d_arr = np.array(d_dims, dtype=float)

    fits = {}

    if all(v == 0 for v in k80_over_d_vals):
        fits["k80_over_d_power"] = {"power": 0, "r2": 0}
    else:
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(lambda x, a, b: a * x ** b, d_arr, k80_over_d_vals,
                                p0=[1.0, -0.5], maxfev=5000)
            fits["k80_over_d_power"] = {"a": float(popt[0]), "b": float(popt[1])}
        except Exception:
            fits["k80_over_d_power"] = {"a": 0, "b": 0, "error": "fit failed"}

    if all(v == 0 for v in mono_vals):
        fits["mono_power"] = {"power": 0}
    else:
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(lambda x, a, b: a * x ** b, d_arr, mono_vals,
                                p0=[1.0, -0.5], maxfev=5000)
            fits["mono_power"] = {"a": float(popt[0]), "b": float(popt[1])}
        except Exception:
            fits["mono_power"] = {"a": 0, "b": 0, "error": "fit failed"}

    return fits
