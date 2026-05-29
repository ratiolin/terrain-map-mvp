"""Layer 2: Adaptation/Shaping Separation.

Steps 5-9:
  Step 5: Continuous drift training (linear sweep).
  Step 6: Construct purification indicators (PE, CA).
  Step 7: GMM distribution modeling (BIC selection).
  Step 8: Noise-only perturbation control.
  Step 9: Mode proportion curves and sigmoid fitting.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.core_env import MultiDimEnv, LinearDriftEnv
from core_mvp_v4.core_models import V4Model, get_designed_hidden_dim, DualHeadModel
from core_mvp_v4.core_logger import ExperimentLogger
from core_mvp_v4.core_metrics import fit_gmm_and_select, fit_sigmoid_proportion

D_DIMS = [4, 8, 16, 32]
K = 2
SEEDS = list(range(8))
DEFAULT_LR = 1e-3
LAMBDA_CTRL = 0.1


def run_layer2(results_dir, d_dims=None, seeds=None, total_steps=10000):
    if d_dims is None:
        d_dims = D_DIMS
    if seeds is None:
        seeds = SEEDS

    layer_dir = os.path.join(results_dir, "layer2")
    os.makedirs(layer_dir, exist_ok=True)
    all_results = {}

    for d in d_dims:
        seed_results = {}
        for seed in seeds:
            print(f"  Layer 2 - d={d}, seed={seed}")
            res = _run_layer2_single(d, seed, total_steps, layer_dir)
            seed_results[seed] = res

        agg = {}
        scalar_keys = ["PE_mean", "CA_mean", "PE_CA_corr", "best_components",
                        "slope_c", "inflection_d", "mode0_prop_low_g", "mode0_prop_high_g"]
        for key in scalar_keys:
            vals = [s.get(key, float('nan')) for s in seed_results.values()]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        all_results[f"d{d}"] = agg

        with open(os.path.join(layer_dir, f"d{d}_aggregated.json"), "w") as f:
            json.dump({"per_seed": {str(k): v for k, v in seed_results.items()},
                        "aggregated": agg}, f, indent=2)

    summary = {
        "slope_vs_d": {f"d{d}": all_results[f"d{d}"].get("slope_c", {}) for d in d_dims},
        "inflection_vs_d": {f"d{d}": all_results[f"d{d}"].get("inflection_d", {}) for d in d_dims},
        "best_components_vs_d": {f"d{d}": all_results[f"d{d}"].get("best_components", {}) for d in d_dims},
    }
    with open(os.path.join(layer_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Layer 2 complete. Results in {layer_dir}")
    return all_results


def _run_layer2_single(d, seed, total_steps, layer_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_dim = get_designed_hidden_dim(d)
    env = LinearDriftEnv(d=d, k=K, g_start=-0.5, g_end=2.0, T=total_steps,
                         theta=0.5, noise_std=0.05, force_scale=0.1,
                         action_scale=0.1, seed=seed)
    env.calibrate_noise_scales()

    model = V4Model(state_dim=d, hidden_dim=hidden_dim, action_dim=K)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)

    logger = ExperimentLogger(layer_dir, f"d{d}_drift", seed)

    pe_values = []
    ca_values = []
    g_values = []
    predictions = []
    actions_list = []

    state = env.reset()

    for t in range(total_steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        action, h, risk_pred = model(s_t)
        a_np = action.squeeze(0).detach().numpy()
        a_eff = a_np + np.random.randn(*a_np.shape) * 0.03

        next_state, risk, _, info = env.step(a_eff)
        g_val = info.get("drift", env.drift)

        risk_t = torch.tensor([[risk]], dtype=torch.float32)
        pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
        action_loss = torch.mean(action ** 2)
        loss = pred_loss + LAMBDA_CTRL * action_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pe_raw = float(abs(risk - risk_pred.item()))
        ca_raw = float(np.linalg.norm(a_np))

        pe_values.append(pe_raw)
        ca_values.append(ca_raw)
        g_values.append(g_val)
        predictions.append(float(risk_pred.item()))
        actions_list.append(a_np.copy())

        logger.log(state, a_np, next_state, pe_raw, ca_raw, 0.0, g_val)
        state = next_state

    logger.save()

    e_noise = _estimate_noise_error(d, seed + 10000, total_steps // 4)
    pe_values = np.array(pe_values)
    ca_values = np.array(ca_values)
    g_values = np.array(g_values)
    pe = np.maximum(0, pe_values - e_noise)
    ca_base = float(np.percentile(ca_values, 5))
    ca = np.maximum(0, ca_values - ca_base)

    pairs = np.column_stack([pe, ca])

    g_bins = {
        "low": g_values < np.median(g_values),
        "mid": (g_values >= np.median(g_values)) & (g_values < np.percentile(g_values, 75)),
        "high": g_values >= np.percentile(g_values, 75),
    }

    best_comp_low, models_low = fit_gmm_and_select(pairs[g_bins["low"]])
    best_comp_mid, models_mid = fit_gmm_and_select(pairs[g_bins["mid"]])
    best_comp_high, models_high = fit_gmm_and_select(pairs[g_bins["high"]])

    best_overall = max(set([best_comp_low, best_comp_mid, best_comp_high]))

    gmm_final = None
    if best_overall >= 2:
        try:
            from sklearn.mixture import GaussianMixture
            gmm_final = GaussianMixture(n_components=2, random_state=seed,
                                        covariance_type='full', n_init=5)
            labels = gmm_final.fit_predict(pairs)
        except Exception:
            labels = np.zeros(len(pairs))
    else:
        labels = np.zeros(len(pairs))

    g_sorted_idx = np.argsort(g_values)
    g_sorted = g_values[g_sorted_idx]
    labels_sorted = labels[g_sorted_idx] if len(labels) > 0 else np.zeros(len(g_sorted))

    window = max(100, total_steps // 100)
    props = []
    g_centers = []
    for i in range(0, len(g_sorted) - window, window // 2):
        props.append(float(np.mean(labels_sorted[i:i + window] == 0)))
        g_centers.append(float(np.mean(g_sorted[i:i + window])))

    sigmoid_fit = fit_sigmoid_proportion(np.array(g_centers), np.array(props))

    noise_results = _run_noise_control(d, seed + 20000, total_steps // 4)

    result = {
        "PE_mean": float(np.mean(pe_values)),
        "CA_mean": float(np.mean(ca_values)),
        "PE_CA_corr": float(np.corrcoef(pe_values, ca_values)[0, 1]) if len(pe_values) > 1 else 0,
        "best_components": best_overall,
        "best_comp_low": best_comp_low,
        "best_comp_mid": best_comp_mid,
        "best_comp_high": best_comp_high,
        "slope_c": sigmoid_fit.get("slope_c", 0.0),
        "inflection_d": sigmoid_fit.get("inflection_d", 0.0),
        "sigmoid_r2": sigmoid_fit.get("r2", 0.0),
        "mode0_prop_low_g": float(np.mean(labels[g_bins["low"]] == 0) if any(g_bins["low"]) else 0),
        "mode0_prop_high_g": float(np.mean(labels[g_bins["high"]] == 0) if any(g_bins["high"]) else 0),
        "noise_best_components": noise_results.get("best_components", 1),
        "noise_gmm_type": noise_results.get("gmm_type", "none"),
    }
    return result


def _estimate_noise_error(d, seed, steps):
    env = MultiDimEnv(d=d, k=K, drift=0.0, seed=seed)
    env.calibrate_noise_scales()
    env.reset()

    hidden_dim = get_designed_hidden_dim(d)
    model = V4Model(state_dim=d, hidden_dim=hidden_dim, action_dim=K)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)

    errors = []
    for _ in range(steps):
        s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
        action, h, risk_pred = model(s_t)
        a_np = np.zeros(K)
        next_state, risk, _, _ = env.step(a_np)

        risk_t = torch.tensor([[risk]], dtype=torch.float32)
        pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
        loss = pred_loss + LAMBDA_CTRL * torch.mean(action ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        errors.append(float(abs(risk - risk_pred.item())))

    return float(np.mean(errors[-steps // 4:]))


def _run_noise_control(d, seed, steps):
    env = MultiDimEnv(d=d, k=K, drift=0.0, seed=seed)
    env.calibrate_noise_scales()
    env.reset()

    hidden_dim = get_designed_hidden_dim(d)
    model = V4Model(state_dim=d, hidden_dim=hidden_dim, action_dim=K)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)

    pe_vals = []
    ca_vals = []

    for _ in range(steps):
        s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
        action, h, risk_pred = model(s_t)
        a_np = action.squeeze(0).detach().numpy()

        noise_pert = np.zeros(env.d)
        noise_pert[env.k:] = np.random.randn(env.d - env.k) * 0.1
        perturbed_state = env.get_state() + noise_pert
        env.state[:] = perturbed_state[:]

        next_state, risk, _, _ = env.step(a_np)

        risk_t = torch.tensor([[risk]], dtype=torch.float32)
        pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
        loss = pred_loss + LAMBDA_CTRL * torch.mean(action ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pe_vals.append(float(abs(risk - risk_pred.item())))
        ca_vals.append(float(np.linalg.norm(a_np)))

    pairs = np.column_stack([pe_vals, ca_vals])
    best_comp, models = fit_gmm_and_select(pairs)

    return {"best_components": best_comp, "gmm_type": "noise_control"}
