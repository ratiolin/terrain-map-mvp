"""Layer 3: Endogenous Boundary Recognition.

Steps 10-13:
  Step 10: Raw signal statistics (panic, g, mode correlations).
  Step 11: Intervention A — remove panic (replace with constant/noise), re-run policy.
  Step 12: Intervention B — inject false panic pulses at constant g.
  Step 13: Noise floor — completely uncontrollable environment, panic distribution.
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.core_env import DriftScheduleEnv, MultiDimEnv
from core_mvp_v4.core_models import V4Model, DualHeadModel, get_designed_hidden_dim
from core_mvp_v4.core_panic import PanicController, Mode, EMARunning
from core_mvp_v4.core_logger import ExperimentLogger
from core_mvp_v4.core_metrics import compute_spearman_correlation

D_DIMS = [4, 8, 16, 32]
K = 2
SEEDS = list(range(8))
DEFAULT_LR = 1e-3
LAMBDA_CTRL = 0.1


def _clean_nan(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def run_layer3(results_dir, d_dims=None, seeds=None, total_steps=20000):
    if d_dims is None:
        d_dims = D_DIMS
    if seeds is None:
        seeds = SEEDS

    layer_dir = os.path.join(results_dir, "layer3")
    os.makedirs(layer_dir, exist_ok=True)
    all_results = {}

    for d in d_dims:
        seed_results = {}

        for seed in seeds:
            print(f"  Layer 3 - d={d}, seed={seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)

            hidden_dim = get_designed_hidden_dim(d)
            env = DriftScheduleEnv(d=d, k=K, seed=seed)
            env.calibrate_noise_scales()

            model = DualHeadModel(state_dim=d, hidden_dim=hidden_dim, action_dim=K)
            panic_ctrl = PanicController(action_dim=K)
            optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)
            logger = ExperimentLogger(layer_dir, f"d{d}_panic", seed)

            panic_records = []
            panic_raw_records = []
            g_records = []
            mode_records = []
            switch_times = []
            prev_mode = "SHAPE"

            env.reset()
            state = env.get_state()

            for t in range(total_steps):
                s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
                a_shape, a_adapt, h, risk_pred = model(s_t)
                a_shape_np = a_shape.squeeze(0).detach().numpy()
                a_adapt_np = a_adapt.squeeze(0).detach().numpy()

                noise = np.random.randn(K) * 0.03
                a_shape_noisy = a_shape_np + noise
                a_adapt_noisy = a_adapt_np + noise

                if panic_ctrl.mode == Mode.SHAPE:
                    a_noisy = a_shape_noisy
                    a_used_for_loss = a_shape.squeeze(0)
                else:
                    a_noisy = a_adapt_noisy
                    a_used_for_loss = a_adapt.squeeze(0)

                next_state, risk, _, info = env.step(a_noisy)

                g_val = info.get("drift", 0.5)
                risk_t = torch.tensor([[risk]], dtype=torch.float32)
                pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
                action_loss = torch.mean(a_used_for_loss ** 2)
                loss = pred_loss + LAMBDA_CTRL * action_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                panic_info = panic_ctrl.update(env, float(pred_loss.item()),
                                                a_shape_np, a_adapt_np, t)

                current_mode = panic_info["mode"]
                if current_mode != prev_mode:
                    switch_times.append(t)
                    prev_mode = current_mode

                panic_records.append(panic_info["panic"])
                panic_raw_records.append(panic_info["panic_raw"])
                g_records.append(g_val)
                mode_records.append(0 if current_mode == "SHAPE" else 1)

                logger.log(state,
                           a_shape_np if current_mode == "SHAPE" else a_adapt_np,
                           next_state,
                           float(pred_loss.item()),
                           float(action_loss.item()),
                           panic_info["panic"],
                           current_mode)
                state = next_state

            logger.save()

            panic_arr = np.array(panic_records)
            panic_raw_arr = np.array(panic_raw_records)
            g_arr = np.array(g_records)
            mode_arr = np.array(mode_records)

            rho_pg, p_pg = compute_spearman_correlation(panic_arr, g_arr)
            rho_pm, p_pm = compute_spearman_correlation(panic_arr, mode_arr)

            fpr, fnr = _compute_fpr_fnr(panic_arr, g_arr, threshold_high=0.5)
            tau_delay = _compute_switch_delay(g_arr, mode_arr, switch_times)

            panic_mean_val = float(np.mean(panic_arr)) if len(panic_arr) > 0 else 0.0
            panic_std_val = float(np.std(panic_arr)) if len(panic_arr) > 0 else 0.0

            intervention_a_results = _run_intervention_a_remove_panic(
                model, env, d, seed, total_steps, panic_mean_val, panic_std_val)
            intervention_b_results = _run_intervention_b_false_panic(
                model, env, d, seed, total_steps // 4, panic_raw_arr)
            noise_floor = _compute_noise_floor_proper(d, seed + 30000, total_steps // 4, hidden_dim)

            seed_results[seed] = {
                "rho_panic_g": rho_pg,
                "rho_panic_mode": rho_pm,
                "fpr": fpr,
                "fnr": fnr,
                "switch_delay_tau": tau_delay,
                "n_switches": len(switch_times),
                "n_switches_no_panic": intervention_a_results.get("n_switches", 0),
                "mode0_prop_original": intervention_a_results.get("mode0_prop_original", 0),
                "mode0_prop_no_panic": intervention_a_results.get("mode0_prop_no_panic", 0),
                "false_panic_trigger_rate": intervention_b_results.get("trigger_rate", 0.0),
                "false_panic_switches": intervention_b_results.get("n_switches", 0),
                "noise_floor_mean": noise_floor.get("panic_mean", 0.0),
                "noise_floor_std": noise_floor.get("panic_std", 0.0),
            }

            print(f"    d={d} seed={seed}: rho(p,g)={rho_pg:.3f}, switches={len(switch_times)}, "
                  f"switches_no_panic={intervention_a_results.get('n_switches',0)}, "
                  f"false_trig={intervention_b_results.get('trigger_rate',0):.3f}")

        agg = {}
        for key in ["rho_panic_g", "rho_panic_mode", "fpr", "fnr", "switch_delay_tau",
                     "n_switches", "n_switches_no_panic",
                     "mode0_prop_original", "mode0_prop_no_panic",
                     "false_panic_trigger_rate", "false_panic_switches",
                     "noise_floor_mean", "noise_floor_std"]:
            vals = [_clean_nan(s.get(key)) for s in seed_results.values()]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            if vals:
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        all_results[f"d{d}"] = agg

        with open(os.path.join(layer_dir, f"d{d}_aggregated.json"), "w") as f:
            json.dump({"per_seed": {str(k): v for k, v in seed_results.items()},
                        "aggregated": agg}, f, indent=2)

    summary = {
        "rho_panic_g_vs_d": {f"d{d}": all_results[f"d{d}"].get("rho_panic_g", {}) for d in d_dims},
        "fpr_vs_d": {f"d{d}": all_results[f"d{d}"].get("fpr", {}) for d in d_dims},
        "fnr_vs_d": {f"d{d}": all_results[f"d{d}"].get("fnr", {}) for d in d_dims},
        "false_panic_trigger_vs_d": {f"d{d}": all_results[f"d{d}"].get("false_panic_trigger_rate", {}) for d in d_dims},
        "noise_floor_vs_d": {f"d{d}": all_results[f"d{d}"].get("noise_floor_mean", {}) for d in d_dims},
    }
    with open(os.path.join(layer_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Layer 3 complete. Results in {layer_dir}")
    return all_results


def _compute_fpr_fnr(panic_arr, g_arr, threshold_high=0.5):
    g_median = np.median(g_arr)
    high_panic = panic_arr > threshold_high
    high_g = g_arr > g_median

    tp = np.sum(high_panic & high_g)
    fp = np.sum(high_panic & ~high_g)
    fn = np.sum(~high_panic & high_g)
    tn = np.sum(~high_panic & ~high_g)

    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    return float(fpr), float(fnr)


def _compute_switch_delay(g_arr, mode_arr, switch_times):
    if not switch_times:
        return 0.0
    delays = []
    g_median = np.median(g_arr)
    for sw in switch_times:
        window = g_arr[max(0, sw - 50):min(len(g_arr), sw + 50)]
        if len(window) > 0:
            g_sw = window[len(window) // 2]
        else:
            g_sw = g_arr[sw] if sw < len(g_arr) else 0
        threshold = g_median + 0.2 * (g_arr.max() - g_median)
        crossing_idx = np.argmax(g_arr[:sw][::-1] < threshold)
        delays.append(crossing_idx)
    return float(np.mean(delays)) if delays else 0.0


def _run_intervention_a_remove_panic(model, env, d, seed, total_steps, panic_mean, panic_std):
    """Step 11: Replace panic with constant/noise, re-run policy, record mode behavior."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_dim = get_designed_hidden_dim(d)
    env.reset()
    state = env.get_state()

    panic_ctrl_orig = PanicController(action_dim=K)
    panic_ctrl_const = _ConstantPanicController(panic_mean)

    n_switches_orig = 0
    n_switches_const = 0
    prev_mode_orig = "SHAPE"
    prev_mode_const = "SHAPE"
    mode_labels_orig = []
    mode_labels_const = []
    n_val_steps = min(total_steps, 5000)

    env.reset()
    state = env.get_state()

    for t in range(n_val_steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        a_shape, a_adapt, h, risk_pred = model(s_t)
        a_shape_np = a_shape.squeeze(0).detach().numpy()
        a_adapt_np = a_adapt.squeeze(0).detach().numpy()

        noise = np.random.randn(K) * 0.03

        saved_env = env.save_state()

        panic_info_orig = panic_ctrl_orig.update(env, float(abs(risk_pred.item() - (
            float(np.linalg.norm(state[:K]))))), a_shape_np, a_adapt_np, t)

        env.restore_state(saved_env)
        panic_info_const = panic_ctrl_const.update(env, 0.0, a_shape_np, a_adapt_np, t)

        current_orig = panic_info_orig["mode"]
        current_const = panic_info_const["mode"]
        if current_orig != prev_mode_orig:
            n_switches_orig += 1
            prev_mode_orig = current_orig
        if current_const != prev_mode_const:
            n_switches_const += 1
            prev_mode_const = current_const

        mode_labels_orig.append(0 if current_orig == "SHAPE" else 1)
        mode_labels_const.append(0 if current_const == "SHAPE" else 1)

        env.restore_state(saved_env)
        if panic_ctrl_orig.mode == Mode.SHAPE:
            a_noisy = a_shape_np + noise
        else:
            a_noisy = a_adapt_np + noise
        next_state, risk, _, info = env.step(a_noisy)
        state = next_state

    return {
        "n_switches": n_switches_const,
        "n_switches_original": n_switches_orig,
        "mode0_prop_original": float(np.mean(mode_labels_orig)) if mode_labels_orig else 0,
        "mode0_prop_no_panic": float(np.mean(mode_labels_const)) if mode_labels_const else 0,
    }


def _run_intervention_b_false_panic(model, env, d, seed, steps, panic_raw_ref):
    """Step 12: Inject artificial panic pulses at constant g, record mode switches."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env.drift = 0.0
    env.reset()
    state = env.get_state()

    panic_ctrl = PanicController(action_dim=K)
    n_switches = 0
    prev_mode = "SHAPE"
    pulse_amplitude = 3.0
    if len(panic_raw_ref) > 0:
        pulse_amplitude = float(np.mean(np.abs(panic_raw_ref))) * 2.0

    pulse_times = [steps // 4, steps // 2, 3 * steps // 4]

    for t in range(steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        a_shape, a_adapt, h, risk_pred = model(s_t)
        a_shape_np = a_shape.squeeze(0).detach().numpy()
        a_adapt_np = a_adapt.squeeze(0).detach().numpy()

        noise = np.random.randn(K) * 0.03

        if t in pulse_times:
            fake_panic = pulse_amplitude + np.random.randn() * 0.1
        else:
            fake_panic = 0.0

        panic_info = panic_ctrl.update(env, float(abs(fake_panic)),
                                        a_shape_np, a_adapt_np, t)

        current_mode = panic_info["mode"]
        if current_mode != prev_mode:
            n_switches += 1
            prev_mode = current_mode

        if panic_ctrl.mode == Mode.SHAPE:
            a_noisy = a_shape_np + noise
        else:
            a_noisy = a_adapt_np + noise

        next_state, risk, _, info = env.step(a_noisy)
        state = next_state

    pulse_count = len(pulse_times)
    switches_during_pulse = 0
    for pt in pulse_times:
        window_start = pt
        window_end = min(steps, pt + 100)
        switches_in_window = 0
        cur_pm = "SHAPE" if window_start == 0 else _get_mode_at(panic_ctrl, pt - 1)
        for wt in range(window_start, window_end):
            m = _get_mode_at_simple(panic_ctrl, wt)
            if m != cur_pm:
                switches_in_window += 1
                cur_pm = m

    return {
        "trigger_rate": float(n_switches) / max(len(pulse_times), 1),
        "n_switches": n_switches,
        "pulse_amplitude": pulse_amplitude,
        "n_pulses": len(pulse_times),
    }


class _ConstantPanicController:
    """Returns constant panic signal, always in SHAPE mode."""

    def __init__(self, constant_value=0.0):
        self.constant = constant_value
        self.mode = Mode.SHAPE

    def update(self, env, pred_error, action_shape, action_adapt, t):
        return {
            "panic_raw": self.constant,
            "panic": 0.0,
            "mode": "SHAPE",
            "ctrl": 0.0,
            "trend": 0.0,
            "var_signal": 0.0,
        }


def _get_mode_at(panic_ctrl, t):
    return "SHAPE"


def _get_mode_at_simple(panic_ctrl, t):
    return "SHAPE"


def _compute_noise_floor_proper(d, seed, steps, hidden_dim):
    """Step 13: Completely uncontrollable env, record panic distribution via PanicController."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MultiDimEnv(d=d, k=K, drift=0.5, seed=seed)
    env.action_scale = 0.0
    env.calibrate_noise_scales()
    env.reset()

    model = DualHeadModel(state_dim=d, hidden_dim=hidden_dim, action_dim=K)
    panic_ctrl = PanicController(action_dim=K)

    panic_values = []
    state = env.get_state()

    for t in range(steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        a_shape, a_adapt, h, risk_pred = model(s_t)
        a_shape_np = a_shape.squeeze(0).detach().numpy()
        a_adapt_np = a_adapt.squeeze(0).detach().numpy()

        a_noisy = a_shape_np + np.random.randn(K) * 0.03
        next_state, risk, _, info = env.step(a_noisy)

        panic_info = panic_ctrl.update(env, float(abs(risk - risk_pred.item())),
                                        a_shape_np, a_adapt_np, t)
        panic_values.append(panic_info["panic"])
        state = next_state

    return {
        "panic_mean": float(np.mean(panic_values)),
        "panic_std": float(np.std(panic_values)),
    }
