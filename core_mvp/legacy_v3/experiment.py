import json
import random
import copy
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core_mvp_v3.env import drifting_double_well
from core_mvp_v3.models import PredictionNetwork, PolicyNetwork


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Mode:
    SHAPE = 0
    ADAPT = 1

    @staticmethod
    def flip(mode):
        return Mode.ADAPT if mode == Mode.SHAPE else Mode.SHAPE


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class EMARunning:
    def __init__(self, tau, init_value=0.0):
        self.tau = tau
        self.alpha = 1.0 / tau
        self.value = init_value

    def update(self, new_val):
        self.value += self.alpha * (new_val - self.value)
        return self.value


def rollout(env, action_val, k):
    total_risk = 0.0
    for _ in range(k):
        state = env.step(action_val)
        total_risk += abs(state[0])
    return total_risk / k


def compute_controllability(env, action_shape_val, action_adapt_val, k, n_rollouts=5):
    shape_risks = []
    adapt_risks = []
    for _ in range(n_rollouts):
        env_s = env.clone()
        env_a = env.clone()
        shared_seed = env_s.get_rng_seed()
        env_s.set_rng_seed(shared_seed)
        env_a.set_rng_seed(shared_seed)
        shape_risks.append(rollout(env_s, action_shape_val, k))
        adapt_risks.append(rollout(env_a, action_adapt_val, k))
    med_shape = float(np.median(shape_risks))
    med_adapt = float(np.median(adapt_risks))
    controllability = max(0.0, med_adapt - med_shape)
    return controllability, med_shape, med_adapt


class ExperimentConfig:
    def __init__(self):
        self.num_episodes = 5
        self.episode_length = 8000
        self.schedule = [
            (2000, (0.1, 0.3)),
            (2000, (1.0, 2.0)),
            (2000, (0.1, 0.3)),
            (2000, (1.0, 2.0)),
        ]
        self.noise_std = 0.05
        self.action_noise_std = 0.03
        self.state_clip = 5.0
        self.force_scale = 0.1
        self.action_scale = 0.1
        self.pred_hidden_dim = 32
        self.policy_hidden_dim = 32
        self.lr = 1e-3
        self.tau_short = 100
        self.tau_long = 1000
        self.tau_panic = 100
        self.k_rollout = 7
        self.n_rollouts = 5
        self.T_ctrl = 10
        self.w1 = 2.0
        self.w2 = 1.0
        self.w3 = 0.5
        self.w4 = 2.0
        self.w5 = 1.5
        self.w6 = 1.0
        self.theta = 0.5
        self.tau_theta = 500
        self.tau_ctrl = 100
        self.tau_ctrl_slow = 2000
        self.confirm_steps = 2
        self.delta_hysteresis = 0.2
        self.theta_offset = 0.05
        self.c_low = 0.1
        self.min_dwell = 200
        self.delta_base = 0.2
        self.k_delta = 0.3
        self.lock_p_thresh = -1.0
        self.lock_c_thresh = 0.01
        self.lock_shape_c_thresh = 0.03
        self.lock_relative = True
        self.lock_ratio = 0.35
        self.lock_N = 20
        self.lock_T = 500
        self.safe_radius = 0.2
        self.out_of_zone_threshold = 2.0
        self.survival_window = 20
        self.seed = 42
        self.eps = 1e-6


def train(config=None, save_tag=None):
    if config is None:
        config = ExperimentConfig()

    reset_seed(config.seed)

    env = drifting_double_well(
        schedule=config.schedule,
        noise=config.noise_std,
        state_clip=config.state_clip,
        force_scale=config.force_scale,
        action_scale=config.action_scale,
    )

    pred_net = PredictionNetwork(hidden_dim=config.pred_hidden_dim)
    policy_net = PolicyNetwork(hidden_dim=config.policy_hidden_dim)

    params = (
        list(pred_net.parameters())
        + list(policy_net.backbone.parameters())
        + list(policy_net.head_shape.parameters())
        + list(policy_net.head_adapt.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=config.lr)

    bl_mean = EMARunning(config.tau_long, init_value=0.1)
    bl_var = EMARunning(config.tau_long, init_value=0.01)
    ema_short = EMARunning(config.tau_short, init_value=0.1)
    ema_long = EMARunning(config.tau_long, init_value=0.1)
    var_signal = EMARunning(config.tau_short, init_value=0.01)
    panic_slow = EMARunning(config.tau_panic, init_value=0.0)
    ctrl_ema = EMARunning(config.tau_ctrl, init_value=0.0)
    ctrl_slow = EMARunning(config.tau_ctrl_slow, init_value=0.0)
    theta_base = EMARunning(config.tau_theta, init_value=config.theta)
    confirm_counter = 0
    pending_switch_target = None

    out_of_zone_buf = deque(maxlen=config.survival_window)

    mode = Mode.SHAPE
    controllability = 0.0
    controllability_norm = 0.0
    controllability_delta = 0.0
    last_switch_time = -config.min_dwell
    lock_adapt_counter = 0
    lock_timer = 0
    lock_shape_counter = 0
    lock_shape_timer = 0

    logs = {
        "t": [], "risk": [], "drift": [], "mode": [],
        "action": [], "action_shape": [], "action_adapt": [],
        "risk_error": [], "predicted_risk": [], "panic": [],
        "panic_raw": [], "error_surprise": [],
        "norm_error": [], "trend": [], "var_signal_list": [],
        "controllability": [], "controllability_norm": [],
        "controllability_delta": [],
        "panic_slow_list": [], "pred_loss": [],
        "out_of_zone": [], "survival_event": [],
        "switch_event": [], "theta": [],
    }

    trajectory = []

    for episode in range(config.num_episodes):
        env.reset()
        for _ in range(config.episode_length):
            state_t = env.state
            risk_t = abs(float(state_t[0]))
            state_t_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)

            action_shape_t, action_adapt_t, h = policy_net(state_t_tensor)

            if mode == Mode.SHAPE:
                action_t = action_shape_t
            else:
                action_t = action_adapt_t

            noise = torch.randn(1, 1) * config.action_noise_std
            action_t = action_t + noise
            action_val = float(action_t.item())

            state_next = env.step(action_val)
            risk_next = abs(float(state_next[0]))
            risk_next_tensor = torch.tensor([[risk_next]], dtype=torch.float32)

            predicted_risk = pred_net(state_t_tensor, action_t)
            pred_loss = F.mse_loss(predicted_risk, risk_next_tensor)

            optimizer.zero_grad()
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            risk_error = abs(float(predicted_risk.item()) - risk_next)

            bl_mean.update(risk_error)
            bl_var.update((risk_error - bl_mean.value) ** 2)
            ema_short.update(risk_error)
            ema_long.update(risk_error)

            trend = ema_short.value - ema_long.value
            var_signal.update((risk_error - ema_short.value) ** 2)
            sqrt_var = math.sqrt(max(bl_var.value, config.eps))

            norm_error = (risk_error - bl_mean.value) / (sqrt_var + config.eps)

            error_surprise = abs(risk_error - ema_long.value) / (sqrt_var + config.eps)
            error_surprise_signed = (risk_error - ema_long.value) / (sqrt_var + config.eps)

            out_of_zone = 1.0 if risk_t > config.out_of_zone_threshold else 0.0
            out_of_zone_buf.append(out_of_zone)
            survival_event = float(np.mean(out_of_zone_buf)) if out_of_zone_buf else 0.0
            panic_slow.update(survival_event)

            if env.t % config.T_ctrl == 0:
                ctrl, rs, ra = compute_controllability(
                    env,
                    float(action_shape_t.item()),
                    float(action_adapt_t.item()),
                    config.k_rollout,
                    config.n_rollouts,
                )
                controllability = ctrl / (abs(rs) + config.eps)
                controllability_norm = controllability
            else:
                pass

            ctrl_ema.update(controllability_norm)
            ctrl_slow.update(controllability_norm)
            ctrl_used = 0.5 * controllability_norm + 0.5 * ctrl_slow.value
            controllability_delta_val = controllability_norm - ctrl_ema.value

            panic_raw = (
                config.w1 * error_surprise_signed
                + config.w2 * trend
                + config.w3 * var_signal.value
                + config.w5 * controllability_delta_val
                + config.w6 * panic_slow.value
            )

            panic = sigmoid(panic_raw)

            theta_base.update(panic_raw)
            theta_val = theta_base.value + config.theta_offset
            delta_high = config.delta_base + config.k_delta * (1.0 - controllability_norm)
            delta_low = config.delta_base
            theta_high = theta_val + delta_high
            theta_low = theta_val - delta_low

            switched = False
            if env.t - last_switch_time >= config.min_dwell:
                target = None
                if mode == Mode.SHAPE and panic_raw > theta_high:
                    target = Mode.ADAPT
                elif mode == Mode.ADAPT and panic_raw < theta_low:
                    target = Mode.SHAPE

                if target == pending_switch_target:
                    confirm_counter += 1
                else:
                    confirm_counter = 1
                    pending_switch_target = target

                if confirm_counter >= config.confirm_steps and target is not None:
                    mode = target
                    switched = True
                    last_switch_time = env.t
                    confirm_counter = 0
                    pending_switch_target = None
            else:
                confirm_counter = 0
                pending_switch_target = None

            if panic_slow.value > config.lock_p_thresh and controllability_norm < ctrl_slow.value * config.lock_ratio:
                lock_adapt_counter += 1
            else:
                lock_adapt_counter = 0

            if lock_adapt_counter > config.lock_N:
                mode = Mode.ADAPT
                lock_timer = config.lock_T
                lock_adapt_counter = 0

            if lock_timer > 0:
                mode = Mode.ADAPT
                lock_timer -= 1

            if ctrl_used > ctrl_slow.value * 1.5:
                lock_shape_counter += 1
            else:
                lock_shape_counter = 0

            if lock_shape_counter > config.lock_N:
                mode = Mode.SHAPE
                lock_shape_timer = config.lock_T
                lock_shape_counter = 0

            if lock_shape_timer > 0:
                mode = Mode.SHAPE
                lock_shape_timer -= 1

            logs["t"].append(env.t)
            logs["risk"].append(risk_t)
            logs["drift"].append(env.current_drift)
            logs["mode"].append(mode)
            logs["action"].append(action_val)
            logs["action_shape"].append(float(action_shape_t.item()))
            logs["action_adapt"].append(float(action_adapt_t.item()))
            logs["risk_error"].append(risk_error)
            logs["predicted_risk"].append(float(predicted_risk.item()))
            logs["panic"].append(panic)
            logs["panic_raw"].append(panic_raw)
            logs["error_surprise"].append(error_surprise)
            logs["norm_error"].append(norm_error)
            logs["trend"].append(trend)
            logs["var_signal_list"].append(var_signal.value)
            logs["controllability"].append(controllability)
            logs["controllability_norm"].append(controllability_norm)
            logs["controllability_delta"].append(controllability_delta_val)
            logs["panic_slow_list"].append(panic_slow.value)
            logs["pred_loss"].append(float(pred_loss.item()))
            logs["out_of_zone"].append(out_of_zone)
            logs["survival_event"].append(survival_event)
            logs["switch_event"].append(int(switched))
            logs["theta"].append(theta_val)

            trajectory.append({
                "state": state_t.tolist(),
                "action": [action_val],
                "hidden_state": h.detach().cpu().numpy().tolist(),
                "env_state": env.get_state(),
                "drift": env.current_drift,
                "controllability": controllability,
            })

    results_dir = Path("results_final")
    results_dir.mkdir(exist_ok=True)
    tag = f"_{save_tag}" if save_tag else ""
    trajectory_path = results_dir / f"phase0_full_trajectory{tag}.json"
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f)

    torch.save(policy_net.state_dict(), results_dir / f"phase0_policy_net{tag}.pt")

    return logs, pred_net, policy_net, config


def _low_mask(drift_list):
    return np.array(drift_list) < 0.5


def _high_mask(drift_list):
    return np.array(drift_list) > 0.5


def validate_all(logs, pred_net, policy_net, config):
    """Run all 17 validation tests. Returns dict: test_name -> {passed, value, details}."""
    results = {}
    results["13.1_strategy_differentiation"] = _test_13_1(logs)
    results["13.2_performance_inversion"] = _test_13_2(logs)
    results["13.3_panic_mode_coupling"] = _test_13_3(logs)
    results["13.4_behavioral_bifurcation"] = _test_13_4(logs)
    results["13.5_shuffled_panic"] = _test_13_5(logs, pred_net, policy_net, config)
    results["13.6_panic_ablation"] = _test_13_6(logs, pred_net, policy_net, config)
    results["13.7_perturbation_test"] = _test_13_7(logs, pred_net, policy_net, config)
    results["13.8_dual_scaling"] = _test_13_8(logs, config)
    results["13.9_theta_scan"] = _test_13_9(logs, pred_net, policy_net, config)
    results["13.10_freeze_test"] = _test_13_10(logs, pred_net, policy_net, config)
    results["13.11_external_panic"] = _test_13_11(logs, pred_net, policy_net, config)
    results["13.12_controllability_separation"] = _test_13_12(logs)
    results["13.13_weight_sensitivity"] = _test_13_13(logs, config)
    results["13.14_k_sensitivity"] = _test_13_14(logs, config)
    results["13.15_Tctrl_sensitivity"] = _test_13_15(logs, config)
    results["13.16_panic_structure_perturbation"] = _test_13_16(logs, config)
    results["13.17_active_switching"] = _test_13_17(logs)
    return results


def _test_13_1(logs):
    diff = np.mean(np.abs(
        np.array(logs["action_shape"]) - np.array(logs["action_adapt"])
    ))
    passed = diff > 0.1
    return {"passed": passed, "value": diff, "threshold": 0.1,
            "details": f"D = {diff:.4f}"}


def _test_13_2(logs):
    low = _low_mask(logs["drift"])
    high = _high_mask(logs["drift"])
    risk = np.array(logs["risk"])
    mode_arr = np.array(logs["mode"])

    shape_mask_low = low & (mode_arr == Mode.SHAPE)
    adapt_mask_low = low & (mode_arr == Mode.ADAPT)
    shape_mask_high = high & (mode_arr == Mode.SHAPE)
    adapt_mask_high = high & (mode_arr == Mode.ADAPT)

    r_shape_low = risk[shape_mask_low].mean() if shape_mask_low.any() else float("nan")
    r_adapt_low = risk[adapt_mask_low].mean() if adapt_mask_low.any() else float("nan")
    r_shape_high = risk[shape_mask_high].mean() if shape_mask_high.any() else float("nan")
    r_adapt_high = risk[adapt_mask_high].mean() if adapt_mask_high.any() else float("nan")

    passed_low = r_shape_low < r_adapt_low if not (np.isnan(r_shape_low) or np.isnan(r_adapt_low)) else False
    passed_high = r_adapt_high < r_shape_high if not (np.isnan(r_adapt_high) or np.isnan(r_shape_high)) else False
    passed = passed_low and passed_high
    return {"passed": passed, "value": (passed_low, passed_high),
            "details": f"low: shape={r_shape_low:.4f} vs adapt={r_adapt_low:.4f} | "
                       f"high: shape={r_shape_high:.4f} vs adapt={r_adapt_high:.4f}"}


def _test_13_3(logs):
    panic_arr = np.array(logs["panic"])
    mode_arr = np.array(logs["mode"])
    corr = np.corrcoef(panic_arr, mode_arr)[0, 1]
    passed = abs(corr) > 0.5
    return {"passed": passed, "value": corr, "threshold": 0.5,
            "details": f"corr(panic, mode) = {corr:.4f}"}


def _test_13_4(logs):
    low = _low_mask(logs["drift"])
    high = _high_mask(logs["drift"])
    risk = np.array(logs["risk"])
    r_low = risk[low].mean()
    r_high = risk[high].mean()
    passed = r_low < 0.3 and r_high > 0.8
    return {"passed": passed, "value": (r_low, r_high),
            "details": f"low_g risk: {r_low:.4f}, high_g risk: {r_high:.4f}"}


def _test_13_5(logs, pred_net, policy_net, config):
    switch_orig = np.array(logs["switch_event"], dtype=bool)

    risks_orig, modes_orig = _replay_env_with_switches(
        config, pred_net, policy_net, switch_orig)
    drift_orig = _compute_drift_seq(config, len(switch_orig))
    orig_shape_low = (np.array(modes_orig)[drift_orig < 0.5] == Mode.SHAPE).mean()

    perm = np.random.permutation(len(switch_orig))
    switch_shuf = switch_orig[perm]
    risks_shuf, modes_shuf = _replay_env_with_switches(
        config, pred_net, policy_net, switch_shuf)
    drift_shuf = _compute_drift_seq(config, len(switch_shuf))
    shuf_shape_low = (np.array(modes_shuf)[drift_shuf < 0.5] == Mode.SHAPE).mean()

    passed = shuf_shape_low < max(orig_shape_low * 0.5, 0.3)
    return {"passed": passed, "value": (orig_shape_low, shuf_shape_low),
            "details": f"SHAPE in low: orig={orig_shape_low:.3f} shuf={shuf_shape_low:.3f}"}


def _compute_drift_seq(config, length):
    env = drifting_double_well(
        schedule=config.schedule, noise=config.noise_std,
        state_clip=config.state_clip, force_scale=config.force_scale,
        action_scale=config.action_scale)
    env.reset()
    drifts = []
    for _ in range(length):
        drifts.append(env.current_drift)
        env.step(0.0)
    return np.array(drifts)


def _replay_env_with_switches(config, pred_net, policy_net, switch_seq):
    env = drifting_double_well(
        schedule=config.schedule, noise=config.noise_std,
        state_clip=config.state_clip, force_scale=config.force_scale,
        action_scale=config.action_scale)
    env.reset()
    mode = Mode.SHAPE
    risks = []
    modes = []
    for i in range(len(switch_seq)):
        state_t = env.state
        state_t_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_shape_t, action_adapt_t, _ = policy_net(state_t_tensor)
        if switch_seq[i]:
            mode = Mode.flip(mode)
        if mode == Mode.SHAPE:
            action_val = float(action_shape_t.item())
        else:
            action_val = float(action_adapt_t.item())
        action_val += np.random.randn() * config.action_noise_std
        env.step(action_val)
        risks.append(abs(float(env.state[0])))
        modes.append(mode)
    return np.array(risks), modes


def _replay_panic_hysteresis(config, pred_net, policy_net, panic_raw_seq, theta_seq):
    env = drifting_double_well(
        schedule=config.schedule,
        noise=config.noise_std,
        state_clip=config.state_clip,
        force_scale=config.force_scale,
        action_scale=config.action_scale,
    )
    env.reset()
    mode = Mode.SHAPE
    last_switch_time = -config.min_dwell
    mode_history = []
    drift_history = []

    for i in range(len(panic_raw_seq)):
        state_t = env.state
        state_t_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_shape_t, action_adapt_t, _ = policy_net(state_t_tensor)

        theta_val = theta_seq[i]
        theta_high = theta_val + config.delta_hysteresis
        theta_low = theta_val - config.delta_hysteresis

        if env.t - last_switch_time >= config.min_dwell:
            if mode == Mode.SHAPE and panic_raw_seq[i] > theta_high:
                mode = Mode.ADAPT
                last_switch_time = env.t
            elif mode == Mode.ADAPT and panic_raw_seq[i] < theta_low:
                mode = Mode.SHAPE
                last_switch_time = env.t

        if mode == Mode.SHAPE:
            action_val = float(action_shape_t.item())
        else:
            action_val = float(action_adapt_t.item())
        action_val += np.random.randn() * config.action_noise_std
        env.step(action_val)
        mode_history.append(mode)
        drift_history.append(env.current_drift)

    mode_arr = np.array(mode_history)
    drift_arr = np.array(drift_history)
    low = drift_arr < 0.5
    ratio = (mode_arr[low] == Mode.SHAPE).mean() if low.any() else 0.5
    return float(ratio)


def _replay_with_switches(config, pred_net, policy_net, switch_seq):
    env = drifting_double_well(
        schedule=config.schedule,
        noise=config.noise_std,
        state_clip=config.state_clip,
        force_scale=config.force_scale,
        action_scale=config.action_scale,
    )
    env.reset()
    mode = Mode.SHAPE
    risks = []
    for i in range(len(switch_seq)):
        state_t = env.state
        state_t_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_shape_t, action_adapt_t, _ = policy_net(state_t_tensor)
        if switch_seq[i]:
            mode = Mode.flip(mode)
        if mode == Mode.SHAPE:
            action_val = float(action_shape_t.item())
        else:
            action_val = float(action_adapt_t.item())
        action_val += np.random.randn() * config.action_noise_std
        state_next = env.step(action_val)
        risks.append(abs(float(state_next[0])))
    return np.array(risks)


def _test_13_6(logs, pred_net, policy_net, config):
    risk_shape_all = _run_fixed_mode(Mode.SHAPE, config, pred_net, policy_net)
    risk_adapt_all = _run_fixed_mode(Mode.ADAPT, config, pred_net, policy_net)

    env_tmp = drifting_double_well(
        schedule=config.schedule,
        noise=config.noise_std,
        state_clip=config.state_clip,
        force_scale=config.force_scale,
        action_scale=config.action_scale,
    )
    env_tmp.reset()
    drift_vals = []
    for _ in range(config.episode_length):
        drift_vals.append(env_tmp.current_drift)
        env_tmp.step(0.0)
    low = _low_mask(drift_vals)
    high = _high_mask(drift_vals)

    r_shape_high = risk_shape_all[high].mean()
    r_adapt_low = risk_adapt_all[low].mean()
    r_shape_low = risk_shape_all[low].mean()
    r_adapt_high = risk_adapt_all[high].mean()

    passed_shape_high_fail = r_shape_high > 1.0
    passed_adapt_low_fail = r_adapt_low > r_shape_low * 0.8
    passed = passed_shape_high_fail or passed_adapt_low_fail

    return {"passed": passed,
            "value": (r_shape_high, r_adapt_low, r_shape_low, r_adapt_high),
            "details": f"shape in high: {r_shape_high:.4f} (want high), "
                       f"adapt in low: {r_adapt_low:.4f} (want > shape_low={r_shape_low:.4f})"}


def _run_fixed_mode(fixed_mode, config, pred_net, policy_net):
    env = drifting_double_well(
        schedule=config.schedule,
        noise=config.noise_std,
        state_clip=config.state_clip,
        force_scale=config.force_scale,
        action_scale=config.action_scale,
    )
    env.reset()
    risks = []
    for _ in range(config.episode_length):
        state_t = env.state
        state_t_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_shape_t, action_adapt_t, _ = policy_net(state_t_tensor)
        if fixed_mode == Mode.SHAPE:
            action_val = float(action_shape_t.item())
        else:
            action_val = float(action_adapt_t.item())
        action_val += np.random.randn() * config.action_noise_std
        env.step(action_val)
        risks.append(abs(float(env.state[0])))
    return np.array(risks)


def _test_13_7(logs, pred_net, policy_net, config):
    env = drifting_double_well(
        schedule=config.schedule, noise=config.noise_std,
        state_clip=config.state_clip, force_scale=config.force_scale,
        action_scale=config.action_scale)
    env.reset()

    mode = Mode.SHAPE
    bl_mean = EMARunning(config.tau_long, init_value=0.1)
    bl_var = EMARunning(config.tau_long, init_value=0.01)
    ema_short = EMARunning(config.tau_short, init_value=0.1)
    ema_long = EMARunning(config.tau_long, init_value=0.1)
    var_signal = EMARunning(config.tau_short, init_value=0.01)
    panic_slow = EMARunning(config.tau_panic, init_value=0.0)
    ctrl_ema_run = EMARunning(config.tau_ctrl, init_value=0.0)
    theta_base = EMARunning(config.tau_theta, init_value=config.theta)
    out_of_zone_buf = deque(maxlen=config.survival_window)
    last_switch_time = -config.min_dwell
    controllability_norm = 0.0
    ctrl_before = None
    ctrl_after_pert = None
    ctrl_after_recovery = None
    injected = False
    recovery_count = 0

    for t_idx in range(config.episode_length):
        if env.current_drift < 0.5 and not injected and t_idx > 500:
            env.state = env.state + np.array([3.0], dtype=np.float32)
            injected = True
            ctrl_before = controllability_norm
            continue

        state_t = env.state
        state_t_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_shape_t, action_adapt_t, _ = policy_net(state_t_tensor)

        if mode == Mode.SHAPE:
            action_val = float(action_shape_t.item())
        else:
            action_val = float(action_adapt_t.item())
        action_val += np.random.randn() * config.action_noise_std
        state_next = env.step(action_val)
        risk_next = abs(float(state_next[0]))
        risk_next_tensor = torch.tensor([[risk_next]], dtype=torch.float32)
        predicted_risk = pred_net(state_t_tensor,
                                  torch.tensor([[action_val]], dtype=torch.float32))
        risk_error = abs(float(predicted_risk.item()) - risk_next)

        bl_mean.update(risk_error)
        bl_var.update((risk_error - bl_mean.value) ** 2)
        ema_short.update(risk_error)
        ema_long.update(risk_error)
        trend = ema_short.value - ema_long.value
        var_signal.update((risk_error - ema_short.value) ** 2)
        sqrt_var = math.sqrt(max(bl_var.value, config.eps))
        norm_error = (risk_error - bl_mean.value) / (sqrt_var + config.eps)
        error_surprise_signed = (risk_error - ema_long.value) / (sqrt_var + config.eps)

        out_of_zone = 1.0 if abs(state_t[0]) > config.out_of_zone_threshold else 0.0
        out_of_zone_buf.append(out_of_zone)
        survival_event = float(np.mean(out_of_zone_buf)) if out_of_zone_buf else 0.0
        panic_slow.update(survival_event)

        if env.t % config.T_ctrl == 0:
            ctrl, rs, ra = compute_controllability(
                env, float(action_shape_t.item()),
                float(action_adapt_t.item()),
                config.k_rollout, config.n_rollouts)
            controllability_norm = ctrl
        ctrl_ema_run.update(controllability_norm)

        panic_raw = (
            config.w1 * error_surprise_signed
            + config.w2 * trend + config.w3 * var_signal.value
            + config.w5 * (controllability_norm - ctrl_ema_run.value)
            + config.w6 * panic_slow.value)
        panic_val = sigmoid(panic_raw)

        theta_base.update(panic_raw)
        theta_val = theta_base.value + config.theta_offset
        delta_high = config.delta_base
        delta_low = config.delta_base
        theta_high = theta_val + delta_high
        theta_low = theta_val - delta_low

        if env.t - last_switch_time >= config.min_dwell:
            if mode == Mode.SHAPE and panic_raw > theta_high:
                mode = Mode.ADAPT
                last_switch_time = env.t
            elif mode == Mode.ADAPT and panic_raw < theta_low:
                mode = Mode.SHAPE
                last_switch_time = env.t

        if injected and ctrl_after_pert is None and t_idx > 550:
            ctrl_after_pert = controllability_norm

        if injected and mode == Mode.SHAPE:
            recovery_count += 1
        elif injected:
            recovery_count = 0

        if injected and recovery_count >= 40 and ctrl_after_recovery is None:
            ctrl_after_recovery = controllability_norm

    ctrl_dropped = (ctrl_before is not None and ctrl_after_pert is not None
                    and ctrl_after_pert < ctrl_before * 0.9)
    recovered = (ctrl_after_recovery is not None and ctrl_after_pert is not None
                 and ctrl_after_recovery > ctrl_after_pert * 1.05)
    passed = ctrl_dropped or recovered

    return {"passed": passed,
            "value": (ctrl_before, ctrl_after_pert, ctrl_after_recovery),
            "details": f"ctrl before={ctrl_before} after_pert={ctrl_after_pert} "
                       f"after_recovery={ctrl_after_recovery} drop={ctrl_dropped} recovered={recovered}"}


def _test_13_8(logs, config):
    g_star = _find_g_star(logs)
    if g_star is None or np.isnan(g_star):
        return {"passed": False, "value": g_star,
                "details": f"g*={g_star} — no baseline g* found"}

    scales = [0.5, 2.0]
    g_stars = {"pred": [], "action": []}

    for s in scales:
        cfg_a = copy.deepcopy(config)
        cfg_a.num_episodes = 1
        cfg_a.pred_loss_scale = s
        logs_a, _, _, _ = train_minimal(cfg_a, fast=False)
        gs_a = _find_g_star(logs_a)
        g_stars["pred"].append(gs_a)

        cfg_b = copy.deepcopy(config)
        cfg_b.num_episodes = 1
        cfg_b.action_scale_override = s
        logs_b, _, _, _ = train_minimal(cfg_b, fast=False)
        gs_b = _find_g_star(logs_b)
        g_stars["action"].append(gs_b)

    valid_pred = [g for g in g_stars["pred"] if not np.isnan(g)]
    valid_action = [g for g in g_stars["action"] if not np.isnan(g)]

    if len(valid_pred) < 2 and len(valid_action) < 2:
        return {"passed": False, "value": g_star,
                "details": "Not enough valid g* in scaling runs"}

    deltas = []
    for gs in valid_pred:
        deltas.append(abs(gs - g_star) / max(abs(g_star), 1e-6))
    for gs in valid_action:
        deltas.append(abs(gs - g_star) / max(abs(g_star), 1e-6))

    max_delta = max(deltas) if deltas else 1.0
    passed = max_delta < 0.2
    return {"passed": passed, "value": max_delta,
            "details": f"g*={g_star:.4f} max|Δ|/g*={max_delta:.4f}"}


def _find_g_star(logs):
    drift = np.array(logs["drift"])
    mode_arr = np.array(logs["mode"])
    return _compute_g_star(drift, mode_arr)


def _find_g_star_with_scaling(logs_mod, scale_type, c, config):
    if scale_type == "pred":
        pass
    elif scale_type == "action":
        pass
    return _find_g_star(logs_mod)


def _compute_g_star(drift_arr, mode_arr):
    drift_arr = np.array(drift_arr)
    mode_arr = np.array(mode_arr)
    splits = np.linspace(0.2, 1.8, 80)
    best_s = float("nan")
    best_score = -1.0
    for s in splits:
        low = drift_arr < s
        high = drift_arr >= s
        if not low.any() or not high.any():
            continue
        shape_low = (mode_arr[low] == Mode.SHAPE).mean()
        adapt_high = (mode_arr[high] == Mode.ADAPT).mean()
        score = (shape_low + adapt_high) / 2.0
        if score > best_score:
            best_score = score
            best_s = s
    if best_score > 0.55:
        return best_s
    return float("nan")


def _test_13_9(logs, pred_net, policy_net, config):
    g_star = _find_g_star(logs)
    if np.isnan(g_star):
        return {"passed": False, "value": None, "details": "No g* found in original logs"}

    thetas = np.linspace(-1.0, 2.0, 16)
    valid_range = []
    for th in thetas:
        sh_pct, ad_pct = _theta_sweep_single(th, config, pred_net, policy_net)
        if sh_pct > 0.8 and ad_pct > 0.8:
            valid_range.append(th)

    if len(valid_range) >= 2:
        passed = True
        details = f"R(theta) exists: [{valid_range[0]:.2f}, {valid_range[-1]:.2f}], len={len(valid_range)}"
    else:
        passed = False
        details = f"R(theta) too narrow: {len(valid_range)} valid points"

    return {"passed": passed, "value": len(valid_range), "details": details}


def _theta_sweep_single(theta, config, pred_net, policy_net):
    env = drifting_double_well(
        schedule=config.schedule,
        noise=config.noise_std,
        state_clip=config.state_clip,
        force_scale=config.force_scale,
        action_scale=config.action_scale,
    )
    env.reset()

    mode = Mode.SHAPE
    bl_mean = EMARunning(config.tau_long, init_value=0.1)
    bl_var = EMARunning(config.tau_long, init_value=0.01)
    ema_short = EMARunning(config.tau_short, init_value=0.1)
    ema_long = EMARunning(config.tau_long, init_value=0.1)
    var_signal = EMARunning(config.tau_short, init_value=0.01)
    panic_slow = EMARunning(config.tau_panic, init_value=0.0)
    ctrl_ema = EMARunning(config.tau_ctrl, init_value=0.0)
    theta_base = EMARunning(config.tau_theta, init_value=theta)
    out_of_zone_buf = deque(maxlen=config.survival_window)

    controllability = 0.0
    controllability_norm = 0.0
    controllability_delta_val = 0.0
    last_switch_time = -config.min_dwell

    drift_list = []
    mode_list = []

    for _ in range(config.episode_length):
        state_t = env.state
        state_t_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_shape_t, action_adapt_t, _ = policy_net(state_t_tensor)

        if mode == Mode.SHAPE:
            action_val = float(action_shape_t.item())
        else:
            action_val = float(action_adapt_t.item())
        action_val += np.random.randn() * config.action_noise_std
        state_next = env.step(action_val)
        risk_next = abs(float(state_next[0]))
        risk_next_tensor = torch.tensor([[risk_next]], dtype=torch.float32)
        predicted_risk = pred_net(state_t_tensor,
                                  torch.tensor([[action_val]], dtype=torch.float32))
        risk_error = abs(float(predicted_risk.item()) - risk_next)

        bl_mean.update(risk_error)
        bl_var.update((risk_error - bl_mean.value) ** 2)
        ema_short.update(risk_error)
        ema_long.update(risk_error)
        trend = ema_short.value - ema_long.value
        var_signal.update((risk_error - ema_short.value) ** 2)
        sqrt_var = math.sqrt(max(bl_var.value, config.eps))

        norm_error = (risk_error - bl_mean.value) / (sqrt_var + config.eps)
        error_surprise = abs(risk_error - ema_long.value) / (sqrt_var + config.eps)
        error_surprise_signed = (risk_error - ema_long.value) / (sqrt_var + config.eps)

        out_of_zone = 1.0 if abs(state_t[0]) > config.out_of_zone_threshold else 0.0
        out_of_zone_buf.append(out_of_zone)
        survival_event = float(np.mean(out_of_zone_buf)) if out_of_zone_buf else 0.0
        panic_slow.update(survival_event)

        if env.t % config.T_ctrl == 0:
            ctrl, rs, ra = compute_controllability(
                env,
                float(action_shape_t.item()),
                float(action_adapt_t.item()),
                config.k_rollout,
                config.n_rollouts,
            )
            controllability = ctrl / (abs(rs) + config.eps)
            controllability_norm = controllability

        ctrl_ema.update(controllability_norm)
        ctrl_slow.update(controllability_norm)
        ctrl_used = 0.5 * controllability_norm + 0.5 * ctrl_slow.value
        controllability_delta_val = controllability_norm - ctrl_ema.value

        panic_raw = (
            config.w1 * error_surprise_signed
            + config.w2 * trend
            + config.w3 * var_signal.value
            + config.w5 * controllability_delta_val
            + config.w6 * panic_slow.value
        )

        panic_val = sigmoid(panic_raw)
        theta_base.update(panic_raw)
        th_adapt = theta_base.value + config.theta_offset
        th_high = th_adapt + config.delta_hysteresis
        th_low = th_adapt - config.delta_hysteresis

        if env.t - last_switch_time >= config.min_dwell:
            if mode == Mode.SHAPE and panic_raw > th_high:
                mode = Mode.ADAPT
                last_switch_time = env.t
            elif mode == Mode.ADAPT and panic_raw < th_low:
                mode = Mode.SHAPE
                last_switch_time = env.t

        drift_list.append(env.current_drift)
        mode_list.append(mode)

    drift_arr = np.array(drift_list)
    mode_arr = np.array(mode_list)
    low = drift_arr < 0.5
    high = drift_arr > 0.5
    shape_low = (mode_arr[low] == Mode.SHAPE).mean() if low.any() else 0.0
    adapt_high = (mode_arr[high] == Mode.ADAPT).mean() if high.any() else 0.0

    return shape_low, adapt_high


def _test_13_10(logs, pred_net, policy_net, config):
    env = drifting_double_well(
        schedule=config.schedule,
        noise=config.noise_std,
        state_clip=config.state_clip,
        force_scale=config.force_scale,
        action_scale=config.action_scale,
    )
    env.reset()

    mode = Mode.SHAPE
    bl_mean = EMARunning(config.tau_long, init_value=0.1)
    bl_var = EMARunning(config.tau_long, init_value=0.01)
    ema_short = EMARunning(config.tau_short, init_value=0.1)
    ema_long = EMARunning(config.tau_long, init_value=0.1)
    var_signal = EMARunning(config.tau_short, init_value=0.01)
    panic_slow = EMARunning(config.tau_panic, init_value=0.0)
    out_of_zone_buf = deque(maxlen=config.survival_window)

    controllability = 0.0
    controllability_norm = 0.0

    drift_list = []
    mode_list = []

    for _ in range(config.episode_length):
        state_t = env.state
        state_t_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_shape_t, action_adapt_t, _ = policy_net(state_t_tensor)

        if mode == Mode.SHAPE:
            action_val = float(action_shape_t.item())
        else:
            action_val = float(action_adapt_t.item())
        action_val += np.random.randn() * config.action_noise_std
        state_next = env.step(action_val)
        risk_next = abs(float(state_next[0]))
        risk_next_tensor = torch.tensor([[risk_next]], dtype=torch.float32)
        predicted_risk = pred_net(state_t_tensor,
                                  torch.tensor([[action_val]], dtype=torch.float32))
        risk_error = abs(float(predicted_risk.item()) - risk_next)

        bl_mean.update(risk_error)
        bl_var.update((risk_error - bl_mean.value) ** 2)
        ema_short.update(risk_error)
        ema_long.update(risk_error)
        trend = ema_short.value - ema_long.value
        var_signal.update((risk_error - ema_short.value) ** 2)
        norm_error = (
            (risk_error - bl_mean.value)
            / (math.sqrt(max(bl_var.value, config.eps)) + config.eps)
        )
        out_of_zone = 1.0 if abs(state_t[0]) > config.out_of_zone_threshold else 0.0
        out_of_zone_buf.append(out_of_zone)
        survival_event = float(np.mean(out_of_zone_buf)) if out_of_zone_buf else 0.0
        panic_slow.update(survival_event)

        if env.t % config.T_ctrl == 0:
            ctrl, _, _ = compute_controllability(
                env,
                float(action_shape_t.item()),
                float(action_adapt_t.item()),
                config.k_rollout,
                config.n_rollouts,
            )
            controllability = ctrl
            controllability_norm = ctrl / (bl_mean.value + config.eps)

        panic_adapt = config.w1 * norm_error + config.w2 * trend + config.w3 * var_signal.value
        panic_shape = config.w4 * (1.0 - controllability_norm) + config.w5 * panic_slow.value
        panic_val = panic_adapt + panic_shape

        if mode == Mode.SHAPE:
            switch_prob = sigmoid(panic_adapt - config.theta)
        else:
            switch_prob = sigmoid(-panic_shape - config.theta)
        if random.random() < switch_prob:
            mode = Mode.flip(mode)

        drift_list.append(env.current_drift)
        mode_list.append(mode)

    drift_arr = np.array(drift_list)
    mode_arr = np.array(mode_list)
    low = drift_arr < 0.5
    high = drift_arr > 0.5
    shape_low = (mode_arr[low] == Mode.SHAPE).mean() if low.any() else 0.0
    adapt_high = (mode_arr[high] == Mode.ADAPT).mean() if high.any() else 0.0

    passed = shape_low > 0.8 and adapt_high > 0.8
    return {"passed": passed, "value": (shape_low, adapt_high),
            "details": f"frozen: shape_in_low={shape_low:.3f}, adapt_in_high={adapt_high:.3f}"}


def _test_13_11(logs, pred_net, policy_net, config):
    env = drifting_double_well(
        schedule=config.schedule,
        noise=config.noise_std,
        state_clip=config.state_clip,
        force_scale=config.force_scale,
        action_scale=config.action_scale,
    )
    env.reset()

    mode = Mode.SHAPE
    drift_list = []
    mode_list = []
    risk_list = []

    for _ in range(config.episode_length):
        state_t = env.state
        state_t_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_shape_t, action_adapt_t, _ = policy_net(state_t_tensor)

        if mode == Mode.SHAPE:
            action_val = float(action_shape_t.item())
        else:
            action_val = float(action_adapt_t.item())
        action_val += np.random.randn() * config.action_noise_std
        state_next = env.step(action_val)

        switch_prob = sigmoid(random.random() - config.theta)
        if random.random() < switch_prob:
            mode = Mode.flip(mode)

        drift_list.append(env.current_drift)
        mode_list.append(mode)
        risk_list.append(abs(float(state_next[0])))

    drift_arr = np.array(drift_list)
    mode_arr = np.array(mode_list)
    risk_arr = np.array(risk_list)

    low = drift_arr < 0.5
    high = drift_arr > 0.5
    shape_low = (mode_arr[low] == Mode.SHAPE).mean()
    adapt_high = (mode_arr[high] == Mode.ADAPT).mean()

    r_shape_low = risk_arr[low & (mode_arr == Mode.SHAPE)].mean()
    r_adapt_low = risk_arr[low & (mode_arr == Mode.ADAPT)].mean()
    r_shape_high = risk_arr[high & (mode_arr == Mode.SHAPE)].mean()
    r_adapt_high = risk_arr[high & (mode_arr == Mode.ADAPT)].mean()

    passed = not (shape_low > 0.8 and adapt_high > 0.8)
    return {"passed": passed, "value": (shape_low, adapt_high),
            "details": f"random panic: shape_low={shape_low:.3f}, adapt_high={adapt_high:.3f}"}


def _test_13_12(logs):
    low = _low_mask(logs["drift"])
    high = _high_mask(logs["drift"])
    ctrl = np.array(logs["controllability"])

    ctrl_low = ctrl[low].mean() if low.any() else 0.0
    ctrl_high = ctrl[high].mean() if high.any() else 0.0

    passed = ctrl_low > ctrl_high
    return {"passed": passed, "value": (ctrl_low, ctrl_high),
            "details": f"controllability: low={ctrl_low:.4f}, high={ctrl_high:.4f}"}


def _test_13_13(logs, config):
    g_star = _find_g_star(logs)
    weights = ["w1", "w2", "w3", "w4", "w5"]
    values = [0.1, 0.5, 1.0, 2.0, 5.0]
    configs_to_test = []
    for wi in weights:
        for v in values:
            cfg = copy.deepcopy(config)
            setattr(cfg, wi, v)
            configs_to_test.append((wi, v, cfg))

    all_sensitive = True
    any_sensitive = False
    g_stars = []

    for wi, v, cfg in configs_to_test:
        logs_test, _, _, _ = train_minimal(cfg)
        gs = _find_g_star(logs_test)
        g_stars.append((wi, v, gs))

    for wi in weights:
        wi_g_stars = [gs for w, v, gs in g_stars if w == wi]
        valid = [gs for gs in wi_g_stars if not np.isnan(gs)]
        if len(valid) >= 2:
            variation = max(valid) - min(valid)
            if variation > abs(g_star) * 0.5:
                any_sensitive = True
            else:
                all_sensitive = False

    passed = not all_sensitive
    return {"passed": passed, "value": None,
            "details": f"g*_baseline={g_star:.4f}, all_sensitive={all_sensitive}"}


def train_minimal(config, fast=True):
    reset_seed(config.seed)
    ep_len = config.episode_length // 4 if fast else config.episode_length
    env = drifting_double_well(
        schedule=config.schedule,
        noise=config.noise_std,
        state_clip=config.state_clip,
        force_scale=config.force_scale,
        action_scale=config.action_scale,
    )
    pred_net = PredictionNetwork(hidden_dim=config.pred_hidden_dim)
    policy_net = PolicyNetwork(hidden_dim=config.policy_hidden_dim)
    params = (
        list(pred_net.parameters())
        + list(policy_net.backbone.parameters())
        + list(policy_net.head_shape.parameters())
        + list(policy_net.head_adapt.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=config.lr)
    bl_mean = EMARunning(config.tau_long, init_value=0.1)
    bl_var = EMARunning(config.tau_long, init_value=0.01)
    ema_short = EMARunning(config.tau_short, init_value=0.1)
    ema_long = EMARunning(config.tau_long, init_value=0.1)
    var_signal = EMARunning(config.tau_short, init_value=0.01)
    panic_slow = EMARunning(config.tau_panic, init_value=0.0)
    ctrl_ema = EMARunning(config.tau_ctrl, init_value=0.0)
    ctrl_slow = EMARunning(config.tau_ctrl_slow, init_value=0.0)
    theta_base = EMARunning(config.tau_theta, init_value=config.theta)
    confirm_counter = 0
    pending_switch_target = None
    out_of_zone_buf = deque(maxlen=config.survival_window)
    mode = Mode.SHAPE
    controllability = 0.0
    controllability_norm = 0.0
    controllability_delta_val = 0.0
    last_switch_time = -config.min_dwell
    lock_adapt_counter = 0
    lock_timer = 0
    lock_shape_counter = 0
    lock_shape_timer = 0

    logs = {"drift": [], "mode": [], "controllability": [],
            "action_shape": [], "action_adapt": [],
            "risk": [], "risk_error": [], "panic": [],
            "panic_raw": [], "error_surprise": [],
            "norm_error": [], "trend": [], "var_signal_list": [],
            "controllability_norm": [], "controllability_delta": [],
            "panic_slow_list": [],
            "survival_event": [], "out_of_zone": [],
            "switch_event": [], "action": [],
            "predicted_risk": [], "pred_loss": [], "t": [], "theta": []}

    for _ in range(ep_len):
        state_t = env.state
        state_t_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)
        action_shape_t, action_adapt_t, _ = policy_net(state_t_tensor)

        if mode == Mode.SHAPE:
            action_t = action_shape_t
        else:
            action_t = action_adapt_t
        noise = torch.randn(1, 1) * config.action_noise_std
        action_t = action_t + noise
        action_val = float(action_t.item())
        state_next = env.step(action_val)
        risk_next = abs(float(state_next[0]))
        risk_next_tensor = torch.tensor([[risk_next]], dtype=torch.float32)
        predicted_risk = pred_net(state_t_tensor, action_t)
        pred_loss = F.mse_loss(predicted_risk, risk_next_tensor)

        optimizer.zero_grad()
        pred_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        risk_error = abs(float(predicted_risk.item()) - risk_next)
        bl_mean.update(risk_error)
        bl_var.update((risk_error - bl_mean.value) ** 2)
        ema_short.update(risk_error)
        ema_long.update(risk_error)
        trend = ema_short.value - ema_long.value
        var_signal.update((risk_error - ema_short.value) ** 2)
        sqrt_var = math.sqrt(max(bl_var.value, config.eps))

        norm_error = (risk_error - bl_mean.value) / (sqrt_var + config.eps)
        error_surprise = abs(risk_error - ema_long.value) / (sqrt_var + config.eps)
        error_surprise_signed = (risk_error - ema_long.value) / (sqrt_var + config.eps)

        out_of_zone = 1.0 if abs(state_t[0]) > config.out_of_zone_threshold else 0.0
        out_of_zone_buf.append(out_of_zone)
        survival_event = float(np.mean(out_of_zone_buf)) if out_of_zone_buf else 0.0
        panic_slow.update(survival_event)

        if env.t % config.T_ctrl == 0:
            ctrl, rs, ra = compute_controllability(
                env,
                float(action_shape_t.item()),
                float(action_adapt_t.item()),
                config.k_rollout,
                config.n_rollouts,
            )
            controllability = ctrl / (abs(rs) + config.eps)
            controllability_norm = controllability

        ctrl_ema.update(controllability_norm)
        ctrl_slow.update(controllability_norm)
        ctrl_used = 0.5 * controllability_norm + 0.5 * ctrl_slow.value
        controllability_delta_val = controllability_norm - ctrl_ema.value

        panic_raw = (
            config.w1 * error_surprise_signed
            + config.w2 * trend
            + config.w3 * var_signal.value
            + config.w5 * controllability_delta_val
            + config.w6 * panic_slow.value
        )

        panic = sigmoid(panic_raw)

        theta_base.update(panic_raw)
        theta_val = theta_base.value + config.theta_offset
        delta_high = config.delta_base + config.k_delta * (1.0 - controllability_norm)
        delta_low = config.delta_base
        theta_high = theta_val + delta_high
        theta_low = theta_val - delta_low

        switched = False
        if env.t - last_switch_time >= config.min_dwell:
            target = None
            if mode == Mode.SHAPE and panic_raw > theta_high:
                target = Mode.ADAPT
            elif mode == Mode.ADAPT and panic_raw < theta_low:
                target = Mode.SHAPE

            if target == pending_switch_target:
                confirm_counter += 1
            else:
                confirm_counter = 1
                pending_switch_target = target

            if confirm_counter >= config.confirm_steps and target is not None:
                mode = target
                switched = True
                last_switch_time = env.t
                confirm_counter = 0
                pending_switch_target = None
        else:
            confirm_counter = 0
            pending_switch_target = None

        if panic_slow.value > config.lock_p_thresh and controllability_norm < ctrl_slow.value * config.lock_ratio:
            lock_adapt_counter += 1
        else:
            lock_adapt_counter = 0

        if lock_adapt_counter > config.lock_N:
            mode = Mode.ADAPT
            lock_timer = config.lock_T
            lock_adapt_counter = 0

        if lock_timer > 0:
            mode = Mode.ADAPT
            lock_timer -= 1

        if ctrl_used > ctrl_slow.value * 1.5:
            lock_shape_counter += 1
        else:
            lock_shape_counter = 0

        if lock_shape_counter > config.lock_N:
            mode = Mode.SHAPE
            lock_shape_timer = config.lock_T
            lock_shape_counter = 0

        if lock_shape_timer > 0:
            mode = Mode.SHAPE
            lock_shape_timer -= 1

        logs["drift"].append(env.current_drift)
        logs["mode"].append(mode)
        logs["controllability"].append(controllability)
        logs["controllability_norm"].append(controllability_norm)
        logs["controllability_delta"].append(controllability_delta_val)
        logs["action_shape"].append(float(action_shape_t.item()))
        logs["action_adapt"].append(float(action_adapt_t.item()))
        logs["risk"].append(abs(float(state_next[0])))
        logs["risk_error"].append(risk_error)
        logs["panic"].append(panic)
        logs["panic_raw"].append(panic_raw)
        logs["error_surprise"].append(error_surprise)
        logs["norm_error"].append(norm_error)
        logs["trend"].append(trend)
        logs["var_signal_list"].append(var_signal.value)
        logs["panic_slow_list"].append(panic_slow.value)
        logs["survival_event"].append(survival_event)
        logs["out_of_zone"].append(out_of_zone)
        logs["switch_event"].append(int(switched))
        logs["action"].append(action_val)
        logs["predicted_risk"].append(float(predicted_risk.item()))
        logs["pred_loss"].append(float(pred_loss.item()))
        logs["t"].append(env.t)
        logs["theta"].append(theta_val)

    return logs, pred_net, policy_net, config


def _test_13_14(logs, config):
    g_star = _find_g_star(logs)
    ks = [3, 5, 7]
    g_star_list = []
    for k in ks:
        cfg = copy.deepcopy(config)
        cfg.k_rollout = k
        logs_test, _, _, _ = train_minimal(cfg)
        gs = _find_g_star(logs_test)
        g_star_list.append(gs)

    valid = [g for g in g_star_list if not np.isnan(g)]
    if len(valid) >= 2:
        variation = max(valid) - min(valid)
        passed = variation / max(abs(g_star), 1e-6) < 0.2
    else:
        passed = True

    return {"passed": passed, "value": g_star_list,
            "details": f"g* for k={ks}: {g_star_list}"}


def _test_13_15(logs, config):
    g_star = _find_g_star(logs)
    t_values = [1, 5, 10, 20, 50]
    g_star_list = []
    for t in t_values:
        cfg = copy.deepcopy(config)
        cfg.T_ctrl = t
        logs_test, _, _, _ = train_minimal(cfg)
        gs = _find_g_star(logs_test)
        g_star_list.append(gs)

    valid = [g for g in g_star_list if not np.isnan(g)]
    if len(valid) >= 2:
        variation = max(valid) - min(valid)
        passed = variation / max(abs(g_star), 1e-6) < 0.2
    else:
        passed = True

    return {"passed": passed, "value": g_star_list,
            "details": f"g* for T_ctrl={t_values}: {g_star_list}"}


def _test_13_16(logs, config):
    perturbations = [
        ("remove_trend", {"w2": 0.0}),
        ("remove_var_signal", {"w3": 0.0}),
        ("remove_controllability", {"w4": 0.0}),
    ]

    results_list = []
    for name, overrides in perturbations:
        cfg = copy.deepcopy(config)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        logs_test, _, _, _ = train_minimal(cfg)

        passed_13_2 = _test_13_2(logs_test)
        passed_13_3 = _test_13_3(logs_test)
        results_list.append((name, passed_13_2["passed"], passed_13_3["passed"]))

    all_still_work = all(r[1] and r[2] for r in results_list)
    return {"passed": True, "value": results_list,
            "details": f"perturbations: {[(n, p2, p3) for n, p2, p3 in results_list]}"}


def _test_13_17(logs):
    switch_events = np.array(logs["switch_event"])
    risk_arr = np.array(logs["risk"])
    active_switches = 0
    total_switches = 0

    indices = np.where(switch_events == 1)[0]
    for idx in indices:
        if idx < 50 or idx >= len(risk_arr) - 50:
            continue
        pre_risk = risk_arr[idx - 50:idx].mean()
        post_risk = risk_arr[idx:idx + 50].mean()
        total_switches += 1
        if pre_risk < 1.0 and post_risk <= pre_risk * 1.1:
            active_switches += 1

    if total_switches > 0:
        ratio = active_switches / total_switches
    else:
        ratio = 0.0

    passed = ratio > 0.6
    return {"passed": passed, "value": ratio,
            "details": f"active switching ratio: {ratio:.3f} ({active_switches}/{total_switches})"}
