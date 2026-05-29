import numpy as np
from scipy.special import expit as sigmoid
from copy import deepcopy


class Mode:
    SHAPE = 0
    ADAPT = 1

    @staticmethod
    def flip(mode):
        return Mode.ADAPT if mode == Mode.SHAPE else Mode.SHAPE


class EMARunning:
    def __init__(self, tau, init_value=0.0):
        self.tau = tau
        self.alpha = 1.0 / tau if tau > 0 else 1.0
        self.value = init_value

    def update(self, x):
        self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


class PanicController:
    """Panic-based mode switching for closed-loop control.

    Adapted from core_mvp_v3 experiment.py panic mechanism.
    """

    def __init__(self, action_dim=2, tau_short=100, tau_long=1000, tau_panic=100,
                 w1=2.0, w2=1.0, w3=0.5, w5=1.5, w6=1.0,
                 theta=0.5, confirm_steps=2, min_dwell=200,
                 T_ctrl=10, k_rollout=7, n_rollouts=5,
                 lock_N=20, lock_T=500):
        self.action_dim = action_dim
        self.tau_lshort = tau_short
        self.tau_long = tau_long
        self.tau_panic = tau_panic

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w5 = w5
        self.w6 = w6

        self.theta = theta
        self.theta_high = theta + 0.2
        self.theta_low = theta - 0.2
        self.confirm_steps = confirm_steps
        self.min_dwell = min_dwell

        self.T_ctrl = T_ctrl
        self.k_rollout = k_rollout
        self.n_rollouts = n_rollouts

        self.lock_N = lock_N
        self.lock_T = lock_T

        self._reset_state()

    def _reset_state(self):
        self.mode = Mode.SHAPE
        self.err_mean_ema = EMARunning(self.tau_long, 0.0)
        self.err_var_ema = EMARunning(self.tau_long, 0.0)
        self.err_short_ema = EMARunning(self.tau_lshort, 0.0)
        self.err_long_ema = EMARunning(self.tau_long, 0.0)
        self.var_short_ema = EMARunning(self.tau_lshort, 0.0)
        self.panic_slow_ema = EMARunning(self.tau_panic, 0.0)
        self.ctrl_ema = EMARunning(self.tau_long, 0.0)
        self.theta_baseline_ema = EMARunning(500, 0.0)

        self.last_err = 0.0
        self.last_panic_raw = 0.0
        self.confirm_counter = 0
        self.pending_mode = None
        self.dwell_counter = 0

        self.lock_counter = 0
        self.lock_until = 0

    def compute_controllability(self, env, action_shape, action_adapt):
        shape_risks = []
        adapt_risks = []

        saved = env.save_state()
        for _ in range(self.n_rollouts):
            env.restore_state(saved)
            s = env.get_state()
            for _ in range(self.k_rollout):
                ns, r, _, _ = env.step(action_shape)
                s = ns
            shape_risks.append(float(np.linalg.norm(s[:env.k])))

            env.restore_state(saved)
            s = env.get_state()
            for _ in range(self.k_rollout):
                ns, r, _, _ = env.step(action_adapt)
                s = ns
            adapt_risks.append(float(np.linalg.norm(s[:env.k])))

        env.restore_state(saved)

        return max(0.0, float(np.median(adapt_risks)) - float(np.median(shape_risks)))

    def update(self, env, pred_error, action_shape, action_adapt, t):
        ctrl = 0.0
        if t % self.T_ctrl == 0:
            ctrl = self.compute_controllability(env, action_shape, action_adapt)
        else:
            ctrl = self.ctrl_ema.value

        self.err_mean_ema.update(pred_error)
        self.err_var_ema.update((pred_error - self.err_mean_ema.value) ** 2)
        self.err_short_ema.update(pred_error)
        self.err_long_ema.update(pred_error)

        sq_err = pred_error ** 2
        self.var_short_ema.update((sq_err - self.err_var_ema.value) ** 2)

        trend = self.err_short_ema.value - self.err_long_ema.value

        var_signal = self.var_short_ema.value

        err_std = np.sqrt(max(self.err_var_ema.value, 1e-8))
        err_surprise = 0.0
        if err_std > 0:
            err_surprise = (pred_error - self.err_mean_ema.value) / err_std

        ctrl_delta = ctrl - self.ctrl_ema.value
        self.ctrl_ema.update(ctrl)

        self.panic_slow_ema.update(abs(err_surprise) if abs(err_surprise) > 1.0 else 0.0)

        panic_raw = (
            self.w1 * err_surprise
            + self.w2 * trend
            + self.w3 * var_signal
            + self.w5 * ctrl_delta
            + self.w6 * self.panic_slow_ema.value
        )

        panic = float(sigmoid(panic_raw))

        self.theta_baseline_ema.update(abs(panic_raw))
        baseline = self.theta_baseline_ema.value
        self.theta_high = max(0.1, baseline + 0.3)
        self.theta_low = max(0.05, baseline - 0.1)

        self.last_panic_raw = panic_raw
        self.dwell_counter += 1

        if self.dwell_counter >= self.min_dwell:
            if panic_raw > self.theta_high and self.mode == Mode.SHAPE:
                if self.pending_mode != Mode.ADAPT:
                    self.pending_mode = Mode.ADAPT
                    self.confirm_counter = 1
                else:
                    self.confirm_counter += 1
                    if self.confirm_counter >= self.confirm_steps:
                        self.mode = Mode.ADAPT
                        self.dwell_counter = 0
                        self.pending_mode = None
                        self.confirm_counter = 0
            elif panic_raw < self.theta_low and self.mode == Mode.ADAPT:
                if self.pending_mode != Mode.SHAPE:
                    self.pending_mode = Mode.SHAPE
                    self.confirm_counter = 1
                else:
                    self.confirm_counter += 1
                    if self.confirm_counter >= self.confirm_steps:
                        self.mode = Mode.SHAPE
                        self.dwell_counter = 0
                        self.pending_mode = None
                        self.confirm_counter = 0
            else:
                self.pending_mode = None
                self.confirm_counter = 0

        mode_label = "SHAPE" if self.mode == Mode.SHAPE else "ADAPT"

        return {
            "panic_raw": panic_raw,
            "panic": panic,
            "mode": mode_label,
            "ctrl": ctrl,
            "trend": float(trend),
            "var_signal": float(var_signal),
        }
