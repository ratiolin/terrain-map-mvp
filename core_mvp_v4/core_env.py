import numpy as np
import copy


class MultiDimEnv:
    """High-dimensional environment with k controllable dimensions (double-well)
    and d-k OU noise dimensions.
    """

    def __init__(self, d=4, k=2, theta=0.5, noise_std=0.05, coupling=0.05,
                 drift=0.5, force_scale=0.1, action_scale=0.1, seed=None):
        self.d = d
        self.k = k
        self.theta = theta
        self.noise_std = noise_std
        self.coupling = coupling
        self.drift = drift
        self.force_scale = force_scale
        self.action_scale = action_scale

        self.state_dim = d
        self.action_dim = k

        self._rng = np.random.RandomState(seed)
        self.state = np.zeros(d, dtype=np.float32)
        self.t = 0
        self._ou_sigma = 1.0

    def _double_well_grad(self, x, g):
        return 4.0 * x ** 3 - 2.0 * (1.0 + g) * x

    def reset(self):
        self.state = np.zeros(self.d, dtype=np.float32)
        self.state[:self.k] = self._rng.uniform(-0.5, 0.5, size=self.k)
        self.state[self.k:] = self._rng.randn(self.d - self.k) * 0.1
        self.t = 0
        return self.state.copy()

    def step(self, action):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()
        x_ctrl = self.state[:self.k].copy()

        grad = self._double_well_grad(x_ctrl, self.drift)
        force = -self.force_scale * grad
        control = self.action_scale * action[:self.k]
        noise_ctrl = self._rng.normal(0, self.noise_std, size=self.k)
        x_ctrl_next = x_ctrl + force + control + noise_ctrl
        x_ctrl_next = np.clip(x_ctrl_next, -5.0, 5.0)

        n_noise = self.d - self.k
        if n_noise > 0:
            x_noise = self.state[self.k:].copy()
            dx_noise = -self.theta * x_noise + self._ou_sigma * self._rng.randn(n_noise)
            x_noise_next = x_noise + dx_noise
            if self.coupling > 0:
                coupling_dim = min(self.k, n_noise)
                x_noise_next[:coupling_dim] += self.coupling * x_ctrl_next[:coupling_dim]
            self.state = np.concatenate([x_ctrl_next, x_noise_next]).astype(np.float32)
        else:
            self.state = x_ctrl_next.astype(np.float32)

        self.t += 1
        risk = float(np.linalg.norm(x_ctrl_next))
        return self.state.copy(), risk, False, {"drift": self.drift}

    def forward_static(self, state, action):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()
        x_ctrl = state[:self.k].copy()

        grad = self._double_well_grad(x_ctrl, self.drift)
        force = -self.force_scale * grad
        control = self.action_scale * action[:self.k]
        x_ctrl_next = x_ctrl + force + control
        x_ctrl_next = np.clip(x_ctrl_next, -5.0, 5.0)

        n_noise = self.d - self.k
        if n_noise > 0:
            x_noise = state[self.k:].copy()
            dx_noise = -self.theta * x_noise
            x_noise_next = x_noise + dx_noise
            if self.coupling > 0:
                coupling_dim = min(self.k, n_noise)
                x_noise_next[:coupling_dim] += self.coupling * x_ctrl_next[:coupling_dim]
            return np.concatenate([x_ctrl_next, x_noise_next]).astype(np.float32)
        return x_ctrl_next.astype(np.float32)

    def set_drift(self, drift):
        self.drift = drift

    def get_info(self):
        return {"drift": self.drift, "t": self.t}

    def save_state(self):
        return {
            "state": self.state.copy(),
            "t": self.t,
            "rng_state": self._rng.get_state(),
            "drift": self.drift,
        }

    def restore_state(self, saved):
        self.state = saved["state"].copy()
        self.t = saved["t"]
        self._rng.set_state(saved["rng_state"])
        self.drift = saved["drift"]

    def clone(self):
        return copy.deepcopy(self)

    def get_state(self):
        return self.state.copy()

    def set_seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def estimate_control_variance(self, steps=1000):
        saved = self.save_state()
        self.reset()
        trajectory = []
        for _ in range(steps):
            grad = self._double_well_grad(self.state[:self.k], self.drift)
            self.state[:self.k] += (
                -self.force_scale * grad
                + self._rng.normal(0, self.noise_std, size=self.k)
            )
            self.state[:self.k] = np.clip(self.state[:self.k], -5.0, 5.0)
            trajectory.append(self.state[:self.k].copy())
        self.restore_state(saved)
        trajectory = np.array(trajectory)
        return float(np.mean(np.var(trajectory, axis=0)))

    def calibrate_noise_scales(self):
        """Ensure total noise variance ≈ total control variance."""
        var_ctrl = self.estimate_control_variance()
        n_noise = self.d - self.k
        if n_noise > 0:
            self._ou_sigma = np.sqrt(2.0 * self.theta * var_ctrl)
        else:
            self._ou_sigma = 1.0
        return var_ctrl, self._ou_sigma


class MultiModeEnv(MultiDimEnv):
    """Multi-mode environment supporting closed/open/pseudo modes.

    closed: action affects both controlled dims and noise dims (standard)
    open:   action is zeroed out, state evolves by physics + noise only
    pseudo: action is passed to env BUT does NOT affect controlled dims
            (noise dims CAN still be affected, as in closed mode)

    This enables testing whether apparent closed-loop effects come from
    causal control of the dynamics or mere statistical correlation.
    """

    VALID_MODES = ('closed', 'open', 'pseudo')

    def __init__(self, d_total, k_controlled, mode='closed',
                 theta=0.5, noise_std=0.05, coupling=0.05,
                 drift=0.5, force_scale=0.1, action_scale=0.1, seed=None,
                 pseudo_affects_noise=True):
        super().__init__(d=d_total, k=k_controlled, theta=theta,
                         noise_std=noise_std, coupling=coupling,
                         drift=drift, force_scale=force_scale,
                         action_scale=action_scale, seed=seed)
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode}")
        self.mode = mode
        self.pseudo_affects_noise = pseudo_affects_noise

    def set_mode(self, mode):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode}")
        self.mode = mode

    def get_mode(self):
        return self.mode

    def step(self, action):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()
        original_action = action.copy()

        if self.mode == 'open':
            action = np.zeros_like(action)

        x_ctrl = self.state[:self.k].copy()
        grad = self._double_well_grad(x_ctrl, self.drift)
        force = -self.force_scale * grad
        noise_ctrl = self._rng.normal(0, self.noise_std, size=self.k)

        if self.mode == 'pseudo' and not self.pseudo_affects_noise:
            action = np.zeros_like(action)

        control = self.action_scale * action[:self.k]

        if self.mode == 'pseudo':
            x_ctrl_next = x_ctrl + force + noise_ctrl
        else:
            x_ctrl_next = x_ctrl + force + control + noise_ctrl

        x_ctrl_next = np.clip(x_ctrl_next, -5.0, 5.0)

        n_noise = self.d - self.k
        noise_action = (self.action_scale * original_action[:min(self.k, n_noise)]
                        if n_noise > 0 and self.mode != 'open'
                        else np.zeros(min(self.k, n_noise)))

        if n_noise > 0:
            x_noise = self.state[self.k:].copy()
            dx_noise = -self.theta * x_noise + self._ou_sigma * self._rng.randn(n_noise)
            x_noise_next = x_noise + dx_noise
            n_affected = min(self.k, n_noise)
            x_noise_next[:n_affected] += noise_action[:n_affected]
            if self.coupling > 0:
                coupling_dim = min(self.k, n_noise)
                x_noise_next[:coupling_dim] += self.coupling * x_ctrl_next[:coupling_dim]
            self.state = np.concatenate([x_ctrl_next, x_noise_next]).astype(np.float32)
        else:
            self.state = x_ctrl_next.astype(np.float32)

        self.t += 1
        risk = float(np.linalg.norm(x_ctrl_next))
        return self.state.copy(), risk, False, {"drift": self.drift}


class DriftScheduleEnv(MultiDimEnv):
    """MultiDimEnv with a piecewise constant drift schedule."""

    def __init__(self, d=4, k=2, theta=0.5, noise_std=0.05, coupling=0.05,
                 force_scale=0.1, action_scale=0.1, seed=None,
                 schedule=None):
        super().__init__(d=d, k=k, theta=theta, noise_std=noise_std,
                         coupling=coupling, drift=0.5,
                         force_scale=force_scale, action_scale=action_scale,
                         seed=seed)
        if schedule is None:
            self.schedule = [
                (0, 2500, -0.5),
                (2500, 5000, 0.0),
                (5000, 7500, 0.5),
                (7500, 10000, 1.5),
            ]
        else:
            self.schedule = schedule

    def get_drift_at_t(self, t_override=None):
        t_val = self.t if t_override is None else t_override
        for start, end, drift_val in self.schedule:
            if start <= t_val < end:
                return drift_val
        return self.schedule[-1][2]

    def step(self, action):
        self.drift = self.get_drift_at_t()
        return super().step(action)


class ContinuousDriftEnv(MultiDimEnv):
    """MultiDimEnv with sinusoidal drift g(t) = A * sin(2*pi*t / T)."""

    def __init__(self, d=4, k=2, A=1.0, T=2000, theta=0.5,
                 noise_std=0.05, force_scale=0.1, action_scale=0.1, seed=None):
        super().__init__(d=d, k=k, theta=theta, noise_std=noise_std,
                         drift=0.0, force_scale=force_scale,
                         action_scale=action_scale, seed=seed)
        self.A = A
        self.T_period = T

    def _current_drift(self, t_override=None):
        t_val = self.t if t_override is None else t_override
        return float(self.A * np.sin(2.0 * np.pi * t_val / self.T_period))

    def step(self, action):
        self.drift = self._current_drift()
        return super().step(action)

    def forward_static(self, state, action, t_offset=0):
        orig_drift = self.drift
        self.drift = self._current_drift(t_offset)
        result = super().forward_static(state, action)
        self.drift = orig_drift
        return result


class LinearDriftEnv(MultiDimEnv):
    """MultiDimEnv with linear drift sweep g(t) = g_start + (g_end - g_start) * t / T."""

    def __init__(self, d=4, k=2, g_start=-1.0, g_end=2.0, T=10000,
                 theta=0.5, noise_std=0.05, force_scale=0.1,
                 action_scale=0.1, seed=None):
        super().__init__(d=d, k=k, theta=theta, noise_std=noise_std,
                         drift=g_start, force_scale=force_scale,
                         action_scale=action_scale, seed=seed)
        self.g_start = g_start
        self.g_end = g_end
        self.T_total = T

    def _current_drift(self):
        if self.t >= self.T_total:
            return self.g_end
        return self.g_start + (self.g_end - self.g_start) * self.t / self.T_total

    def step(self, action):
        self.drift = self._current_drift()
        return super().step(action)


class PseudoActionGenerator:
    """Fixed random linear delay mapping: action_t = W @ state_{t-1}.

    Generates actions with statistical correlation to past state but
    absolutely no causal control — the mapping is random and frozen.
    Used for pseudo-loop rollouts in Layer 1.
    """

    def __init__(self, state_dim, action_dim, seed=0, scale=0.1):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(action_dim, state_dim).astype(np.float32) * scale
        self.prev_state = np.zeros(state_dim, dtype=np.float32)
        self.action_dim = action_dim

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.prev_state = np.asarray(initial_state, dtype=np.float32).copy()
        else:
            self.prev_state = np.zeros(self.W.shape[1], dtype=np.float32)

    def step(self, state):
        action = self.W @ self.prev_state
        self.prev_state = np.asarray(state, dtype=np.float32).copy()
        return action
