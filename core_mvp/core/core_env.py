import numpy as np
import copy
import torch


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
                 drift=0.5, force_scale=0.1, action_scale=0.2, seed=None,
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

    @staticmethod
    def step_batch(states, actions, rngs, d_total, k, mode='closed'):
        """Fully vectorized: step N environments in a single numpy call."""
        N = states.shape[0]
        x_ctrl = states[:, :k]
        grad = 4.0 * x_ctrl ** 3 - 2.0 * (1.0 + 0.5) * x_ctrl
        force = -0.1 * grad
        control = 0.2 * actions[:, :k]
        noise_ctrl = np.array([r.normal(0, 0.05, size=k) for r in rngs])
        if mode == 'pseudo' or mode == 'open':
            x_ctrl_next = x_ctrl + force + noise_ctrl
        else:
            x_ctrl_next = x_ctrl + force + control + noise_ctrl
        x_ctrl_next = np.clip(x_ctrl_next, -5.0, 5.0)
        if d_total > k:
            x_noise = states[:, k:]
            dx_noise = -0.5 * x_noise + np.array([rg.randn(d_total - k) for rg in rngs])
            x_noise_next = x_noise + dx_noise
            next_states = np.concatenate([x_ctrl_next, x_noise_next], axis=1).astype(np.float32)
        else:
            next_states = x_ctrl_next.astype(np.float32)
        risks = np.linalg.norm(x_ctrl_next, axis=1).astype(np.float32)
        return next_states, risks

    @staticmethod
    def step_batch_gpu(states, actions, d_total, k, mode='closed'):
        """All-GPU vectorized env step: states/actions are torch Tensors on GPU."""
        x_ctrl = states[:, :k]
        grad = 4.0 * x_ctrl ** 3 - 2.0 * (1.0 + 0.5) * x_ctrl
        force = -0.1 * grad
        control = 0.2 * actions[:, :k]
        noise_ctrl = torch.randn_like(x_ctrl) * 0.05
        if mode == 'pseudo' or mode == 'open':
            x_ctrl_next = x_ctrl + force + noise_ctrl
        else:
            x_ctrl_next = x_ctrl + force + control + noise_ctrl
        x_ctrl_next = torch.clamp(x_ctrl_next, -5.0, 5.0)
        if d_total > k:
            x_noise = states[:, k:]
            dx_noise = -0.5 * x_noise + torch.randn_like(x_noise)
            x_noise_next = x_noise + dx_noise
            next_states = torch.cat([x_ctrl_next, x_noise_next], dim=1)
        else:
            next_states = x_ctrl_next
        risks = torch.norm(x_ctrl_next, dim=1)
        return next_states.detach(), risks.detach()


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


class DiscreteEnv:
    """GridWorld with drift — discrete action space, absorbing hazard zone.

    State: (x, y) on [0, grid_size-1]².
    Target: bottom row (y = grid_size-1). Hazard zone: top rows y < hazard_row.
    Drift: with probability drift_prob, the agent slides upward (away from target).
    Risk: distance to safety (hazard_row - y), clipped ≥ 0.
    Episode ends when agent reaches target row (risk = 0).
    """

    def __init__(self, grid_size=10, drift_prob=0.1, seed=None):
        self.grid_size = grid_size
        self.drift_prob = drift_prob
        self._rng = np.random.RandomState(seed)
        self.state = np.array([grid_size // 2, grid_size // 2], dtype=np.int32)
        self.target_row = grid_size - 1
        self.hazard_row = grid_size // 3
        self.state_dim = 2
        self.action_dim = 4

    def reset(self):
        self.state = np.array([self.grid_size // 2, self.grid_size // 2], dtype=np.int32)
        return self.state.copy().astype(np.float32)

    def step(self, action):
        action = int(action)
        if self._rng.random() < self.drift_prob:
            self.state[1] = max(0, self.state[1] - 1)
        else:
            if action == 0:   self.state[1] = min(self.grid_size - 1, self.state[1] + 1)
            elif action == 1: self.state[1] = max(0, self.state[1] - 1)
            elif action == 2: self.state[0] = max(0, self.state[0] - 1)
            elif action == 3: self.state[0] = min(self.grid_size - 1, self.state[0] + 1)
        self.state = np.clip(self.state, 0, self.grid_size - 1)
        risk = max(0.0, float(self.hazard_row - self.state[1]))
        done = (risk == 0.0)
        return self.state.copy().astype(np.float32), risk, done, {}

    def set_seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def get_state(self):
        return self.state.copy().astype(np.float32)

    def save_state(self):
        return {"state": self.state.copy(), "rng_state": self._rng.get_state()}

    def restore_state(self, saved):
        self.state = saved["state"].copy()
        self._rng.set_state(saved["rng_state"])


class ImpulsiveEnv:
    """Double-well with random impulsive perturbations.

    Wraps MultiModeEnv.  At each step, with probability impulse_prob,
    adds Gaussian noise of scale impulse_scale to ALL state dimensions.
    """

    def __init__(self, d=10, k=2, impulse_prob=0.02, impulse_scale=0.5, seed=None,
                 mode="closed", **kwargs):
        self.base_env = MultiModeEnv(d_total=d, k_controlled=k, mode=mode, seed=seed, **kwargs)
        self.base_env.calibrate_noise_scales()
        self.impulse_prob = impulse_prob
        self.impulse_scale = impulse_scale
        self._rng = np.random.RandomState(seed)
        self.state_dim = d
        self.action_dim = k

    def reset(self):
        return self.base_env.reset()

    def step(self, action):
        state, risk, done, info = self.base_env.step(action)
        if self._rng.random() < self.impulse_prob:
            impulse = self._rng.randn(self.base_env.d) * self.impulse_scale
            state = state + impulse.astype(np.float32)
            self.base_env.state = state.copy()
            risk = float(np.linalg.norm(state[:self.base_env.k]))
        return state.copy(), risk, done, info

    def set_seed(self, seed):
        self._rng = np.random.RandomState(seed)
        self.base_env.set_seed(seed)

    def get_state(self):
        return self.base_env.get_state()

    def save_state(self):
        return {"base": self.base_env.save_state(), "rng": self._rng.get_state()}

    def restore_state(self, saved):
        self.base_env.restore_state(saved["base"])
        self._rng.set_state(saved["rng"])


class DriftLevelEnv:
    """Water-level control with discrete actions and continuous drift.

    State: 1D water_level ∈ [0, 1].
    Actions: 0=small_drain, 1=large_drain, 2=small_fill, 3=large_fill.
    Drift: constant upward bias (water rises over time).
    Risk: distance from safe zone [0.2, 0.8] — high near boundaries.
    """

    def __init__(self, drift=0.005, seed=None):
        self.drift = drift
        self._rng = np.random.RandomState(seed)
        self.water_level = 0.5
        self.state_dim = 1
        self.action_dim = 4

    def reset(self):
        self.water_level = 0.5
        return np.array([self.water_level], dtype=np.float32)

    def step(self, action):
        action = int(action) % 4
        if action == 0:   self.water_level -= 0.02          # small drain
        elif action == 1: self.water_level -= 0.06          # large drain
        elif action == 2: self.water_level += 0.02          # small fill
        elif action == 3: self.water_level += 0.06          # large fill
        self.water_level += self.drift                       # constant upward drift
        self.water_level += self._rng.normal(0, 0.01)        # noise
        self.water_level = np.clip(self.water_level, 0.0, 1.0)
        # risk: distance from safe center 0.5
        risk = abs(self.water_level - 0.5) * 2.0
        done = False
        return np.array([self.water_level], dtype=np.float32), float(risk), done, {}

    def set_seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def get_state(self):
        return np.array([self.water_level], dtype=np.float32)

    def save_state(self):
        return {"level": self.water_level, "rng": self._rng.get_state()}

    def restore_state(self, saved):
        self.water_level = saved["level"]
        self._rng.set_state(saved["rng"])


class ContinuousDriftLevelEnv:
    """Water-level control with continuous actions and downward drift.

    State: 1D water_level ∈ [0, 1].
    Action: continuous ∈ [-1, 1] applied as delta × action_scale.
    Drift: constant downward bias (water falls → risk increases).
    Risk: distance from safe midpoint 0.5.
    """

    def __init__(self, drift=0.02, action_scale=0.1, noise_std=0.01, seed=None):
        self.drift = drift
        self.action_scale = action_scale
        self.noise_std = noise_std
        self._rng = np.random.RandomState(seed)
        self.water_level = 0.5
        self.state_dim = 1
        self.action_dim = 1

    def reset(self):
        self.water_level = 0.5
        return np.array([self.water_level], dtype=np.float32)

    def step(self, action):
        action = float(np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()[0])
        self.water_level -= self.drift                       # constant downward drift (must pump up)
        self.water_level += self.action_scale * action        # continuous control
        self.water_level += self._rng.normal(0, self.noise_std)
        self.water_level = np.clip(self.water_level, 0.0, 1.0)
        risk = abs(self.water_level - 0.5) * 2.0             # 0 at center, 1 at edges
        done = False
        return np.array([self.water_level], dtype=np.float32), float(risk), done, {}

    def set_seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def get_state(self):
        return np.array([self.water_level], dtype=np.float32)

    def save_state(self):
        return {"level": self.water_level, "rng": self._rng.get_state()}

    def restore_state(self, saved):
        self.water_level = saved["level"]
        self._rng.set_state(saved["rng"])


class LatentShiftDoubleWellEnv:
    """Double-well with a hidden shift; agent sees x_obs, not x_true.

    Every shift_flip_interval steps, the hidden shift flips sign.
    The agent receives grad_risk = ∂risk/∂action as a direction hint.
    Causal test: blocked model (action.detach) loses this hint → collapse.
    """

    def __init__(self, shift_flip_interval=50, action_scale=0.5,
                 noise_std=0.05, drift_strength=0.1, seed=None):
        self.shift_flip_interval = shift_flip_interval
        self.action_scale = action_scale
        self.noise_std = noise_std
        self.drift_strength = drift_strength
        self._rng = np.random.RandomState(seed)
        self.shift = 0.0
        self.drift_direction = 0.0
        self.x_obs = 0.0
        self.x_true = 0.0
        self.step_count = 0
        self.state_dim = 1
        self.action_dim = 1

    def _potential(self, x):
        return (x**2 - 1)**2

    def _gradient(self, x):
        return 4 * x * (x**2 - 1)

    def reset(self):
        self.shift = float(self._rng.choice([-1.0, 1.0]))
        self.drift_direction = float(self._rng.choice([-1.0, 1.0]))
        self.x_obs = 0.0
        self.x_true = self.x_obs + self.shift
        self.step_count = 0
        return np.array([self.x_obs], dtype=np.float32)

    def step(self, action):
        action = float(np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()[0])
        self.step_count += 1
        if self.step_count % self.shift_flip_interval == 0:
            self.shift *= -1
            self.drift_direction *= -1
            self.x_true = self.x_obs + self.shift
        force = -self._gradient(self.x_true)
        drift = self.drift_strength * self.drift_direction
        self.x_true += drift + force * 0.05 + self.action_scale * action + self._rng.randn() * self.noise_std
        self.x_true = max(-5.0, min(5.0, self.x_true))
        self.x_obs = self.x_true - self.shift
        risk = float(self._potential(self.x_true))
        grad_risk = float(self._gradient(self.x_true) * self.action_scale)
        done = False
        return np.array([self.x_obs], dtype=np.float32), risk, grad_risk, done, {}

    def set_seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def get_state(self):
        return np.array([self.x_obs], dtype=np.float32)


class QuadraticDriftEnv:
    """Parabolic risk: risk = (level-0.5)² — never saturates, gradient always valid.

    Drift pulls level toward 0. Agent must maintain level near 0.5.
    Optimal steady-state action: drift / action_scale / dt = 0.02/0.1/0.1 = 2.0
    """

    def __init__(self, drift=0.02, action_scale=0.1, dt=0.1, seed=None):
        self.drift = drift
        self.action_scale = action_scale
        self.dt = dt
        self._rng = np.random.RandomState(seed)
        self.level = 0.5
        self.state_dim = 1
        self.action_dim = 1

    def reset(self):
        self.level = 0.5
        return np.array([self.level], dtype=np.float32)

    def step(self, action):
        action = float(np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()[0])
        self.level += self.action_scale * action * self.dt
        self.level -= self.drift * self.dt
        self.level += self._rng.randn() * 0.005
        risk = float((self.level - 0.5) ** 2)
        grad_risk = 2.0 * (self.level - 0.5) * self.action_scale * self.dt
        done = False
        return np.array([self.level], dtype=np.float32), risk, grad_risk, done, {}

    def set_seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def get_state(self):
        return np.array([self.level], dtype=np.float32)
