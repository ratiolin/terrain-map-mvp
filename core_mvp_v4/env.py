import numpy as np
import copy


class MultiDimDoubleWell:
    def __init__(self, d=16, k=2, theta=0.5, noise_std=0.05,
                 coupling=0.05, drift=0.5, force_scale=0.1, action_scale=0.1,
                 seed=None):
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
        self._control_var = None

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

        x_noise = self.state[self.k:].copy()
        dx_noise = -self.theta * x_noise + self._rng.randn(self.d - self.k)
        x_noise_next = x_noise + dx_noise
        if self.coupling > 0 and self.d - self.k > 0:
            coupling_dim = min(self.k, self.d - self.k)
            x_noise_next[:coupling_dim] += self.coupling * x_ctrl_next[:coupling_dim]

        self.state = np.concatenate([x_ctrl_next, x_noise_next]).astype(np.float32)
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

        x_noise = state[self.k:].copy()
        dx_noise = -self.theta * x_noise
        x_noise_next = x_noise + dx_noise
        if self.coupling > 0 and self.d - self.k > 0:
            coupling_dim = min(self.k, self.d - self.k)
            x_noise_next[:coupling_dim] += self.coupling * x_ctrl_next[:coupling_dim]

        return np.concatenate([x_ctrl_next, x_noise_next]).astype(np.float32)

    def set_drift(self, drift):
        self.drift = drift

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

    def estimate_control_variance(self, steps=500):
        saved = self.save_state()
        self.reset()
        trajectory = []
        for _ in range(steps):
            self.state[:self.k] += (
                -self.force_scale * self._double_well_grad(self.state[:self.k], self.drift)
                + self._rng.normal(0, self.noise_std, size=self.k)
            )
            self.state[:self.k] = np.clip(self.state[:self.k], -5.0, 5.0)
            trajectory.append(self.state[:self.k].copy())
        self.restore_state(saved)
        trajectory = np.array(trajectory)
        return np.var(trajectory)

    def match_noise_variance(self, steps=500):
        var_ctrl = self.estimate_control_variance(steps)
        mean_var = np.mean(var_ctrl) if hasattr(var_ctrl, '__len__') else var_ctrl
        n_noise = self.d - self.k
        if n_noise > 0:
            self._ou_sigma = np.sqrt(2 * self.theta * mean_var)
            self._noise_scale = self._ou_sigma
        else:
            self._noise_scale = 0.0
        self._control_var = mean_var
        return mean_var


class ContinuousDriftEnv:
    def __init__(self, d=4, k=2, A=1.0, T=2000, theta=0.5,
                 noise_std=0.05, force_scale=0.1, action_scale=0.1,
                 seed=None):
        self.d = d
        self.k = k
        self.A = A
        self.T = T
        self.theta = theta
        self.noise_std = noise_std
        self.force_scale = force_scale
        self.action_scale = action_scale

        self.state_dim = d
        self.action_dim = k

        self._rng = np.random.RandomState(seed)
        self.state = np.zeros(d, dtype=np.float32)
        self.t = 0
        self.control_var = None

    def _double_well_grad(self, x, g):
        return 4.0 * x ** 3 - 2.0 * (1.0 + g) * x

    def _current_drift(self):
        g = self.A * np.sin(2 * np.pi * self.t / self.T)
        return float(g)

    def reset(self):
        self.state = np.zeros(self.d, dtype=np.float32)
        self.state[:self.k] = self._rng.uniform(-0.5, 0.5, size=self.k)
        self.state[self.k:] = self._rng.randn(self.d - self.k) * 0.1
        self.t = 0
        return self.state.copy()

    def step(self, action):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()
        g = self._current_drift()

        x_ctrl = self.state[:self.k].copy()
        grad = self._double_well_grad(x_ctrl, g)
        force = -self.force_scale * grad
        control = self.action_scale * action[:self.k]
        noise_ctrl = self._rng.normal(0, self.noise_std, size=self.k)
        x_ctrl_next = x_ctrl + force + control + noise_ctrl
        x_ctrl_next = np.clip(x_ctrl_next, -5.0, 5.0)

        x_noise = self.state[self.k:].copy()
        dx_noise = -self.theta * x_noise + self._rng.randn(self.d - self.k)
        x_noise_next = x_noise + dx_noise

        self.state = np.concatenate([x_ctrl_next, x_noise_next]).astype(np.float32)
        self.t += 1

        risk = float(np.linalg.norm(x_ctrl_next))
        return self.state.copy(), risk, False, {"drift": g}

    def get_current_drift(self):
        return self._current_drift()

    def save_state(self):
        return {
            "state": self.state.copy(),
            "t": self.t,
            "rng_state": self._rng.get_state(),
        }

    def restore_state(self, saved):
        self.state = saved["state"].copy()
        self.t = saved["t"]
        self._rng.set_state(saved["rng_state"])

    def clone(self):
        return copy.deepcopy(self)

    def get_state(self):
        return self.state.copy()

    def set_seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def forward_static(self, state, action, t_offset=0):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()
        g = self.A * np.sin(2 * np.pi * t_offset / self.T)

        x_ctrl = state[:self.k].copy()
        grad = self._double_well_grad(x_ctrl, g)
        force = -self.force_scale * grad
        control = self.action_scale * action[:self.k]

        x_ctrl_next = x_ctrl + force + control
        x_ctrl_next = np.clip(x_ctrl_next, -5.0, 5.0)

        x_noise = state[self.k:].copy()
        dx_noise = -self.theta * x_noise
        x_noise_next = x_noise + dx_noise

        return np.concatenate([x_ctrl_next, x_noise_next]).astype(np.float32)


class TripleWellEnv:
    """H1: Triple-well potential with drifting stability.

    Three stable positions at x ∈ {-1.5, 0, 1.5}.
    """

    def __init__(self, d=8, k=2, theta=0.5, noise_std=0.05,
                 drift=0.5, force_scale=0.1, action_scale=0.1, seed=None):
        self.d = d
        self.k = k
        self.theta = theta
        self.noise_std = noise_std
        self.drift = drift
        self.force_scale = force_scale
        self.action_scale = action_scale
        self.state_dim = d
        self.action_dim = k
        self._rng = np.random.RandomState(seed)
        self.state = np.zeros(d, dtype=np.float32)
        self.t = 0

    def _triple_well_grad(self, x, g):
        a = 1.5
        b = 0.1
        return (4 * x * (x**2 - a**2) * (x**2 + b) +
                2 * x * (x**2 - a**2)**2) - g * x

    def reset(self):
        wells = np.array([-1.5, 0.0, 1.5])
        for i in range(min(self.k, 3)):
            self.state[i] = wells[self._rng.randint(0, 3)]
        for i in range(min(self.k, 3), self.k):
            self.state[i] = self._rng.uniform(-2.0, 2.0)
        self.state[self.k:] = self._rng.randn(self.d - self.k) * 0.1
        self.t = 0
        return self.state.copy()

    def step(self, action):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()
        x_ctrl = self.state[:self.k].copy()
        grad = self._triple_well_grad(x_ctrl, self.drift)
        force = -self.force_scale * grad
        control = self.action_scale * np.clip(action[:self.k], -1.0, 1.0)
        noise_ctrl = self._rng.normal(0, self.noise_std, size=self.k)
        x_ctrl_next = x_ctrl + force + control + noise_ctrl
        x_ctrl_next = np.clip(x_ctrl_next, -3.0, 3.0)
        x_noise = self.state[self.k:].copy()
        dx_noise = -self.theta * x_noise + self._rng.randn(self.d - self.k)
        x_noise_next = x_noise + dx_noise
        self.state = np.concatenate([x_ctrl_next, x_noise_next]).astype(np.float32)
        self.t += 1
        risk = float(np.linalg.norm(x_ctrl_next))
        return self.state.copy(), risk, False, {"drift": self.drift}

    def forward_static(self, state, action):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()
        x_ctrl = state[:self.k].copy()
        grad = self._triple_well_grad(x_ctrl, self.drift)
        force = -self.force_scale * grad
        control = self.action_scale * np.clip(action[:self.k], -1.0, 1.0)
        x_ctrl_next = x_ctrl + force + control
        x_ctrl_next = np.clip(x_ctrl_next, -3.0, 3.0)
        x_noise = state[self.k:].copy()
        dx_noise = -self.theta * x_noise
        x_noise_next = x_noise + dx_noise
        return np.concatenate([x_ctrl_next, x_noise_next]).astype(np.float32)

    def get_state(self):
        return self.state.copy()

    def save_state(self):
        return {"state": self.state.copy(), "t": self.t,
                "rng_state": self._rng.get_state(), "drift": self.drift}

    def restore_state(self, saved):
        self.state = saved["state"].copy()
        self.t = saved["t"]
        self._rng.set_state(saved["rng_state"])
        self.drift = saved["drift"]

    def clone(self):
        return copy.deepcopy(self)
