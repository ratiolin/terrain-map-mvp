import numpy as np


class ControlDoubleWell:
    def __init__(self, kappa=4.0, drift_rate=0.02, noise=0.1,
                 flip_period=200, state_clip=3.0, force_scale=0.1,
                 alpha=0.1, target_state=1.0):
        self.kappa = kappa
        self.drift_rate = drift_rate
        self.noise = noise
        self.flip_period = flip_period
        self.state_clip = state_clip
        self.force_scale = force_scale
        self.alpha = alpha
        self.target_state = target_state
        self.t = 0
        self.drift = 0.0
        self.drift_history = []
        self.state = np.random.uniform(-1.5, 1.5, size=(1,)).astype(np.float32)
        self.cost_history = []
        self.in_target_zone = []

    def set_target(self, target):
        self.target_state = target

    def reset(self):
        self.state = np.random.uniform(-1.5, 1.5, size=(1,)).astype(np.float32)
        self.t = 0
        self.drift = 0.0
        self.drift_history = []
        self.cost_history = []
        self.in_target_zone = []
        return self._make_obs()

    def _make_obs(self):
        return np.array([self.state[0]], dtype=np.float32)

    def grad_potential(self, x):
        b = 1.0 + self.drift
        return 4.0 * x**3 - 2.0 * b * x + np.sign(x) * self.kappa

    def step(self, action):
        self.drift = self.drift_rate * self.t
        self.drift_history.append(float(self.drift))
        x = float(np.clip(self.state[0], -self.state_clip, self.state_clip))
        force = -self.force_scale * self.grad_potential(x)
        control = float(action) * self.alpha
        x_next = x + force + control + np.random.normal(0, self.noise)
        x_next = np.clip(x_next, -self.state_clip, self.state_clip)
        self.state = np.array([x_next], dtype=np.float32)
        cost = (x_next - self.target_state) ** 2
        self.cost_history.append(float(cost))
        self.in_target_zone.append(1 if abs(x_next - self.target_state) < 0.5 else 0)
        self.t += 1
        return self._make_obs(), float(-cost), False
