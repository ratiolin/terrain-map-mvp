import numpy as np


class DriftingDoubleWell:
    def __init__(self, kappa, drift_rate, noise=0.1, state_clip=3.0,
                 force_scale=0.1,
                 flip_mode="deterministic",
                 flip_period=500,
                 flip_prob=0.002,
                 add_context=False):
        self.kappa = kappa
        self.drift_rate = drift_rate
        self.noise = noise
        self.state_clip = state_clip
        self.force_scale = force_scale
        self.flip_mode = flip_mode
        self.flip_period = flip_period
        self.flip_prob = flip_prob
        self.add_context = add_context
        self.sign = 1.0
        self.t = 0
        self.state = self._sample().astype(np.float32)

    def _sample(self):
        return np.random.uniform(-1.5, 1.5, size=(1,))

    def reset(self):
        self.state = self._sample().astype(np.float32)
        return self._make_obs()

    def _flip(self):
        if self.flip_mode == "deterministic":
            if self.t % self.flip_period == 0 and self.t > 0:
                self.sign *= -1.0
        elif self.flip_mode == "random":
            if np.random.rand() < self.flip_prob:
                self.sign *= -1.0

    def _make_obs(self):
        if self.add_context:
            return np.array([self.state[0], self.sign], dtype=np.float32)
        return self.state.copy()

    def potential(self, x, t, sign):
        a = 1.0
        b = 1.0 + self.drift_rate * t
        return a * x**4 - b * x**2 + sign * self.kappa * x

    def grad_potential(self, x, t, sign):
        a = 1.0
        b = 1.0 + self.drift_rate * t
        return 4 * a * x**3 - 2 * b * x + sign * self.kappa

    def step(self, action):
        self._flip()
        x = float(np.clip(self.state[0], -self.state_clip, self.state_clip))
        force = -self.force_scale * self.grad_potential(x, self.t, self.sign)
        control = float(action) * 0.05
        x_next = x + control + force
        x_next += np.random.normal(0, self.noise)
        x_next = np.clip(x_next, -self.state_clip, self.state_clip)
        self.state = np.array([x_next], dtype=np.float32)
        r = -self.potential(x, self.t, self.sign)
        self.t += 1
        return self._make_obs(), r, False
