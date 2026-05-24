import numpy as np
from collections import deque


class DriftingDoubleWell:
    def __init__(self, kappa, drift_rate=0.0, noise=0.1, state_clip=3.0,
                 force_scale=0.1,
                 flip_mode="deterministic",
                 flip_period=500,
                 flip_prob=0.002,
                 add_context=False,
                 omega=None, w1=None, w2=None, A1=1.0, A2=0.5,
                 memory_k=None, memory_alpha=None):
        self.kappa = kappa
        self.drift_rate = drift_rate
        self.noise = noise
        self.state_clip = state_clip
        self.force_scale = force_scale
        self.flip_mode = flip_mode
        self.flip_period = flip_period
        self.flip_prob = flip_prob
        self.add_context = add_context
        self.omega = omega
        self.w1 = w1 if w1 is not None else omega
        self.w2 = w2 if w2 is not None else (omega * 0.5 if omega is not None else None)
        self.A1 = A1
        self.A2 = A2
        self.memory_k = memory_k
        self.memory_alpha = memory_alpha
        self.sign = 1.0
        self.t = 0
        self.drift = 0.0
        self.drift_history = []
        self._spectral = (self.w1 is not None)
        self.state = self._sample().astype(np.float32)
        if self.memory_k is not None and self.memory_k > 1:
            self._history = deque([float(self.state[0])] * self.memory_k, maxlen=self.memory_k)
        if self.memory_alpha is not None:
            self.ema_state = float(self.state[0])

    def _sample(self):
        return np.random.uniform(-1.5, 1.5, size=(1,))

    def reset(self):
        self.state = self._sample().astype(np.float32)
        if hasattr(self, '_history'):
            self._history = deque([float(self.state[0])] * self.memory_k, maxlen=self.memory_k)
        if hasattr(self, 'ema_state'):
            self.ema_state = float(self.state[0])
        return self._make_obs()

    def _flip(self):
        if self.flip_mode == "deterministic":
            if self.t % self.flip_period == 0 and self.t > 0:
                self.sign *= -1.0
        elif self.flip_mode == "random":
            if np.random.rand() < self.flip_prob:
                self.sign *= -1.0

    def _make_obs(self):
        parts = [self.state[0]]
        if self.add_context:
            parts.append(self.sign)
        if hasattr(self, '_history'):
            parts = list(self._history)
            if self.add_context:
                parts.append(self.sign)
            return np.array(parts, dtype=np.float32)
        if hasattr(self, 'ema_state'):
            parts.append(self.ema_state)
        return np.array(parts, dtype=np.float32)

    def potential(self, x, sign):
        a = 1.0
        b = 1.0 + self.drift
        return a * x**4 - b * x**2 + sign * self.kappa * x

    def grad_potential(self, x, sign):
        a = 1.0
        b = 1.0 + self.drift
        return 4 * a * x**3 - 2 * b * x + sign * self.kappa

    def step(self, action):
        self._flip()
        if self._spectral:
            self.drift = self.A1 * np.sin(self.w1 * self.t) + self.A2 * np.sin(self.w2 * self.t)
        else:
            self.drift = self.drift_rate * self.t
        self.drift_history.append(float(self.drift))
        x = float(np.clip(self.state[0], -self.state_clip, self.state_clip))
        force = -self.force_scale * self.grad_potential(x, self.sign)
        control = float(action) * 0.05
        x_next = x + control + force
        x_next += np.random.normal(0, self.noise)
        x_next = np.clip(x_next, -self.state_clip, self.state_clip)
        self.state = np.array([x_next], dtype=np.float32)
        r = -self.potential(x, self.sign)
        self.t += 1
        if hasattr(self, '_history'):
            self._history.append(float(x_next))
        if hasattr(self, 'ema_state'):
            self.ema_state = float(self.memory_alpha * self.ema_state + (1.0 - self.memory_alpha) * x_next)
        return self._make_obs(), r, False
