import numpy as np


class DriftingDoubleWell:
    def __init__(self, kappa=1.0, drift_rate=0.02, noise=0.1,
                 flip_period=500, state_clip=3.0, force_scale=0.1,
                 coupling_mode="weak", coupling_beta=0.8, coupling_gamma=0.5,
                 regime_visible=False):
        self.kappa = kappa
        self.drift_rate = drift_rate
        self.noise = noise
        self.flip_period = flip_period
        self.state_clip = state_clip
        self.force_scale = force_scale
        self.coupling_mode = coupling_mode
        self.coupling_beta = coupling_beta
        self.coupling_gamma = coupling_gamma
        self.regime_visible = regime_visible
        self.sign = 1.0
        self.t = 0
        self.drift = 0.0
        self.drift_history = []
        self.effective_drift = 0.0
        self.state = np.random.uniform(-1.5, 1.5, size=(1,)).astype(np.float32)

    def snapshot(self):
        return {
            "state": self.state.copy(),
            "sign": self.sign,
            "t": self.t,
            "drift": self.drift,
            "effective_drift": self.effective_drift,
            "drift_history": list(self.drift_history),
        }

    def restore(self, snap):
        self.state = snap["state"].copy()
        self.sign = snap["sign"]
        self.t = snap["t"]
        self.drift = snap["drift"]
        self.effective_drift = snap["effective_drift"]
        self.drift_history = list(snap["drift_history"])

    def reset(self):
        self.state = np.random.uniform(-1.5, 1.5, size=(1,)).astype(np.float32)
        self.t = 0
        self.drift = 0.0
        self.effective_drift = 0.0
        self.drift_history = []
        self.sign = 1.0
        return self._make_obs()

    def _flip(self):
        if self.t % self.flip_period == 0 and self.t > 0:
            self.sign *= -1.0

    def _make_obs(self):
        if self.regime_visible:
            return np.array([self.state[0], self.sign], dtype=np.float32)
        return np.array([self.state[0]], dtype=np.float32)

    def grad_potential(self, x, sign):
        b = 1.0 + self.effective_drift
        return 4.0 * x**3 - 2.0 * b * x + sign * self.kappa

    def step(self, action):
        self._flip()
        baseline_drift = self.drift_rate * self.t
        self.drift_history.append(float(baseline_drift))

        if self.coupling_mode == "drift":
            self.effective_drift = baseline_drift + self.coupling_beta * float(action)
        elif self.coupling_mode == "potential":
            self.effective_drift = baseline_drift
        else:
            self.effective_drift = baseline_drift

        x = float(np.clip(self.state[0], -self.state_clip, self.state_clip))
        force = -self.force_scale * self.grad_potential(x, self.sign)

        if self.coupling_mode == "potential":
            force = force - self.coupling_gamma * float(action) * np.sign(x)
        elif self.coupling_mode == "distribution":
            x_next = x + force + np.random.normal(self.coupling_gamma * float(action), self.noise)
            x_next = np.clip(x_next, -self.state_clip, self.state_clip)
            self.state = np.array([x_next], dtype=np.float32)
            self.t += 1
            return self._make_obs(), 0.0, False

        if self.coupling_mode == "weak":
            control = float(action) * 0.05
        else:
            control = float(action) * 0.1

        x_next = x + control + force + np.random.normal(0, self.noise)
        x_next = np.clip(x_next, -self.state_clip, self.state_clip)
        self.state = np.array([x_next], dtype=np.float32)
        self.t += 1
        return self._make_obs(), 0.0, False

    def step_deterministic(self, action):
        x = float(np.clip(self.state[0], -self.state_clip, self.state_clip))
        if self.coupling_mode == "drift":
            self.effective_drift = self.drift_rate * self.t + self.coupling_beta * float(action)
        force = -self.force_scale * self.grad_potential(x, self.sign)
        if self.coupling_mode == "potential":
            force = force - self.coupling_gamma * float(action) * np.sign(x)
        control = float(action) * 0.05 if self.coupling_mode == "weak" else float(action) * 0.1
        x_next = np.clip(x + control + force, -self.state_clip, self.state_clip)
        return np.array([x_next], dtype=np.float32)
