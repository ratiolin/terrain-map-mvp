import numpy as np


def potential(x):
    return (x**2 - 0.25)**2


def grad_potential(x):
    return 4 * x * (x**2 - 0.25)


class DoubleWellEnv:
    def __init__(self, noise=0.0, reset_pos=None):
        self.noise = noise
        self.reset_pos = reset_pos
        self.state = self._sample().astype(np.float32)

    def _sample(self):
        if self.reset_pos is not None:
            return np.array([self.reset_pos])
        return np.random.uniform(-1, 1, size=(1,))

    def reset(self):
        self.state = self._sample().astype(np.float32)
        return self.state

    def step(self, action):
        x = self.state[0]

        force = -0.1 * grad_potential(x)
        control = float(action) * 0.01
        x_next = x + control + force

        x_next += np.random.normal(0, self.noise)

        self.state = np.array([x_next], dtype=np.float32)

        reward = -potential(x)

        return self.state, reward, False


def triple_potential(x):
    return ((x**2 - 0.25)**2) * ((x**2 - 0.81)**2)


def triple_grad(x):
    f1 = (x**2 - 0.25)
    f2 = (x**2 - 0.81)
    df1 = 2 * x
    df2 = 2 * x
    return 2 * f1 * df1 * (f2**2) + 2 * (f1**2) * f2 * df2


class TripleWellEnv:
    def __init__(self, noise=0.0, reset_range=(-1.2, 1.2)):
        self.noise = noise
        self.reset_range = reset_range
        self.state = self._sample().astype(np.float32)

    def _sample(self):
        lo, hi = self.reset_range
        return np.random.uniform(lo, hi, size=(1,))

    def reset(self):
        self.state = self._sample().astype(np.float32)
        return self.state

    def step(self, action):
        x = self.state[0]

        force = -0.1 * triple_grad(x)
        control = float(action) * 0.01
        x_next = x + control + force

        x_next += np.random.normal(0, self.noise)

        self.state = np.array([x_next], dtype=np.float32)

        reward = -triple_potential(x)

        return self.state, reward, False
