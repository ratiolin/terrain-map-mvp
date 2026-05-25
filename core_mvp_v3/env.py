import numpy as np
import copy


class DriftingDoubleWellSchedule:
    def __init__(self, schedule=None, noise=0.05, state_clip=5.0,
                 force_scale=0.1, action_scale=0.1):
        if schedule is None:
            schedule = [
                (2000, (0.1, 0.3)),
                (2000, (1.0, 2.0)),
                (2000, (0.1, 0.3)),
                (2000, (1.0, 2.0)),
            ]
        self.schedule = schedule
        self.segment_lengths = [s[0] for s in schedule]
        self.drift_ranges = [s[1] for s in schedule]
        self.noise = noise
        self.state_clip = state_clip
        self.force_scale = force_scale
        self.action_scale = action_scale

        self._rng = np.random.RandomState()
        self.state = np.array([self._rng.uniform(-0.5, 0.5)], dtype=np.float32)
        self.t = 0
        self._segment_idx = 0
        self._segment_t = 0
        self.current_drift = self._sample_drift(0)

    def _sample_drift(self, segment_idx):
        idx = segment_idx % len(self.drift_ranges)
        lo, hi = self.drift_ranges[idx]
        return float(self._rng.uniform(lo, hi))

    def reset(self):
        self.state = np.array([self._rng.uniform(-0.5, 0.5)], dtype=np.float32)
        self.t = 0
        self._segment_idx = 0
        self._segment_t = 0
        self.current_drift = self._sample_drift(0)
        return self.state.copy()

    def step(self, action):
        if self._segment_t >= self.segment_lengths[self._segment_idx % len(self.segment_lengths)]:
            self._segment_idx += 1
            self._segment_t = 0
            self.current_drift = self._sample_drift(self._segment_idx)

        x = float(self.state[0])
        g = self.current_drift

        grad = 4.0 * x ** 3 - 2.0 * (1.0 + g) * x
        force = -self.force_scale * grad
        control = self.action_scale * float(action)
        noise_val = self._rng.normal(0.0, self.noise)

        x_next = x + force + control + noise_val
        x_next = np.clip(x_next, -self.state_clip, self.state_clip)

        self.state = np.array([x_next], dtype=np.float32)
        self.t += 1
        self._segment_t += 1
        return self.state.copy()

    def save_state(self):
        return {
            "state": self.state.copy(),
            "t": self.t,
            "_segment_idx": self._segment_idx,
            "_segment_t": self._segment_t,
            "current_drift": self.current_drift,
            "rng_state": self._rng.get_state(),
        }

    def restore_state(self, saved):
        self.state = saved["state"].copy()
        self.t = saved["t"]
        self._segment_idx = saved["_segment_idx"]
        self._segment_t = saved["_segment_t"]
        self.current_drift = saved["current_drift"]
        self._rng.set_state(saved["rng_state"])

    def clone(self):
        return copy.deepcopy(self)

    def set_rng_seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def get_rng_seed(self):
        return int(self._rng.get_state()[1][0])


def drifting_double_well(schedule=None, noise=0.05, state_clip=5.0,
                         force_scale=0.1, action_scale=0.1):
    return DriftingDoubleWellSchedule(
        schedule=schedule, noise=noise, state_clip=state_clip,
        force_scale=force_scale, action_scale=action_scale,
    )
