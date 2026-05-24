import copy

import numpy as np
import torch
import torch.optim as optim


class Controller:
    def should_update(self, phi):
        return True


class SlowController:
    def __init__(self):
        self.cnt = 0

    def should_update(self, e):
        self.cnt += 1
        return self.cnt % 10 == 0


class FreezeController:
    def __init__(self, freeze_at=5000):
        self.freeze_at = freeze_at
        self.cnt = 0

    def should_update(self, e):
        self.cnt += 1
        return self.cnt <= self.freeze_at


class BifurcationController:
    def __init__(self, check_interval=300, env_type="cartpole"):
        self.cnt = 0
        self.error_history = []
        self.check_interval = check_interval
        self.env_type = env_type

        self.multi_model_active = False
        self.bifurcated = False

        self.P1 = None
        self.P2 = None
        self.original_agent = None

    def should_update(self, error):
        self.cnt += 1
        self.error_history.append(error)
        return True

    def maybe_bifurcate(self, agent):
        if self.bifurcated:
            return

        if self.env_type == "cartpole":
            return

        if len(self.error_history) < 1000:
            return

        if self.cnt % self.check_interval != 0:
            return

        h = np.array(self.error_history)
        trend = np.polyfit(range(len(h)), h, 1)[0]
        tail_var = np.var(h[-500:])
        tail_mean = np.mean(h[-500:])

        recent_mean = np.mean(h[-500:])
        prev_mean = np.mean(h[-1000:-500])
        stuck = abs(recent_mean - prev_mean) < 0.001

        should_fork = (
            stuck and
            abs(trend) < 5e-4 and
            tail_var < 0.005 and
            tail_mean > 0.02
        )

        if should_fork:
            self._fork(agent)

    def _fork(self, agent):
        self.original_agent = agent

        self.P1 = copy.deepcopy(agent)
        self.P2 = copy.deepcopy(agent)

        self.P1.optimizer = optim.Adam(self.P1.parameters(), lr=1e-3)
        self.P2.optimizer = optim.Adam(self.P2.parameters(), lr=1e-3)

        with torch.no_grad():
            for p in self.P1.predictor.parameters():
                p.add_(torch.randn_like(p) * 0.01)
            for p in self.P2.predictor.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        self.multi_model_active = True
        self.bifurcated = True

    def get_active_agent(self, state):
        if state[0] >= 0:
            return self.P1
        else:
            return self.P2


class GrowthController:
    def __init__(self, check_interval=300, env_type="cartpole",
                 merge_thresh=0.3, prune_thresh=0.03, max_models=8):
        self.cnt = 0
        self.check_interval = check_interval
        self.env_type = env_type
        self.merge_thresh = merge_thresh
        self.prune_thresh = prune_thresh
        self.max_models = max_models

        self.models = []
        self.usage = []
        self.errors = []
        self.birth_step = []
        self.region_bias = []

        self.original_agent = None

    def init_models(self, agent):
        self.original_agent = agent
        m = copy.deepcopy(agent)
        m.optimizer = optim.Adam(m.parameters(), lr=1e-3)
        self.models = [m]
        self.usage = [0]
        self.errors = [[]]
        self.birth_step = [0]
        self.region_bias = [0.0]

    def init_region_bias(self):
        K = len(self.models)
        self.region_bias = [float(v) for v in torch.linspace(-1.0, 1.0, K)]

    def should_update(self, error, model_idx):
        self.cnt += 1
        return True

    def track_error(self, model_idx, error):
        if model_idx < len(self.errors):
            self.errors[model_idx].append(error)

    def track_error(self, model_idx, error):
        if model_idx < len(self.errors):
            self.errors[model_idx].append(error)

    def record_usage(self, model_idx):
        if model_idx < len(self.usage):
            self.usage[model_idx] += 1

    def get_active_agent(self, model_idx):
        return self.models[model_idx]

    def maybe_split(self):
        if self.env_type == "cartpole":
            return

        if self.cnt % self.check_interval != 0:
            return

        new_models = []
        new_biases = []
        for idx, m in enumerate(self.models):
            if idx >= len(self.errors) or len(self.errors[idx]) < 500:
                new_models.append(m)
                if idx < len(self.region_bias):
                    new_biases.append(self.region_bias[idx])
                continue

            h = np.array(self.errors[idx])
            recent = np.mean(h[-250:])
            older = np.mean(h[-500:-250])
            stuck = abs(recent - older) < 0.01
            trend = np.polyfit(range(len(h)), h, 1)[0]
            tail_var = np.var(h[-250:])
            tail_mean = np.mean(h[-250:])

            should_split = (
                stuck and
                len(self.models) < self.max_models and
                abs(trend) < 1e-3 and
                tail_var < 0.01 and
                tail_mean > 0.005
            )

            parent_bias = self.region_bias[idx] if idx < len(self.region_bias) else 0.0

            if should_split:
                child = copy.deepcopy(m)
                child.optimizer = optim.Adam(child.parameters(), lr=1e-3)
                with torch.no_grad():
                    for p in child.predictor.parameters():
                        p.add_(torch.randn_like(p) * 0.3)

                child_bias = parent_bias + float(torch.randn(1).item() * 0.05)
                new_models.append(m)
                new_models.append(child)
                new_biases.append(parent_bias)
                new_biases.append(child_bias)
            else:
                new_models.append(m)
                new_biases.append(parent_bias)

        if len(new_models) != len(self.models):
            old_len = len(self.models)
            self.models = new_models
            self.region_bias = new_biases
            self.usage = self.usage[:old_len] + [0] * (len(new_models) - old_len)
            self.birth_step = self.birth_step[:old_len] + [self.cnt] * (len(new_models) - old_len)
            while len(self.errors) < len(new_models):
                self.errors.append([])

    def _param_vec(self, model):
        vecs = []
        for p in model.predictor.parameters():
            vecs.append(p.data.flatten())
        return torch.cat(vecs)

    def maybe_merge(self):
        if len(self.models) < 2:
            return

        if self.cnt % self.check_interval != 0:
            return

        merged = []
        merged_biases = []
        used = set()

        for i in range(len(self.models)):
            if i in used:
                continue
            base = self.models[i]
            group_usage = self.usage[i]
            group_errors = self.errors[i]
            n_merged = 1
            bias_sum = self.region_bias[i] if i < len(self.region_bias) else 0.0

            for j in range(i + 1, len(self.models)):
                if j in used:
                    continue
                dist = torch.norm(self._param_vec(base) -
                                  self._param_vec(self.models[j])).item()
                if dist < self.merge_thresh:
                    with torch.no_grad():
                        for pb, pj in zip(base.predictor.parameters(),
                                         self.models[j].predictor.parameters()):
                            pb.data.add_(pj.data).mul_(0.5)
                    base.optimizer = optim.Adam(base.parameters(), lr=1e-3)
                    group_usage += self.usage[j]
                    group_errors += self.errors[j]
                    if j < len(self.region_bias):
                        bias_sum += self.region_bias[j]
                    n_merged += 1
                    used.add(j)

            merged.append(base)
            merged_biases.append(bias_sum / n_merged)
            used.add(i)

        if len(merged) < len(self.models):
            self.models = merged
            self.region_bias = merged_biases
            self.usage = [max(1, sum(self.usage) // max(1, len(self.usage)))] * len(merged)
            self.errors = [e if i < len(self.errors) else [] for i, e in enumerate(self.errors[:len(merged)])]

    def maybe_prune(self):
        if len(self.models) < 2:
            return

        if self.cnt % self.check_interval != 0:
            return

        total = max(1, sum(self.usage))
        kept = []
        kept_usage = []
        kept_errors = []
        kept_birth = []
        kept_biases = []

        for i, m in enumerate(self.models):
            age = self.cnt - self.birth_step[i]
            is_young = age < self.check_interval * 4
            frac = self.usage[i] / total
            if frac >= self.prune_thresh or len(self.models) <= 1 or is_young:
                kept.append(m)
                kept_usage.append(self.usage[i])
                if i < len(self.errors):
                    kept_errors.append(self.errors[i])
                else:
                    kept_errors.append([])
                if i < len(self.birth_step):
                    kept_birth.append(self.birth_step[i])
                else:
                    kept_birth.append(self.cnt)
                if i < len(self.region_bias):
                    kept_biases.append(self.region_bias[i])
                else:
                    kept_biases.append(0.0)

        if len(kept) < len(self.models):
            self.models = kept
            self.usage = kept_usage
            self.errors = kept_errors
            self.birth_step = kept_birth
            self.region_bias = kept_biases

    def n_models(self):
        return len(self.models)


class GatingGrowthController(GrowthController):
    def __init__(self, check_interval=300, env_type="cartpole",
                 merge_thresh=0.3, prune_thresh=0.03, max_models=8,
                 use_temporal=True, use_z=False, inertia=0.0):
        super().__init__(check_interval, env_type,
                         merge_thresh, prune_thresh, max_models)
        self.gating = None
        self.gating_optimizer = None
        self.weight_history = []
        self.use_temporal = use_temporal
        self.use_z = use_z
        self.structure_change_history = []
        self.freeze_structure = False
        self.inertia = inertia

    def init_models(self, agent):
        super().init_models(agent)
        obs_dim = agent.predictor[0].in_features - 1
        if self.use_z:
            from gating import ZGatingNet
            self.gating = ZGatingNet(obs_dim, 1, temperature=0.1, inertia=self.inertia)
        elif self.use_temporal:
            from gating import TemporalGatingNet
            self.gating = TemporalGatingNet(obs_dim, 1)
        else:
            from gating import GatingNet
            self.gating = GatingNet(obs_dim, 1)
        self.gating_optimizer = optim.Adam(self.gating.parameters(), lr=1e-3)
        self.weight_history = [[]]

    def gating_reset(self):
        if (self.use_temporal or self.use_z) and self.gating is not None:
            self.gating.reset()

    def gating_weights(self, state):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if self.use_z:
            z, z_logits, z_soft = self.gating(s)
            self._last_logits = z_logits
            self._last_z_soft = z_soft
            w = z.squeeze(0)
        elif self.use_temporal:
            w, logits = self.gating(s)
            self._last_logits = logits
            w = w.squeeze(0)
        else:
            w = self.gating(s)
            w = w.squeeze(0)
        self.weight_history[-1].append(float(w.argmax()))
        return w

    def _check_freeze(self):
        if self.freeze_structure:
            return
        if len(self.structure_change_history) >= 500 and self.cnt > 2000:
            if sum(self.structure_change_history[-500:]) == 0:
                self.freeze_structure = True

    def maybe_split(self):
        self._check_freeze()
        if self.freeze_structure:
            self.structure_change_history.append(False)
            return
        before = len(self.models)
        super().maybe_split()
        after = len(self.models)
        self.structure_change_history.append(after != before)
        if after > before and self.gating is not None:
            for _ in range(after - before):
                self.gating.expand()
                self.weight_history.append([])
            self.gating_optimizer = optim.Adam(self.gating.parameters(), lr=1e-3)

    def maybe_merge(self):
        if self.freeze_structure:
            return
        if len(self.models) < 2:
            return
        if self.cnt % self.check_interval != 0:
            return

        dists = []
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                d = torch.norm(self._param_vec(self.models[i]) -
                               self._param_vec(self.models[j])).item()
                dists.append((d, i, j))
        dists.sort()

        merged_to = {}
        removed = set()
        merged_biases = {}
        for d, i, j in dists:
            if d >= self.merge_thresh:
                break
            if i in removed or j in removed:
                continue
            target = merged_to.get(i, i)
            if j not in removed:
                with torch.no_grad():
                    for pb, pj in zip(self.models[target].predictor.parameters(),
                                     self.models[j].predictor.parameters()):
                        pb.data.add_(pj.data).mul_(0.5)
                self.models[target].optimizer = optim.Adam(
                    self.models[target].parameters(), lr=1e-3)
                if target < len(self.region_bias) and j < len(self.region_bias):
                    if target in merged_biases:
                        n = merged_biases[target][0] + 1
                        merged_biases[target] = (n, (merged_biases[target][1] * (n - 1) + self.region_bias[j]) / n)
                    else:
                        merged_biases[target] = (2, (self.region_bias[target] + self.region_bias[j]) / 2)
                removed.add(j)
                merged_to[j] = target

        changed = bool(removed)
        if removed:
            keep_indices = [i for i in range(len(self.models)) if i not in removed]
            self.models = [self.models[i] for i in keep_indices]
            self.usage = [self.usage[i] for i in keep_indices]
            self.errors = [self.errors[i] for i in keep_indices]
            self.birth_step = [self.birth_step[i] for i in keep_indices]
            self.weight_history = [self.weight_history[i] for i in keep_indices]
            new_biases = []
            for i in keep_indices:
                if i in merged_biases:
                    new_biases.append(merged_biases[i][1])
                elif i < len(self.region_bias):
                    new_biases.append(self.region_bias[i])
                else:
                    new_biases.append(0.0)
            self.region_bias = new_biases
            if self.gating is not None:
                self.gating.shrink(keep_indices)
                self.gating_optimizer = optim.Adam(self.gating.parameters(), lr=1e-3)
        self.structure_change_history.append(changed)

    def maybe_prune(self):
        if self.freeze_structure:
            return
        if len(self.models) < 2:
            return
        if self.cnt % self.check_interval != 0:
            return

        total = max(1, sum(self.usage))
        keep_indices = []
        for i in range(len(self.models)):
            age = self.cnt - self.birth_step[i]
            is_young = age < self.check_interval * 4
            frac = self.usage[i] / total
            if frac >= self.prune_thresh or is_young or len(self.models) <= 1:
                keep_indices.append(i)

        changed = len(keep_indices) < len(self.models)
        if changed:
            self.models = [self.models[i] for i in keep_indices]
            self.usage = [self.usage[i] for i in keep_indices]
            self.errors = [self.errors[i] for i in keep_indices]
            self.birth_step = [self.birth_step[i] for i in keep_indices]
            self.weight_history = [self.weight_history[i] for i in keep_indices]
            self.region_bias = [self.region_bias[i] if i < len(self.region_bias) else 0.0
                               for i in keep_indices]
            if self.gating is not None:
                self.gating.shrink(keep_indices)
                self.gating_optimizer = optim.Adam(self.gating.parameters(), lr=1e-3)
        self.structure_change_history.append(changed)
