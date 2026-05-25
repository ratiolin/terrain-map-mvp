import copy
import numpy as np
import torch
import torch.optim as optim

from core.agent import Expert
from core.gating import GatingNet


class Controller:
    def __init__(self, K=4, hidden_dim=4, gating_hidden=8,
                 check_interval=200, merge_thresh=0.3, prune_thresh=0.03,
                 max_models=8, obs_dim=1):
        self.K = K
        self.hidden_dim = hidden_dim
        self.gating_hidden = gating_hidden
        self.check_interval = check_interval
        self.merge_thresh = merge_thresh
        self.prune_thresh = prune_thresh
        self.max_models = max_models
        self.cnt = 0

        self.models = []
        self.usage = []
        self.errors = []
        self.birth_step = []
        self.structure_change_history = []
        self.freeze_structure = False
        self.gating = None
        self.gating_optimizer = None
        self.weight_history = []

        self.eta = 0.0
        self.drift_log = []
        self.g_log = []
        self.advantage_log = []
        self.eta_log = []

        self.obs_dim = obs_dim
        self._last_logits = None
        self._last_z_soft = None

    def init(self):
        for _ in range(self.K):
            m = Expert(obs_dim=self.obs_dim, hidden_dim=self.hidden_dim)
            if self.models:
                with torch.no_grad():
                    for p in m.predictor.parameters():
                        p.add_(torch.randn_like(p) * 0.3)
            self.models.append(m)
        self.usage = [0] * self.K
        self.errors = [[] for _ in range(self.K)]
        self.birth_step = [0] * self.K
        self.gating = GatingNet(obs_dim=self.obs_dim, K=self.K, hidden_dim=self.gating_hidden)
        self.gating_optimizer = optim.Adam(self.gating.parameters(), lr=1e-3)
        self.weight_history = []

    def reset(self):
        if self.gating is not None:
            self.gating.reset()

    def set_eta(self, eta):
        self.eta = eta
        if self.gating is not None:
            eta_clamped = max(eta, 0.001)
            self.gating.set_temperature(1.0 / eta_clamped)
            inertia = max(0.05, min(0.99, 1.0 / (1.0 + eta_clamped * 0.1)))
            self.gating.set_inertia(inertia)
            decay = max(0.1, min(1.0, 1.0 - 0.05 * eta_clamped))
            self.gating.set_hidden_decay(decay)

    def adapt_eta(self, advantage, lr=0.01, eta_min=0.01, eta_max=10.0):
        if advantage > 0:
            self.eta += lr * advantage
        else:
            self.eta += lr * advantage * 2.0
        self.eta = np.clip(self.eta, eta_min, eta_max)
        return self.eta

    @property
    def g(self):
        return 0.0

    def step_record(self, drift, advantage):
        self.cnt += 1
        g = self.eta * drift
        self.drift_log.append(float(drift))
        self.g_log.append(float(g))
        self.advantage_log.append(float(advantage))
        self.eta_log.append(float(self.eta))

    def route(self, obs):
        s = torch.tensor(obs, dtype=torch.float32)
        z, logits = self.gating(s)
        self._last_logits = logits
        self._last_z_soft = z
        return z, logits

    def record_usage(self, model_idx):
        if model_idx < len(self.usage):
            self.usage[model_idx] += 1

    def track_error(self, model_idx, error):
        if model_idx < len(self.errors):
            self.errors[model_idx].append(error)

    def n_models(self):
        return len(self.models)

    def _param_vec(self, model):
        vecs = []
        for p in model.predictor.parameters():
            vecs.append(p.data.flatten())
        return torch.cat(vecs)

    def maybe_split(self):
        if self.freeze_structure:
            return
        if self.cnt % self.check_interval != 0:
            return

        for idx in range(len(self.models)):
            if idx >= len(self.errors) or len(self.errors[idx]) < 500:
                continue
            h = np.array(self.errors[idx])
            recent = np.mean(h[-250:])
            older = np.mean(h[-500:-250])
            stuck = abs(recent - older) < 0.01
            trend = np.polyfit(range(len(h)), h, 1)[0]
            tail_var = np.var(h[-250:])
            tail_mean = np.mean(h[-250:])

            if stuck and len(self.models) < self.max_models and abs(trend) < 1e-3 and tail_var < 0.01 and tail_mean > 0.005:
                child = copy.deepcopy(self.models[idx])
                child.optimizer = optim.Adam(child.parameters(), lr=1e-3)
                with torch.no_grad():
                    for p in child.predictor.parameters():
                        p.add_(torch.randn_like(p) * 0.3)
                self.models.append(child)
                self.usage.append(0)
                self.birth_step.append(self.cnt)
                self.errors.append([])
                self.gating.expand()
                self.gating_optimizer = optim.Adam(self.gating.parameters(), lr=1e-3)
                self.weight_history.append([])
                break

    def maybe_merge(self):
        if self.freeze_structure or len(self.models) < 2:
            return
        if self.cnt % self.check_interval != 0:
            return

        dists = []
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                d = torch.norm(self._param_vec(self.models[i]) - self._param_vec(self.models[j])).item()
                dists.append((d, i, j))
        dists.sort()

        removed = set()
        for d, i, j in dists:
            if d >= self.merge_thresh:
                break
            if i in removed or j in removed:
                continue
            with torch.no_grad():
                for pb, pj in zip(self.models[i].predictor.parameters(), self.models[j].predictor.parameters()):
                    pb.data.add_(pj.data).mul_(0.5)
            self.models[i].optimizer = optim.Adam(self.models[i].parameters(), lr=1e-3)
            removed.add(j)

        if removed:
            keep = [i for i in range(len(self.models)) if i not in removed]
            self.models = [self.models[i] for i in keep]
            self.usage = [self.usage[i] for i in keep]
            self.errors = [self.errors[i] for i in keep]
            self.birth_step = [self.birth_step[i] for i in keep]
            self.gating.shrink(keep)
            self.gating_optimizer = optim.Adam(self.gating.parameters(), lr=1e-3)

    def maybe_prune(self):
        if self.freeze_structure or len(self.models) < 2:
            return
        if self.cnt % self.check_interval != 0:
            return
        total = max(1, sum(self.usage))
        keep = []
        for i in range(len(self.models)):
            age = self.cnt - self.birth_step[i]
            is_young = age < self.check_interval * 4
            frac = self.usage[i] / total
            if frac >= self.prune_thresh or is_young or len(self.models) <= 1:
                keep.append(i)
        if len(keep) < len(self.models):
            self.models = [self.models[i] for i in keep]
            self.usage = [self.usage[i] for i in keep]
            self.errors = [self.errors[i] for i in keep]
            self.birth_step = [self.birth_step[i] for i in keep]
            self.gating.shrink(keep)
            self.gating_optimizer = optim.Adam(self.gating.parameters(), lr=1e-3)

    def log(self):
        return {
            "eta": self.eta_log.copy(),
            "drift": self.drift_log.copy(),
            "g": self.g_log.copy(),
            "advantage": self.advantage_log.copy(),
        }
