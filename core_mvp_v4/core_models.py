import torch
import torch.nn as nn
import numpy as np


def get_designed_hidden_dim(d, scale_factor=8):
    return max(32, d * scale_factor)


class ClosedLoopModel(nn.Module):
    """Action-aware model: predictor consumes [h, action] → generates gradients
    from prediction loss back into the actor, enabling genuine closed-loop control.

    Architecture:
        h = encoder(state)
        action = actor(h)
        risk_pred = predictor(cat(h, action))
    """

    def __init__(self, state_dim, hidden_dim=64, action_dim=2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        h = self.encoder(state)
        action = self.actor(h)
        risk_pred = self.predictor(torch.cat([h, action], dim=-1))
        return action, h, risk_pred

    def get_hidden(self, state):
        return self.encoder(state)

    def get_action(self, state):
        h = self.encoder(state)
        return self.actor(h)

    def act_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            action = self.get_action(s_t).squeeze(0)
        return action.numpy()

    def compute_grad_norm(self, state_batch):
        """Compute ||d(risk_pred)/d(action)|| for a batch of states.

        Takes gradient of predictor output w.r.t. the action subvector
        of its input, averaged over the batch.
        """
        was_training = self.training
        self.train()
        self.zero_grad()

        s_t = torch.from_numpy(state_batch.astype(np.float32))
        s_t.requires_grad_(False)

        h = self.encoder(s_t)
        action = self.actor(h).detach().requires_grad_(True)

        risk_pred = self.predictor(torch.cat([h, action], dim=-1))
        risk_sum = risk_pred.sum()
        risk_sum.backward(retain_graph=True)

        grad = action.grad
        if grad is None:
            self.zero_grad()
            if not was_training:
                self.eval()
            return 0.0

        grad_norms = torch.norm(grad, dim=-1)
        result = float(grad_norms.mean().item())

        self.zero_grad()
        if not was_training:
            self.eval()
        return result


class V4Model(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, action_dim=2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        h = self.encoder(state)
        action = self.actor(h)
        risk_pred = self.predictor(h)
        return action, h, risk_pred

    def get_hidden(self, state):
        return self.encoder(state)

    def get_action(self, state):
        h = self.encoder(state)
        return self.actor(h)

    def f_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            h = self.encoder(s_t).squeeze(0)
        return h.numpy()

    def act_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            action = self.get_action(s_t).squeeze(0)
        return action.numpy()

    def predict_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            _, _, pred = self(s_t)
        return pred.squeeze().numpy()

    def compute_jacobian_a_h(self, s, eps=1e-3):
        """Compute d(action) / d(hidden) at state s."""
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            h0 = self.encoder(s_t).squeeze(0)
            a0 = self.actor(h0.unsqueeze(0)).squeeze(0)
        h_dim = h0.shape[0]
        a_dim = a0.shape[0]
        J = np.zeros((a_dim, h_dim))
        for i in range(h_dim):
            e = torch.zeros(h_dim)
            e[i] = eps
            with torch.no_grad():
                h_plus = h0 + e
                a_plus = self.actor(h_plus.unsqueeze(0)).squeeze(0)
                h_minus = h0 - e
                a_minus = self.actor(h_minus.unsqueeze(0)).squeeze(0)
            J[:, i] = (a_plus - a_minus).numpy() / (2.0 * eps)
        return J

    def compute_jacobian_h_s(self, s, eps=1e-3):
        """Compute d(hidden) / d(state) at state s."""
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32))
        d = s.shape[0]
        h0 = self.f_numpy(s)
        h_dim = h0.shape[0]
        J = np.zeros((h_dim, d))
        for i in range(d):
            e = np.zeros_like(s, dtype=np.float32)
            e[i] = eps
            J[:, i] = (self.f_numpy(s + e) - self.f_numpy(s - e)) / (2.0 * eps)
        return J


class DualHeadModel(nn.Module):
    """Model with separate SHAPE and ADAPT action heads (for panic-based mode switching)."""

    def __init__(self, state_dim, hidden_dim=64, action_dim=2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_shape = nn.Linear(hidden_dim, action_dim)
        self.head_adapt = nn.Linear(hidden_dim, action_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.head_shape.weight, std=0.1)
        nn.init.normal_(self.head_adapt.weight, std=0.1)

    def forward(self, state):
        h = self.backbone(state)
        action_shape = self.head_shape(h)
        action_adapt = self.head_adapt(h)
        risk_pred = self.predictor(h)
        return action_shape, action_adapt, h, risk_pred

    def get_hidden(self, state):
        return self.backbone(state)

    def f_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            h = self.backbone(s_t).squeeze(0)
        return h.numpy()

    def act_numpy(self, s, mode="shape"):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            h = self.backbone(s_t)
            if mode == "shape":
                action = self.head_shape(h)
            else:
                action = self.head_adapt(h)
            return action.squeeze(0).numpy()
