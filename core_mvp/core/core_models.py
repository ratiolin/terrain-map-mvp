import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_designed_hidden_dim(d, scale_factor=32):
    return max(128, d * scale_factor)


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
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            action = self.get_action(s_t).squeeze(0)
        return action.cpu().numpy()

    def compute_grad_norm(self, state_batch):
        """Compute ||d(risk_pred)/d(action)|| for a batch of states.

        Takes gradient of predictor output w.r.t. the action subvector
        of its input, averaged over the batch.
        """
        was_training = self.training
        self.train()
        self.zero_grad()

        s_t = torch.from_numpy(state_batch.astype(np.float32)).to(DEVICE)
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
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            h = self.encoder(s_t).squeeze(0)
        return h.cpu().numpy()

    def act_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            action = self.get_action(s_t).squeeze(0)
        return action.cpu().numpy()

    def predict_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, _, pred = self(s_t)
        return pred.squeeze().cpu().numpy()

    def compute_jacobian_a_h(self, s, eps=1e-3):
        """Compute d(action) / d(hidden) at state s."""
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            h0 = self.encoder(s_t).squeeze(0)
            a0 = self.actor(h0.unsqueeze(0)).squeeze(0)
        h_dim = h0.shape[0]
        a_dim = a0.shape[0]
        J = np.zeros((a_dim, h_dim))
        for i in range(h_dim):
            e = torch.zeros(h_dim, device=DEVICE)
            e[i] = eps
            with torch.no_grad():
                h_plus = h0 + e
                a_plus = self.actor(h_plus.unsqueeze(0)).squeeze(0)
                h_minus = h0 - e
                a_minus = self.actor(h_minus.unsqueeze(0)).squeeze(0)
            J[:, i] = (a_plus - a_minus).cpu().numpy() / (2.0 * eps)
        return J

    def compute_jacobian_h_s(self, s, eps=1e-3):
        """Compute d(hidden) / d(state) at state s."""
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).to(DEVICE)
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
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            h = self.backbone(s_t).squeeze(0)
        return h.cpu().numpy()

    def act_numpy(self, s, mode="shape"):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            h = self.backbone(s_t)
            if mode == "shape":
                action = self.head_shape(h)
            else:
                action = self.head_adapt(h)
            return action.squeeze(0).cpu().numpy()


class DiscreteClosedLoopModel(nn.Module):
    """Discrete-action closed-loop model with Gumbel-Softmax relaxation.

    Predictor consumes [h, one_hot(action)], maintaining the gradient
    pathway from risk prediction back to action selection.
    """

    def __init__(self, state_dim, hidden_dim=64, n_actions=4):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + n_actions, hidden_dim),
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

    def forward(self, state, temperature=1.0):
        h = self.encoder(state)
        logits = self.actor(h)
        if self.training:
            action_probs = F.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            idx = torch.argmax(logits, dim=-1)
            action_probs = F.one_hot(idx, num_classes=self.n_actions).float()
        risk_pred = self.predictor(torch.cat([h, action_probs], dim=-1))
        return action_probs, risk_pred, logits

    def act_idx(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            h = self.encoder(s_t)
            logits = self.actor(h)
            return int(torch.argmax(logits, dim=-1).item())

    def sample_idx(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            h = self.encoder(s_t)
            logits = self.actor(h)
            probs = F.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())


class LatentShiftModel(nn.Module):
    """Model for LatentShiftDoubleWellEnv — uses grad_risk hint for direction.

    grad_hint: scalar from env providing ∂risk/∂action.
    Predictor consumes [h, action] → gradient pathway from risk to actor.
    """

    def __init__(self, state_dim=1, hidden_dim=64, action_dim=1, use_grad_hint=True):
        super().__init__()
        self.use_grad_hint = use_grad_hint
        encoder_in = state_dim + 1 if use_grad_hint else state_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state, grad_hint=None):
        if self.use_grad_hint and grad_hint is not None:
            s_in = torch.cat([state, grad_hint.unsqueeze(-1)], dim=-1)
        else:
            s_in = state
        h = self.encoder(s_in)
        action = self.actor(h)
        risk_pred = self.predictor(torch.cat([h, action], dim=-1))
        return action, risk_pred


class ExistenceDrivenModel(nn.Module):
    """Loss = risk + λ||a||² — direct survival pressure, no predictor.

    Gradient from risk → action injected via env's analytic grad_risk.
    Models "existence is the reward" — only actions that reduce risk survive.
    """

    def __init__(self, state_dim=1, hidden_dim=64, action_dim=1):
        super().__init__()
        self.encoder = nn.Linear(state_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, state):
        h = torch.relu(self.encoder(state))
        return self.actor(h)
