import torch
import torch.nn as nn
import numpy as np


class V4Model(nn.Module):
    def __init__(self, state_dim, hidden_dim=32, action_dim=2):
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

    def predict_numpy(self, s, a):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        if not isinstance(a, np.ndarray):
            a = np.array(a, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        a_t = torch.from_numpy(a.astype(np.float32)).unsqueeze(0)
        x = torch.cat([s_t, a_t], dim=-1)
        with torch.no_grad():
            p = self.predictor(self.encoder(s_t))
        return p.squeeze().numpy()


def compute_jacobian(model, s, eps=1e-3):
    if not isinstance(s, np.ndarray):
        s = np.array(s, dtype=np.float32)
    d = s.shape[0]
    h0 = model.f_numpy(s)
    h_dim = h0.shape[0]
    J = np.zeros((h_dim, d))
    for i in range(d):
        e = np.zeros_like(s)
        e[i] = eps
        J[:, i] = (model.f_numpy(s + e) - model.f_numpy(s - e)) / (2 * eps)
    return J


def compute_jacobian_batch(model, states, eps=1e-3):
    Js = []
    for s in states:
        Js.append(compute_jacobian(model, s, eps))
    return np.stack(Js, axis=0)


def train_model(model, env, num_episodes=5, episode_length=2000, lr=1e-3,
                lambda_ctrl=0.1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = 0
    episode_losses = []

    for ep in range(num_episodes):
        state = env.reset()
        ep_loss = 0.0
        ep_steps = 0

        for step in range(episode_length):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)

            a_np = action.squeeze(0).detach().numpy()
            next_state, risk_actual, _, _ = env.step(a_np)

            risk_t = torch.tensor([[risk_actual]], dtype=torch.float32)

            pred_loss = nn.functional.mse_loss(risk_pred, risk_t)

            a_norm = torch.mean(action ** 2)

            loss = pred_loss + lambda_ctrl * a_norm

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_steps += 1
            total_steps += 1
            state = next_state

        episode_losses.append(ep_loss / ep_steps)

    return episode_losses


def train_with_signal_loss(model, env, num_episodes=5, episode_length=2000, lr=1e-3,
                           lambda_ctrl=0.1, lambda_signal=0.5, seed=None, alpha=1.0):
    """Train with controllability signal probe loss.

    Loss = pred_loss + lambda_ctrl * action_norm + lambda_signal * MSE(probe(h), C(s))

    C(s) = ||forward_static(s, act(s)) - forward_static(s, 0)|| on control dims

    alpha controls the reward signal mixture: C_noisy = alpha*C(s) + (1-alpha)*randn()
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    signal_probe = nn.Linear(model.hidden_dim, 1)
    params = list(model.parameters()) + list(signal_probe.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    episode_losses = []

    for ep in range(num_episodes):
        state = env.reset()
        ep_loss = 0.0
        ep_steps = 0

        for step in range(episode_length):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)

            a_np = action.squeeze(0).detach().numpy()

            s_next_actual = env.forward_static(state, a_np)
            s_next_zero = env.forward_static(state, np.zeros(env.action_dim))
            delta = s_next_actual[:env.k] - s_next_zero[:env.k]
            C_actual = float(np.linalg.norm(delta))

            if alpha < 1.0:
                C_actual = alpha * C_actual + (1.0 - alpha) * float(np.random.randn())

            next_state, risk_actual, _, _ = env.step(a_np)

            risk_t = torch.tensor([[risk_actual]], dtype=torch.float32)
            C_t = torch.tensor([[C_actual]], dtype=torch.float32)
            C_pred = signal_probe(h)

            pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
            signal_loss = nn.functional.mse_loss(C_pred, C_t)
            action_loss = torch.mean(action ** 2)

            loss = pred_loss + lambda_ctrl * action_loss + lambda_signal * signal_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(signal_probe.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_steps += 1
            state = next_state

        episode_losses.append(ep_loss / ep_steps)

    return episode_losses


def collect_controllability_data(model, env, n_samples=500, sigma_obs=0.0):
    hidden_list = []
    C_list = []
    state = env.reset()
    for _ in range(n_samples):
        s_obs = state.copy()
        if sigma_obs > 0:
            s_obs = s_obs + np.random.normal(0, sigma_obs, size=state.shape)

        h = model.f_numpy(s_obs)
        a = model.act_numpy(s_obs)

        s_next_actual = env.forward_static(state, a)
        s_next_zero = env.forward_static(state, np.zeros(env.action_dim))
        delta = s_next_actual[:env.k] - s_next_zero[:env.k]
        C_val = float(np.linalg.norm(delta))

        hidden_list.append(h)
        C_list.append(C_val)

        ns, _, _, _ = env.step(a)
        state = ns

    return np.array(hidden_list), np.array(C_list)


def train_with_panic(model, env, num_episodes=5, episode_length=2000, lr=1e-3,
                     lambda_ctrl=0.1, panic_mode="baseline", seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = 0
    episode_losses = []

    for ep in range(num_episodes):
        state = env.reset()
        ep_loss = 0.0
        ep_steps = 0

        for step in range(episode_length):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)

            a_np = action.squeeze(0).detach().numpy()

            if panic_mode == "random":
                panic = float(np.random.randn())
                a_np = a_np + panic * 0.1
            elif panic_mode == "none":
                pass

            next_state, risk_actual, _, _ = env.step(a_np)

            if panic_mode == "random_reward":
                risk_actual = float(np.random.randn())

            risk_t = torch.tensor([[risk_actual]], dtype=torch.float32)
            pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
            a_norm = torch.mean(action ** 2)
            loss = pred_loss + lambda_ctrl * a_norm

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_steps += 1
            total_steps += 1
            state = next_state

        episode_losses.append(ep_loss / ep_steps)

    return episode_losses


def train_with_noise_dropout(model, env, num_episodes=5, episode_length=2000, lr=1e-3,
                             lambda_ctrl=0.1, lambda_signal=0.5, seed=None,
                             noise_dropout=0.5):
    """Train with random dropout on noise dimensions (B2: competition test)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    signal_probe = nn.Linear(model.hidden_dim, 1)
    params = list(model.parameters()) + list(signal_probe.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    episode_losses = []

    for ep in range(num_episodes):
        state = env.reset()
        ep_loss = 0.0
        ep_steps = 0

        for step in range(episode_length):
            s_masked = state.copy()
            if env.d > env.k and noise_dropout > 0:
                mask = np.random.rand(env.d - env.k) < noise_dropout
                s_masked[env.k:] = s_masked[env.k:] * ~mask

            s_t = torch.from_numpy(s_masked.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)

            a_np = action.squeeze(0).detach().numpy()

            s_next_actual = env.forward_static(state, a_np)
            s_next_zero = env.forward_static(state, np.zeros(env.action_dim))
            delta = s_next_actual[:env.k] - s_next_zero[:env.k]
            C_actual = float(np.linalg.norm(delta))

            next_state, risk_actual, _, _ = env.step(a_np)

            risk_t = torch.tensor([[risk_actual]], dtype=torch.float32)
            C_t = torch.tensor([[C_actual]], dtype=torch.float32)
            C_pred = signal_probe(h)

            pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
            signal_loss = nn.functional.mse_loss(C_pred, C_t)
            action_loss = torch.mean(action ** 2)

            loss = pred_loss + lambda_ctrl * action_loss + lambda_signal * signal_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(signal_probe.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_steps += 1
            state = next_state

        episode_losses.append(ep_loss / ep_steps)

    return episode_losses


class SplitV4Model(nn.Module):
    """C2: Split encoder — separate branches for control and noise dimensions."""

    def __init__(self, state_dim, hidden_dim=32, action_dim=2, k=2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.k = k
        h_half = hidden_dim // 2

        self.encoder_ctrl = nn.Sequential(
            nn.Linear(k, h_half),
            nn.ReLU(),
            nn.Linear(h_half, h_half),
            nn.ReLU(),
        )
        n_noise = state_dim - k
        self.encoder_noise = nn.Sequential(
            nn.Linear(n_noise, h_half),
            nn.ReLU(),
            nn.Linear(h_half, h_half),
            nn.ReLU(),
        ) if n_noise > 0 else nn.Identity()

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim if n_noise > 0 else h_half, hidden_dim),
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
        s_ctrl = state[:, :self.k]
        h_ctrl = self.encoder_ctrl(s_ctrl)

        if self.state_dim > self.k:
            s_noise = state[:, self.k:]
            h_noise = self.encoder_noise(s_noise)
            h = torch.cat([h_ctrl, h_noise], dim=-1)
        else:
            h = h_ctrl

        h = self.fusion(h)
        action = self.actor(h)
        risk_pred = self.predictor(h)
        return action, h, risk_pred

    def get_hidden(self, state):
        return self.encoder(state) if hasattr(self, 'encoder') else self._compute_hidden(state)

    def _compute_hidden(self, state):
        s_ctrl = state[:, :self.k]
        h_ctrl = self.encoder_ctrl(s_ctrl)
        if self.state_dim > self.k:
            s_noise = state[:, self.k:]
            h_noise = self.encoder_noise(s_noise)
            h = torch.cat([h_ctrl, h_noise], dim=-1)
        else:
            h = h_ctrl
        return self.fusion(h)

    def f_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            h = self._compute_hidden(s_t).squeeze(0)
        return h.numpy()

    def act_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            action, h, _ = self(s_t)
            return action.squeeze(0).numpy()


class AttentionV4Model(nn.Module):
    """C3: Attention-gated model — learns to focus on controllable dims."""

    def __init__(self, state_dim, hidden_dim=32, action_dim=2):
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
        self.attention = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Sigmoid(),
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
            if isinstance(m, nn.Linear) and not isinstance(m, type(self.attention[-1])):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        attn = self.attention(state)
        gated = state * attn
        h = self.encoder(gated)
        action = self.actor(h)
        risk_pred = self.predictor(h)
        return action, h, risk_pred, attn

    def get_hidden(self, state):
        attn = self.attention(state)
        return self.encoder(state * attn)

    def f_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            h = self.get_hidden(s_t).squeeze(0)
        return h.numpy()

    def act_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            action, h, risk_pred, attn = self(s_t)
            return action.squeeze(0).numpy()

    def attn_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            attn = self.attention(s_t).squeeze(0)
        return attn.numpy()


class ScaledInputModel(nn.Module):
    """G2-A/G2-B: Model with learnable per-dimension input scaling weights.

    s_weighted = s * input_weights
    Optionally L1-regularized to encourage sparsity (G2-B).
    """

    def __init__(self, state_dim, hidden_dim=32, action_dim=2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.input_weights = nn.Parameter(torch.ones(state_dim))

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
        gated = state * self.input_weights.unsqueeze(0)
        h = self.encoder(gated)
        action = self.actor(h)
        risk_pred = self.predictor(h)
        return action, h, risk_pred

    def get_hidden(self, state):
        gated = state * self.input_weights.unsqueeze(0)
        return self.encoder(gated)

    def f_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            h = self.get_hidden(s_t).squeeze(0)
        return h.numpy()

    def act_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            action, h, risk_pred = self(s_t)
            return action.squeeze(0).numpy()

    def weight_numpy(self):
        return self.input_weights.detach().numpy()

class GuidedAttentionModel(nn.Module):
    """Step 4: Attention with learnable position embeddings on control dims.

    Adds a learnable embedding per input dimension, plus a binary indicator
    (1 for control dims, 0 for noise). The attention sees [state, embed, indicator].
    """

    def __init__(self, state_dim, hidden_dim=32, action_dim=2, k=2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.k = k

        self.pos_embed = nn.Parameter(torch.randn(state_dim) * 0.02)
        self.register_buffer("ctrl_indicator", torch.cat([
            torch.ones(k), torch.zeros(state_dim - k)
        ]))

        attn_input_dim = state_dim + state_dim + state_dim
        self.attention = nn.Sequential(
            nn.Linear(attn_input_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
            nn.Sigmoid(),
        )
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
        B = state.shape[0]
        pos = self.pos_embed.unsqueeze(0).expand(B, -1)
        ind = self.ctrl_indicator.unsqueeze(0).expand(B, -1)
        attn_input = torch.cat([state, pos, ind], dim=-1)
        attn = self.attention(attn_input)
        gated = state * attn
        h = self.encoder(gated)
        action = self.actor(h)
        risk_pred = self.predictor(h)
        return action, h, risk_pred, attn

    def get_hidden(self, state):
        B = state.shape[0]
        pos = self.pos_embed.unsqueeze(0).expand(B, -1)
        ind = self.ctrl_indicator.unsqueeze(0).expand(B, -1)
        attn_input = torch.cat([state, pos, ind], dim=-1)
        attn = self.attention(attn_input)
        return self.encoder(state * attn)

    def f_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            h = self.get_hidden(s_t).squeeze(0)
        return h.numpy()

    def act_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            action, h, risk_pred, attn = self(s_t)
            return action.squeeze(0).numpy()

    def attn_numpy(self, s):
        if not isinstance(s, np.ndarray):
            s = np.array(s, dtype=np.float32)
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            pos = self.pos_embed.unsqueeze(0)
            ind = self.ctrl_indicator.unsqueeze(0)
            attn_input = torch.cat([s_t, pos, ind], dim=-1)
            attn = self.attention(attn_input).squeeze(0)
        return attn.numpy()
