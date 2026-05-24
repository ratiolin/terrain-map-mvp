import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, hidden_dim, state_dim=1):
        super().__init__()
        input_dim = state_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.state_dim = state_dim

    def predict(self, obs, action):
        act_tensor = torch.tensor([action], dtype=torch.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        x = torch.cat([obs_tensor, act_tensor])
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def count_multi_params(K, expert_hidden, gating_hidden, state_dim=1):
    expert_params = K * ((state_dim + 1) * expert_hidden + expert_hidden
                         + expert_hidden * state_dim + state_dim)
    gru_params = 3 * (
        state_dim * gating_hidden
        + gating_hidden * gating_hidden
        + 2 * gating_hidden
    )
    output_params = gating_hidden * K + K
    direct_params = state_dim * K + K
    gating_params = gru_params + output_params + direct_params
    return expert_params + gating_params


def make_single_large(K, expert_hidden, gating_hidden, state_dim=1):
    total = count_multi_params(K, expert_hidden, gating_hidden, state_dim)
    params_per_hidden = (
        (state_dim + 1) + 1  # Linear(in, H): in*H + H → per H: in + 1
        + 1 + state_dim       # Linear(H, out): H*out + out → per H: out + 1/out...
    )
    H_large = max(1, int((total - state_dim) / ((state_dim + 1) + state_dim + 2)))
    model = MLP(H_large, state_dim=state_dim)
    actual = model.count_params()
    while actual < total * 0.95:
        H_large += 1
        model = MLP(H_large, state_dim=state_dim)
        actual = model.count_params()
    while actual > total * 1.05:
        H_large -= 1
        if H_large < 1:
            H_large = 1
            model = MLP(H_large, state_dim=state_dim)
            actual = model.count_params()
            break
        model = MLP(H_large, state_dim=state_dim)
        actual = model.count_params()
    return model, H_large


def train_single_model(model, env, steps, lr=1e-3, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    obs = env.reset()
    mse_history = []
    state_history = []

    for _ in range(steps):
        action = random.randint(0, 1)
        o_next, _, done = env.step(action)

        pred = model.predict(obs, action)
        target = torch.tensor(o_next, dtype=torch.float32)
        loss = ((pred - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mse_history.append(loss.item())
        state_history.append(float(o_next[0]))

        if done:
            obs = env.reset()
        else:
            obs = o_next

    return mse_history, state_history
