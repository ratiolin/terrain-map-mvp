import torch
import torch.nn as nn
import torch.optim as optim


class TorchControlWell:
    def __init__(self, kappa=4.0, drift_rate=0.02, noise_std=0.1,
                 state_clip=3.0, force_scale=0.1, alpha=0.2,
                 target_state=1.0):
        self.kappa = kappa
        self.drift_rate = drift_rate
        self.noise_std = noise_std
        self.state_clip = state_clip
        self.force_scale = force_scale
        self.alpha = alpha
        self.target_state = target_state
        self.t = 0
        self.drift = 0.0
        self.state = torch.tensor(0.0)

    def reset(self):
        self.state = torch.randn(1) * 0.5
        self.t = 0
        self.drift = 0.0
        return self.state.detach()

    def grad_potential(self, x):
        b = 1.0 + self.drift
        return 4.0 * x**3 - 2.0 * b * x + self.kappa

    def step(self, action):
        self.drift = self.drift_rate * self.t
        x = torch.clamp(self.state, -self.state_clip, self.state_clip)
        force = -self.force_scale * self.grad_potential(x)
        control = action * self.alpha
        eps = torch.randn(1)
        noise = self.noise_std * eps
        x_next = x + force + control + noise
        x_next = torch.clamp(x_next, -self.state_clip, self.state_clip)
        self.state = x_next
        cost = (x_next - self.target_state) ** 2
        self.t += 1
        return cost

    def detach_state(self):
        self.state = self.state.detach()


class ControlPolicy(nn.Module):
    def __init__(self, obs_dim=1, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, obs):
        if obs.dim() == 0:
            obs = obs.unsqueeze(0)
        return torch.tanh(self.net(obs).squeeze(-1))
