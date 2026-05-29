import torch
import torch.nn as nn


class PredictionNetwork(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=1, hidden_dim=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_shape = nn.Linear(hidden_dim, 1)
        self.head_adapt = nn.Linear(hidden_dim, 1)
        self._init_heads()

    def _init_heads(self):
        nn.init.normal_(self.head_shape.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.head_shape.bias, mean=0.0, std=0.01)
        nn.init.normal_(self.head_adapt.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.head_adapt.bias, mean=0.5, std=0.1)

    def forward(self, state):
        h = self.backbone(state)
        action_shape = self.head_shape(h)
        action_adapt = self.head_adapt(h)
        return action_shape, action_adapt, h
