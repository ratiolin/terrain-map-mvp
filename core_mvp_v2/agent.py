import torch
import torch.nn as nn
import torch.optim as optim


class Expert(nn.Module):
    def __init__(self, obs_dim=1, hidden_dim=4, lr=1e-3):
        super().__init__()
        self.obs_dim = obs_dim
        input_dim = obs_dim + 1
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs, action):
        x = torch.cat([
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor([action], dtype=torch.float32)
        ])
        return self.predictor(x)
