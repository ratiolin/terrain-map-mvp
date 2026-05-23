import torch
import torch.nn as nn
import torch.optim as optim
import random


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, lr=1e-3, hidden_dim=32):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(hidden_dim, act_dim)

        self.predictor = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def act(self, obs):
        if random.random() < 0.1:
            return random.randint(0, 1)

        x = self.shared(torch.tensor(obs, dtype=torch.float32))
        logits = self.policy_head(x)
        return torch.argmax(logits).item()

    def predict(self, obs, action):
        act_tensor = torch.tensor([action], dtype=torch.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        x = torch.cat([obs_tensor, act_tensor])
        return self.predictor(x)

    def update(self, obs, action, next_obs):
        pred = self.predict(obs, action)
        target = torch.tensor(next_obs, dtype=torch.float32)
        loss = ((pred - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
