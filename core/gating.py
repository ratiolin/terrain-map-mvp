import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNet(nn.Module):
    def __init__(self, obs_dim, K, hidden_dim=8):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim + obs_dim, K)
        self.hidden = None
        self.temperature = 1.0
        self.inertia = 0.9
        self.prev_z = None
        self.hidden_decay = 1.0

    def set_temperature(self, T):
        self.temperature = T

    def set_inertia(self, inertia):
        self.inertia = max(0.0, min(1.0, inertia))

    def set_hidden_decay(self, decay):
        self.hidden_decay = max(0.0, min(1.0, decay))

    def reset(self):
        self.hidden = None
        self.prev_z = None

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs_batch = obs.unsqueeze(0)
        if self.hidden is not None:
            self.hidden = self.hidden.detach()
            if self.hidden_decay < 1.0:
                self.hidden = self.hidden * self.hidden_decay
        gru_out, self.hidden = self.gru(obs_batch, self.hidden)
        gru_feat = gru_out.squeeze(0)
        x = torch.cat([gru_feat, obs], dim=-1)
        logits = self.linear(x)
        z_raw = F.softmax(logits / max(self.temperature, 0.01), dim=-1)
        if self.prev_z is not None and self.prev_z.size(-1) == z_raw.size(-1):
            z = self.inertia * self.prev_z + (1.0 - self.inertia) * z_raw
        else:
            z = z_raw
        self.prev_z = z.detach()
        return z, logits

    def expand(self):
        old_K = self.linear.out_features
        new_K = old_K + 1
        new_linear = nn.Linear(self.linear.in_features, new_K)
        with torch.no_grad():
            new_linear.weight[:old_K] = self.linear.weight
            new_linear.bias[:old_K] = self.linear.bias
            new_linear.weight[old_K:] = self.linear.weight.mean(dim=0, keepdim=True) + torch.randn(1, self.linear.in_features) * 0.01
            new_linear.bias[old_K] = self.linear.bias.mean() + torch.randn(1).item() * 0.01
        self.linear = new_linear

    def shrink(self, keep_indices):
        new_K = len(keep_indices)
        new_linear = nn.Linear(self.linear.in_features, new_K)
        with torch.no_grad():
            for new_i, old_i in enumerate(keep_indices):
                new_linear.weight[new_i] = self.linear.weight[old_i]
                new_linear.bias[new_i] = self.linear.bias[old_i]
        self.linear = new_linear
