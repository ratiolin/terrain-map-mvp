import torch
import torch.nn as nn


class GatingNet(nn.Module):
    def __init__(self, state_dim, K):
        super().__init__()
        self.hidden = nn.Linear(state_dim, 32)
        self.output = nn.Linear(32, K)

    def forward(self, state):
        x = torch.relu(self.hidden(state))
        return torch.softmax(self.output(x), dim=-1)

    def resize_output(self, K_new):
        old = self.output
        new = nn.Linear(old.in_features, K_new)
        with torch.no_grad():
            k = min(old.out_features, K_new)
            new.weight[:k] = old.weight[:k]
            new.bias[:k] = old.bias[:k]
        self.output = new

    def expand(self):
        self.resize_output(self.output.out_features + 1)

    def shrink(self, indices_to_keep):
        K_new = len(indices_to_keep)
        old = self.output
        new = nn.Linear(old.in_features, K_new)
        with torch.no_grad():
            for ni, oi in enumerate(indices_to_keep):
                if oi < old.out_features:
                    new.weight[ni] = old.weight[oi]
                    new.bias[ni] = old.bias[oi]
        self.output = new


class TemporalGatingNet(nn.Module):
    def __init__(self, state_dim, K, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRUCell(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, K)
        self.hidden = None
        self.hidden_dim = hidden_dim

    def reset(self):
        self.hidden = None

    def forward(self, state):
        s = state.unsqueeze(0) if state.dim() == 1 else state
        if self.hidden is None:
            self.hidden = torch.zeros(s.size(0), self.hidden_dim, device=s.device)
        self.hidden = self.gru(s, self.hidden)
        logits = self.output(self.hidden)
        self.hidden = self.hidden.detach()
        return torch.softmax(logits, dim=-1), logits

    def resize_output(self, K_new):
        old = self.output
        new = nn.Linear(old.in_features, K_new)
        with torch.no_grad():
            k = min(old.out_features, K_new)
            new.weight[:k] = old.weight[:k]
            new.bias[:k] = old.bias[:k]
        self.output = new

    def expand(self):
        self.resize_output(self.output.out_features + 1)

    def shrink(self, indices_to_keep):
        K_new = len(indices_to_keep)
        old = self.output
        new = nn.Linear(old.in_features, K_new)
        with torch.no_grad():
            for ni, oi in enumerate(indices_to_keep):
                if oi < old.out_features:
                    new.weight[ni] = old.weight[oi]
                    new.bias[ni] = old.bias[oi]
        self.output = new


class ZGatingNet(nn.Module):
    def __init__(self, state_dim, K, hidden_dim=32, temperature=0.5):
        super().__init__()
        self.gru = nn.GRUCell(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, K)
        self.direct = nn.Linear(state_dim, K)
        self.hidden = None
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def reset(self):
        self.hidden = None

    def forward(self, state):
        s = state.unsqueeze(0) if state.dim() == 1 else state
        if self.hidden is None:
            self.hidden = torch.zeros(s.size(0), self.hidden_dim, device=s.device)
        self.hidden = self.gru(s, self.hidden)
        z_logits = self.output(self.hidden) + self.direct(s)
        self.hidden = self.hidden.detach()

        z_soft = torch.softmax(z_logits / self.temperature, dim=-1)
        z_hard_idx = z_logits.argmax(dim=-1)
        z_hard = nn.functional.one_hot(z_hard_idx, z_logits.size(-1)).float()
        z = z_hard - z_soft.detach() + z_soft

        return z_soft, z_logits, z_soft

    def resize_output(self, K_new):
        old = self.output
        new = nn.Linear(old.in_features, K_new)
        with torch.no_grad():
            k = min(old.out_features, K_new)
            new.weight[:k] = old.weight[:k]
            new.bias[:k] = old.bias[:k]
        self.output = new

        old_d = self.direct
        new_d = nn.Linear(old_d.in_features, K_new)
        with torch.no_grad():
            k = min(old_d.out_features, K_new)
            new_d.weight[:k] = old_d.weight[:k]
            new_d.bias[:k] = old_d.bias[:k]
        self.direct = new_d

    def expand(self):
        self.resize_output(self.output.out_features + 1)

    def shrink(self, indices_to_keep):
        K_new = len(indices_to_keep)
        old = self.output
        new = nn.Linear(old.in_features, K_new)
        with torch.no_grad():
            for ni, oi in enumerate(indices_to_keep):
                if oi < old.out_features:
                    new.weight[ni] = old.weight[oi]
                    new.bias[ni] = old.bias[oi]
        self.output = new

        old_d = self.direct
        new_d = nn.Linear(old_d.in_features, K_new)
        with torch.no_grad():
            for ni, oi in enumerate(indices_to_keep):
                if oi < old_d.out_features:
                    new_d.weight[ni] = old_d.weight[oi]
                    new_d.bias[ni] = old_d.bias[oi]
        self.direct = new_d
