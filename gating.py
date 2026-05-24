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
    def __init__(self, state_dim, K, hidden_dim=32, temperature=0.5, inertia=0.0):
        super().__init__()
        self.gru = nn.GRUCell(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, K)
        self.direct = nn.Linear(state_dim, K)
        self.hidden = None
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.inertia = inertia

    def reset(self):
        self.hidden = None

    def forward(self, state):
        s = state.unsqueeze(0) if state.dim() == 1 else state
        if self.hidden is None:
            self.hidden = torch.zeros(s.size(0), self.hidden_dim, device=s.device)
        new_hidden = self.gru(s, self.hidden)
        blended = (1 - self.inertia) * new_hidden + self.inertia * self.hidden.detach()
        z_logits = self.output(blended) + self.direct(s)
        self.hidden = blended.detach()

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


class TextZGatingNet(ZGatingNet):
    def __init__(self, state_dim, K, hidden_dim=32, temperature=0.5, emb_dim=16, scale_init=1.0):
        super().__init__(state_dim, K, hidden_dim, temperature)
        self.text_emb = nn.Embedding(3, emb_dim)
        self.text_proj = nn.Linear(emb_dim, K)
        self.text_scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, state, label=None):
        z_soft_orig, z_logits_orig, _ = super().forward(state)
        text_offset = None
        if label is not None:
            if isinstance(label, int):
                label = torch.tensor([label], device=z_logits_orig.device)
            elif isinstance(label, list):
                label = torch.tensor(label, device=z_logits_orig.device)
            if label.dim() == 0:
                label = label.unsqueeze(0)
            emb = self.text_emb(label)
            text_offset = self.text_scale * self.text_proj(emb)
            z_logits = z_logits_orig + text_offset
        else:
            z_logits = z_logits_orig

        z_soft = torch.softmax(z_logits / self.temperature, dim=-1)
        z_hard_idx = z_logits.argmax(dim=-1)
        z_hard = nn.functional.one_hot(z_hard_idx, z_logits.size(-1)).float()
        z = z_hard - z_soft.detach() + z_soft

        if text_offset is not None:
            return z_soft, z_logits, z, z_logits_orig, text_offset
        return z_soft, z_logits, z

    def expand(self):
        old_K = self.output.out_features
        super().expand()
        new_K = old_K + 1
        old_proj = self.text_proj
        new_proj = nn.Linear(old_proj.in_features, new_K)
        with torch.no_grad():
            new_proj.weight[:old_K] = old_proj.weight[:old_K]
            new_proj.bias[:old_K] = old_proj.bias[:old_K]
        self.text_proj = new_proj

    def shrink(self, indices_to_keep):
        super().shrink(indices_to_keep)
        K_new = len(indices_to_keep)
        old_proj = self.text_proj
        new_proj = nn.Linear(old_proj.in_features, K_new)
        with torch.no_grad():
            for ni, oi in enumerate(indices_to_keep):
                if oi < old_proj.out_features:
                    new_proj.weight[ni] = old_proj.weight[oi]
                    new_proj.bias[ni] = old_proj.bias[oi]
        self.text_proj = new_proj


class FreqGatingNet(ZGatingNet):
    def __init__(self, state_dim, K, hidden_dim=32, temperature=0.5, inertia=0.0, memory_k=None):
        super().__init__(state_dim, K, hidden_dim, temperature, inertia)
        self.memory_k = memory_k
        if memory_k is not None and memory_k > 1:
            fft_bins = state_dim // 2 + 1
            self.freq_net = nn.Sequential(
                nn.Linear(fft_bins, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, K)
            )
            self.freq_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, state):
        s = state.unsqueeze(0) if state.dim() == 1 else state
        if self.hidden is None:
            self.hidden = torch.zeros(s.size(0), self.hidden_dim, device=s.device)
        new_hidden = self.gru(s, self.hidden)
        blended = (1 - self.inertia) * new_hidden + self.inertia * self.hidden.detach()
        z_logits = self.output(blended) + self.direct(s)

        if hasattr(self, 'freq_net'):
            fft_amp = torch.abs(torch.fft.rfft(s.float(), dim=-1))
            freq_logits = self.freq_weight * self.freq_net(fft_amp)
            z_logits = z_logits + freq_logits

        self.hidden = blended.detach()

        z_soft = torch.softmax(z_logits / self.temperature, dim=-1)
        z_hard_idx = z_logits.argmax(dim=-1)
        z_hard = nn.functional.one_hot(z_hard_idx, z_logits.size(-1)).float()
        z = z_hard - z_soft.detach() + z_soft

        return z_soft, z_logits, z_soft

    def resize_output(self, K_new):
        super().resize_output(K_new)
        if hasattr(self, 'freq_net'):
            old_f = self.freq_net[-1]
            new_f = nn.Linear(old_f.in_features, K_new)
            with torch.no_grad():
                k = min(old_f.out_features, K_new)
                new_f.weight[:k] = old_f.weight[:k]
                new_f.bias[:k] = old_f.bias[:k]
            self.freq_net[-1] = new_f

    def shrink(self, indices_to_keep):
        super().shrink(indices_to_keep)
        if hasattr(self, 'freq_net'):
            K_new = len(indices_to_keep)
            old_f = self.freq_net[-1]
            new_f = nn.Linear(old_f.in_features, K_new)
            with torch.no_grad():
                for ni, oi in enumerate(indices_to_keep):
                    if oi < old_f.out_features:
                        new_f.weight[ni] = old_f.weight[oi]
                        new_f.bias[ni] = old_f.bias[oi]
            self.freq_net[-1] = new_f
