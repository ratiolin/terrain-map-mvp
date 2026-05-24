import torch
import torch.nn as nn
import torch.nn.functional as F


def _gru_params(state_dim, hidden_dim):
    return 3 * (state_dim * hidden_dim + hidden_dim * hidden_dim + 2 * hidden_dim)


def _zgating_params(state_dim, K, hidden_dim):
    gru = _gru_params(state_dim, hidden_dim)
    out = hidden_dim * K + K
    direct = state_dim * K + K
    return gru + out + direct


def _dual_gru_params(state_dim, K, h1, h2):
    gru1 = _gru_params(state_dim, h1)
    gru2 = _gru_params(state_dim, h2)
    out = (h1 + h2) * K + K
    direct = state_dim * K + K
    return gru1 + gru2 + out + direct


def find_matched_sizes(state_dim, K, H_baseline, ratio_fast=0.25, tolerance=0.05):
    target = _zgating_params(state_dim, K, H_baseline)

    h_capacity = 1
    best_h = h_capacity
    best_diff = float('inf')
    while True:
        p = _dual_gru_params(state_dim, K, h_capacity, h_capacity)
        diff = abs(p - target) / target
        if diff < best_diff:
            best_diff = diff
            best_h = h_capacity
        if p > target * (1 + tolerance):
            break
        h_capacity += 1
    h_ctrl = best_h

    best_hf, best_hs = 1, 1
    best_diff = float('inf')
    for hf in range(1, H_baseline):
        for hs in range(hf, H_baseline * 2):
            if abs(hf / max(1, hs) - ratio_fast) > 0.15:
                continue
            p = _dual_gru_params(state_dim, K, hf, hs)
            diff = abs(p - target) / target
            if diff < best_diff:
                best_diff = diff
                best_hf = hf
                best_hs = hs
    h_fast = best_hf
    h_slow = best_hs

    p_ctrl = _dual_gru_params(state_dim, K, h_ctrl, h_ctrl)
    p_fs = _dual_gru_params(state_dim, K, h_fast, h_slow)

    return {
        "baseline_params": target,
        "capacity_ctrl_h": h_ctrl,
        "capacity_ctrl_params": p_ctrl,
        "capacity_ctrl_err": abs(p_ctrl - target) / target,
        "fast_h": h_fast,
        "slow_h": h_slow,
        "fast_slow_params": p_fs,
        "fast_slow_err": abs(p_fs - target) / target,
    }


class DualGRUGating(nn.Module):
    def __init__(self, state_dim, K, h1, h2, temperature=0.5):
        super().__init__()
        self.gru1 = nn.GRUCell(state_dim, h1)
        self.gru2 = nn.GRUCell(state_dim, h2)
        total_h = h1 + h2
        self.output = nn.Linear(total_h, K)
        self.direct = nn.Linear(state_dim, K)
        self.hidden1 = None
        self.hidden2 = None
        self.h1 = h1
        self.h2 = h2
        self.temperature = temperature

    def reset(self):
        self.hidden1 = None
        self.hidden2 = None

    def forward(self, state):
        s = state.unsqueeze(0) if state.dim() == 1 else state
        if self.hidden1 is None:
            self.hidden1 = torch.zeros(s.size(0), self.h1, device=s.device)
        if self.hidden2 is None:
            self.hidden2 = torch.zeros(s.size(0), self.h2, device=s.device)
        self.hidden1 = self.gru1(s, self.hidden1)
        self.hidden2 = self.gru2(s, self.hidden2)
        h_cat = torch.cat([self.hidden1, self.hidden2], dim=-1)
        z_logits = self.output(h_cat) + self.direct(s)
        self.hidden1 = self.hidden1.detach()
        self.hidden2 = self.hidden2.detach()

        z_soft = F.softmax(z_logits / self.temperature, dim=-1)
        z_hard_idx = z_logits.argmax(dim=-1)
        z_hard = F.one_hot(z_hard_idx, z_logits.size(-1)).float()
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
