"""Tier 1-3: Head Capacity Scan — control complexity in head alone.

Tests direct state→action architectures (no encoder):
Linear, 1-layer MLP, 2-layer MLP, 3-layer MLP, wide MLP.
Compares with encoder-equipped baseline.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell


def _behavioral(model, env, n_steps=500):
    env.reset()
    cost_sum = 0.0
    in_zone = 0
    for _ in range(n_steps):
        a = model.act_numpy(env.get_state())
        ns, risk, _, _ = env.step(a)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
    return cost_sum / n_steps, in_zone / n_steps


class DirectModel(nn.Module):
    def __init__(self, d, k, hidden_sizes=None):
        super().__init__()
        self.d = d
        self.k = k
        layers = []
        in_dim = d
        if hidden_sizes:
            for hs in hidden_sizes:
                layers.append(nn.Linear(in_dim, hs))
                layers.append(nn.ReLU())
                in_dim = hs
        layers.append(nn.Linear(in_dim, k))
        self.net = nn.Sequential(*layers)
        self.hidden_dim = hidden_sizes[-1] if hidden_sizes else d

    def forward(self, x):
        return self.net(x)

    def act_numpy(self, s):
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            a = self.net(s_t).squeeze(0).numpy()
        return a

    def f_numpy(self, s):
        return self.act_numpy(s)


class EncoderModel:
    def __init__(self, direct_model):
        self.model = direct_model

    def act_numpy(self, s):
        return self.model.act_numpy(s)


def _train_direct(model, env, n_ep, ep_len, seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            a_t = model.net(s_t)
            a_np = a_t.squeeze(0).detach().numpy()
            ns, risk, _, _ = env.step(a_np)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            loss = 0.1 * torch.mean(a_t**2)
            opt.zero_grad()
            loss.backward()
            opt.step()


def _train_full(model, env, n_ep, ep_len, seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            ns, risk, _, _ = env.step(a_np)
            loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def run_t1_head_capacity(n_seeds=8, d=16, k=2, hd=192,
                         n_episodes=3, episode_length=2000):
    configs = [
        ("linear", []),
        ("MLP_1x32", [32]),
        ("MLP_2x32", [32, 32]),
        ("MLP_3x32", [32, 32, 32]),
        ("MLP_1x128", [128]),
        ("MLP_1x256", [256]),
        ("MLP_2x128", [128, 128]),
    ]
    results = {"direct": {}, "baseline": {}}

    for label, hidden in configs:
        costs = []
        seeds_done = 0
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = DirectModel(d, k, hidden)
            _train_direct(model, env, n_episodes, episode_length, seed)
            c, z = _behavioral(model, env)
            costs.append(c)
            seeds_done += 1

        results["direct"][label] = {
            "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
            "hidden_sizes": hidden,
        }

    print("  Direct models done. Running encoder baseline...")

    for label, hidden in configs:
        costs = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            from core_mvp_v4.models import V4Model
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
            _train_full(model, env, n_episodes, episode_length, seed)
            c, z = _behavioral(model, env)
            costs.append(c)
        results["baseline"][label] = {
            "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
            "encoder_hd": hd, "head_config": label,
        }

    results["analysis"] = {
        "linear_cost": results["direct"]["linear"]["cost_mean"],
        "best_direct": min(configs, key=lambda x: results["direct"][x[0]]["cost_mean"])[0],
        "best_direct_cost": min(v["cost_mean"] for v in results["direct"].values()),
        "baseline_cost": results["baseline"]["MLP_2x32"]["cost_mean"],
        "conclusion": (
            f"Linear cost={results['direct']['linear']['cost_mean']:.4f}. "
            f"Best direct={min(configs, key=lambda x: results['direct'][x[0]]['cost_mean'])[0]} "
            f"cost={min(v['cost_mean'] for v in results['direct'].values()):.4f}. "
            f"Baseline cost={results['baseline']['MLP_2x32']['cost_mean']:.4f}."
        ),
    }
    return results


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_t1_head_capacity(n_seeds=8)
    with open("core_mvp_v4/results/t1_head_capacity.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("T1-3:", r["analysis"]["conclusion"])
