"""Tier 2-5: Encoder's Unique Role.

Tests what the encoder actually does:
- Sample efficiency
- Generalization (d=16→d=32)
- Noise robustness (observation and hidden noise)
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model


def _behavioral(model, env, n_steps=500, obs_noise=0.0, hidden_noise=0.0):
    env.reset()
    cost_sum = 0.0
    in_zone = 0
    for _ in range(n_steps):
        s_raw = env.get_state()
        if obs_noise > 0:
            s_raw = s_raw + np.random.normal(0, obs_noise, size=s_raw.shape)

        if hidden_noise > 0 and hasattr(model, 'encoder'):
            s_t = torch.from_numpy(s_raw.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0)
            h = h + torch.randn_like(h) * hidden_noise
            a = model.actor(h).squeeze(0).detach().numpy()
        else:
            a = model.act_numpy(s_raw)
        ns, risk, _, _ = env.step(a)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
    return cost_sum / n_steps, in_zone / n_steps


def _train_full(model, env, n_ep, ep_len, seed, cost_history=False):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    costs_by_ep = []
    for i in range(n_ep):
        env.reset()
        ep_costs = []
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            ns, risk, _, _ = env.step(a_np)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            loss = nn.functional.mse_loss(risk_pred, risk_t) + 0.1 * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if cost_history:
                ep_costs.append(risk)
        if cost_history:
            costs_by_ep.append(float(np.mean(ep_costs)))
    return costs_by_ep if cost_history else []


class DirectModel:
    def __init__(self, d, k):
        self.d = d
        self.k = k
        self.net = nn.Sequential(nn.Linear(d, k))
        self.hidden_dim = k

    def act_numpy(self, s):
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            return self.net(s_t).squeeze(0).numpy()


def _train_direct(model, env, n_ep, ep_len, seed, cost_history=False):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    opt = torch.optim.Adam(model.net.parameters(), lr=1e-2)
    costs_by_ep = []
    for i in range(n_ep):
        env.reset()
        ep_costs = []
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            a_t = model.net(s_t)
            a_np = a_t.squeeze(0).detach().numpy()
            ns, risk, _, _ = env.step(a_np)
            loss = 0.1 * torch.mean(a_t**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cost_history:
                ep_costs.append(risk)
        if cost_history:
            costs_by_ep.append(float(np.mean(ep_costs)))
    return costs_by_ep if cost_history else []


def run_t2_encoder_role(n_seeds=8, d=16, k=2, hd=192,
                        n_episodes=5, episode_length=2000):
    results = {"efficiency": {}, "generalization": {}, "noise": {}}

    # Sample efficiency
    enc_curves = []
    dir_curves = []
    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        curve = _train_full(model, env, n_episodes, episode_length, seed, cost_history=True)
        enc_curves.append(curve)

        env2 = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model2 = DirectModel(d, k)
        curve2 = _train_direct(model2, env2, n_episodes, episode_length, seed, cost_history=True)
        dir_curves.append(curve2)

    results["efficiency"] = {
        "encoder_final_cost": float(np.mean([c[-1] if c else 0 for c in enc_curves])),
        "direct_final_cost": float(np.mean([c[-1] if c else 0 for c in dir_curves])),
    }

    # Generalization: train d=16, test d=32 (pad with zeros)
    gen_enc = []; gen_dir = []
    for seed in range(n_seeds):
        env_train = MultiDimDoubleWell(d=16, k=k, drift=0.5, seed=seed, coupling=0.0)
        env_test = MultiDimDoubleWell(d=32, k=k, drift=0.5, seed=seed + 9999, coupling=0.0)

        model_enc = V4Model(state_dim=16, hidden_dim=hd, action_dim=k)
        _train_full(model_enc, env_train, n_episodes, episode_length, seed)

        class PaddedWrapper:
            def __init__(self, inner, orig_d):
                self.inner = inner
                self.orig_d = orig_d
            def act_numpy(self, s):
                return self.inner.act_numpy(s[:self.orig_d])
        wrapper_enc = PaddedWrapper(model_enc, 16)
        c, z = _behavioral(wrapper_enc, env_test)
        gen_enc.append(c)

        model_dir = DirectModel(16, k)
        _train_direct(model_dir, env_train, n_episodes, episode_length, seed)
        wrapper_dir = PaddedWrapper(model_dir, 16)
        c, z = _behavioral(wrapper_dir, env_test)
        gen_dir.append(c)

    results["generalization"] = {
        "encoder_gen_cost": float(np.mean(gen_enc)),
        "direct_gen_cost": float(np.mean(gen_dir)),
    }

    # Noise robustness
    for sigma in [0.1, 0.2]:
        noise_enc = []; noise_dir = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model_enc = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
            _train_full(model_enc, env, n_episodes, episode_length, seed)
            c_enc, _ = _behavioral(model_enc, env, obs_noise=sigma)
            noise_enc.append(c_enc)

            env2 = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model_dir = DirectModel(d, k)
            _train_direct(model_dir, env2, n_episodes, episode_length, seed)
            c_dir, _ = _behavioral(model_dir, env, obs_noise=sigma)
            noise_dir.append(c_dir)

        results["noise"][f"sigma_{sigma}"] = {
            "encoder": float(np.mean(noise_enc)),
            "direct": float(np.mean(noise_dir)),
            "encoder_advantage": float(np.mean(noise_dir) / (np.mean(noise_enc) + 1e-6)),
        }

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    eff_enc = results["efficiency"]["encoder_final_cost"]
    eff_dir = results["efficiency"]["direct_final_cost"]
    gen_enc = results["generalization"]["encoder_gen_cost"]
    gen_dir = results["generalization"]["direct_gen_cost"]
    noise_adv = results["noise"]["sigma_0.1"]["encoder_advantage"]

    parts = []
    if eff_dir < eff_enc * 1.2:
        parts.append("Equal sample efficiency → encoder not needed for convergence speed.")
    else:
        parts.append(f"Encoder speeds convergence ({eff_dir:.2f} vs {eff_enc:.2f}).")

    if gen_dir < gen_enc * 1.5:
        parts.append("Similar generalization → encoder not needed for feature invariance.")
    else:
        parts.append("Encoder improves generalization.")

    if noise_adv > 1.2:
        parts.append(f"Noise advantage {noise_adv:.2f}x → encoder provides noise filtering.")
    else:
        parts.append("No noise advantage → encoder provides no filtering.")

    return {"conclusion": " ".join(parts)}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_t2_encoder_role(n_seeds=8)
    with open("core_mvp_v4/results/t2_encoder_role.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("T2-5:", r["analysis"]["conclusion"])
