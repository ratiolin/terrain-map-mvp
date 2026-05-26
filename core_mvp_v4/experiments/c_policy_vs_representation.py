"""Direction C: Policy Layer vs Representation Layer Separation.

Tests whether control ability resides in the policy head rather than the encoder.

C1: Freeze encoder, retrain policy head from scratch.
C2: Gaussian noise on encoder vs policy head, compare degradation.
C3: Remove encoder entirely, direct state→action mapping.
"""

import json
import copy
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, compute_jacobian
from core_mvp_v4.metrics import compute_k80, alignment


def _behavioral(model, env, d, k, n_steps=500):
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


def _train_full(model, env, n_ep, ep_len, seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            outputs = model(s_t)
            action, h, risk_pred = outputs[0], outputs[1], outputs[2]
            a_np = action.squeeze(0).detach().numpy()
            ns, risk, _, _ = env.step(a_np)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            loss = nn.functional.mse_loss(risk_pred, risk_t) + 0.1 * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def run_c_policy_vs_representation(n_seeds=8, d=16, k=2, hd=192,
                                   n_episodes=3, episode_length=2000):
    results = {"C1_freeze_encoder": [], "C2_noise_sensitivity": [], "C3_no_encoder": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train_full(model, env, n_episodes, episode_length, seed)

        original_cost, original_zone = _behavioral(model, env, d, k)

        # ---- C1: freeze encoder, retrain policy head ----
        model_c1 = copy.deepcopy(model)
        for p in model_c1.encoder.parameters():
            p.requires_grad = False
        def _reinit(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if hasattr(model_c1.actor, 'apply'):
            model_c1.actor.apply(_reinit)
        env_c1 = env.clone()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        opt = torch.optim.Adam([p for p in model_c1.parameters() if p.requires_grad], lr=1e-3)
        for _ in range(n_episodes):
            env_c1.reset()
            for __ in range(episode_length):
                s_t = torch.from_numpy(env_c1.get_state().astype(np.float32)).unsqueeze(0)
                outputs = model_c1(s_t)
                action, h, risk_pred = outputs[0], outputs[1], outputs[2]
                a_np = action.squeeze(0).detach().numpy()
                ns, risk, _, _ = env_c1.step(a_np)
                risk_t = torch.tensor([[risk]], dtype=torch.float32)
                loss = nn.functional.mse_loss(risk_pred, risk_t) + 0.1 * torch.mean(action**2)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_c1.parameters(), 1.0)
                opt.step()
        c1_cost, c1_zone = _behavioral(model_c1, env, d, k)
        results["C1_freeze_encoder"].append({
            "seed": seed,
            "original_cost": original_cost, "original_zone": original_zone,
            "retrained_cost": c1_cost, "retrained_zone": c1_zone,
            "cost_ratio": c1_cost / (original_cost + 1e-6),
            "zone_ratio": c1_zone / (original_zone + 1e-6),
        })

        # ---- C2: noise sensitivity (encoder vs policy head) ----
        c2_sigmas = [0.01, 0.05, 0.1]
        noise_data = {"seed": seed, "sigma": []}
        for sigma in c2_sigmas:
            model_enc = copy.deepcopy(model)
            for p in model_enc.encoder.parameters():
                p.data += torch.randn_like(p) * sigma
            ec, ez = _behavioral(model_enc, env, d, k)

            model_pol = copy.deepcopy(model)
            for p in model_pol.actor.parameters():
                p.data += torch.randn_like(p) * sigma
            pc, pz = _behavioral(model_pol, env, d, k)

            noise_data["sigma"].append({
                "sigma": sigma,
                "encoder_noise_cost": ec, "encoder_noise_zone": ez,
                "policy_noise_cost": pc, "policy_noise_zone": pz,
                "encoder_cost_ratio": ec / (original_cost + 1e-6),
                "policy_cost_ratio": pc / (original_cost + 1e-6),
            })
        results["C2_noise_sensitivity"].append(noise_data)

        # ---- C3: remove encoder, direct state→action ----
        class DirectModel:
            def __init__(self, d, k):
                self.d = d
                self.k = k
                self.actor = nn.Linear(d, k)
                nn.init.xavier_uniform_(self.actor.weight)
                nn.init.zeros_(self.actor.bias)
                self.hidden_dim = d
            def act_numpy(self, s):
                s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    a = self.actor(s_t).squeeze(0).numpy()
                return a
            def get_hidden(self, s):
                return s

        model_c3 = DirectModel(d=d, k=k)
        env_c3 = env.clone()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        opt = torch.optim.Adam(model_c3.actor.parameters(), lr=1e-2)
        for _ in range(n_episodes):
            env_c3.reset()
            for __ in range(episode_length):
                s_t = torch.from_numpy(env_c3.get_state().astype(np.float32)).unsqueeze(0)
                a_t = model_c3.actor(s_t)
                a_np = a_t.squeeze(0).detach().numpy()
                ns, risk, _, _ = env_c3.step(a_np)
                risk_t = torch.tensor([[risk]], dtype=torch.float32)
                loss = 0.1 * torch.mean(a_t**2)
                opt.zero_grad()
                loss.backward()
                opt.step()
        c3_cost, c3_zone = _behavioral(model_c3, env, d, k)
        results["C3_no_encoder"].append({
            "seed": seed,
            "original_cost": original_cost, "direct_cost": c3_cost,
            "direct_zone": c3_zone,
            "cost_ratio": c3_cost / (original_cost + 1e-6),
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    c1_ratios = [r["cost_ratio"] for r in results["C1_freeze_encoder"]]
    c1_mean = float(np.mean(c1_ratios))
    c1_recovered = c1_mean < 1.1

    c2_data = results["C2_noise_sensitivity"]
    enc_ratios_high = float(np.mean([d["sigma"][-1]["encoder_cost_ratio"] for d in c2_data]))
    pol_ratios_high = float(np.mean([d["sigma"][-1]["policy_cost_ratio"] for d in c2_data]))
    pol_more_sensitive = pol_ratios_high > enc_ratios_high

    c3_ratios = [r["cost_ratio"] for r in results["C3_no_encoder"]]
    c3_mean = float(np.mean(c3_ratios))

    parts = []
    if c1_recovered:
        parts.append(f"Frozen encoder retrain recovers to {c1_mean:.2f}x cost → control in policy head, not encoder.")
    else:
        parts.append(f"Frozen encoder cannot recover ({c1_mean:.2f}x cost) → encoder representation is necessary.")

    if pol_more_sensitive:
        parts.append(f"Policy head more noise-sensitive ({pol_ratios_high:.2f}x vs {enc_ratios_high:.2f}x) → policy head is control bottleneck.")
    else:
        parts.append(f"Encoder more noise-sensitive → encoder geometry matters for control.")

    if c3_mean < 1.5:
        parts.append(f"Direct state→action achieves {c3_mean:.2f}x cost → encoder not necessary for control.")
    else:
        parts.append(f"Direct mapping fails ({c3_mean:.2f}x cost) → encoder provides essential transformation.")

    return {"C1_recovered": c1_recovered, "C2_policy_sensitive": pol_more_sensitive,
            "C3_direct_works": c3_mean < 1.5,
            "C1_cost_ratio": c1_mean, "C2_enc_ratio": enc_ratios_high,
            "C2_pol_ratio": pol_ratios_high, "C3_direct_ratio": c3_mean,
            "conclusion": " ".join(parts)}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_c_policy_vs_representation(n_seeds=8)
    with open("core_mvp_v4/results/c_policy_vs_representation.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("C:", r["analysis"]["conclusion"])
