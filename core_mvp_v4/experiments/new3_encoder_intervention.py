"""New3: Encoder Intervention — direct hidden state modification.

Ablates/amplifies/injects along control directions in h-space.
Measures behavioral impact of direct h intervention.
Compares with random direction and policy head interventions.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model


def _train(model, env, n_ep, ep_len, seed):
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


def _get_control_direction(model, mode="jacobian", n_states=100):
    if mode == "jacobian":
        env = MultiDimDoubleWell(d=16, k=2, drift=0.5, seed=0, coupling=0.0)
        env.reset()
        Js = []
        for _ in range(n_states):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            s_t.requires_grad = True
            h_t = model.encoder(s_t)
            a_t = model.actor(h_t)
            for j in range(2):
                grad = torch.autograd.grad(a_t[0, j], h_t, retain_graph=True, allow_unused=True)[0]
                if grad is not None:
                    Js.append(grad.squeeze(0).detach().numpy())
            env.step(model.act_numpy(env.get_state()))
        J_mean = np.mean(Js, axis=0)
        return J_mean / (np.linalg.norm(J_mean) + 1e-8)
    elif mode == "random":
        dir = np.random.randn(192)
        return dir / (np.linalg.norm(dir) + 1e-8)
    return np.ones(192) / np.sqrt(192)


def _intervene(model, env, direction, mode="ablate", strength=1.0, n_steps=500):
    env.reset()
    cost_sum = 0.0
    in_zone = 0
    action_diffs = []
    for _ in range(n_steps):
        s = env.get_state()
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h_raw = model.encoder(s_t).squeeze(0).detach().numpy()
        a_orig = model.actor(torch.from_numpy(h_raw.astype(np.float32)).unsqueeze(0))
        a_orig = a_orig.squeeze(0).detach().numpy()

        if mode == "ablate":
            proj = np.dot(h_raw, direction) * direction
            h_mod = h_raw - strength * proj
        elif mode == "amplify":
            proj = np.dot(h_raw, direction) * direction
            h_mod = h_raw + strength * proj
        elif mode == "inject":
            h_mod = h_raw + strength * direction
        else:
            h_mod = h_raw

        a_mod = model.actor(torch.from_numpy(h_mod.astype(np.float32)).unsqueeze(0))
        a_mod = a_mod.squeeze(0).detach().numpy()
        ns, risk, _, _ = env.step(a_mod)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
        action_diffs.append(float(np.linalg.norm(a_mod - a_orig)))
    return cost_sum / n_steps, in_zone / n_steps, float(np.mean(action_diffs))


def run_new3_encoder_intervention(n_seeds=8, d=16, k=2, hd=192,
                                   n_episodes=3, episode_length=2000):
    modes = ["ablate", "amplify", "inject"]
    strengths = [0.5, 1.0, 2.0, 5.0]
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        jac_dir = _get_control_direction(model, "jacobian")
        rand_dir = _get_control_direction(model, "random")

        orig_c, _, _ = _intervene(model, env, jac_dir, "none", 0.0)

        seed_data = {"seed": seed, "orig_cost": orig_c, "jacobian": {}, "random": {}}

        for mode in modes:
            seed_data["jacobian"][mode] = {}
            seed_data["random"][mode] = {}
            for s_val in strengths:
                jc, jz, jd = _intervene(model, env, jac_dir, mode, s_val)
                seed_data["jacobian"][mode][f"str_{s_val}"] = {
                    "cost": jc, "cost_ratio": jc / (orig_c + 1e-6),
                    "zone": jz, "action_dev": jd,
                }
                rc, rz, rd = _intervene(model, env, rand_dir, mode, s_val)
                seed_data["random"][mode][f"str_{s_val}"] = {
                    "cost": rc, "cost_ratio": rc / (orig_c + 1e-6),
                    "zone": rz, "action_dev": rd,
                }
        results["seeds"].append(seed_data)

    results["analysis"] = _analyze(results, modes, strengths)
    return results


def _analyze(results, modes, strengths):
    jac_ratios = []
    rand_ratios = []
    best_info = {"ratio": 1.0, "mode": "", "dir": ""}

    for mode in modes:
        for s_val in strengths:
            jr = float(np.mean([
                s["jacobian"][mode][f"str_{s_val}"]["cost_ratio"]
                for s in results["seeds"]
            ]))
            rr = float(np.mean([
                s["random"][mode][f"str_{s_val}"]["cost_ratio"]
                for s in results["seeds"]
            ]))
            jac_ratios.append(jr)
            rand_ratios.append(rr)
            if abs(jr - 1.0) > abs(best_info["ratio"] - 1.0):
                best_info = {"ratio": jr, "mode": mode, "dir": "jacobian", "strength": s_val}

    max_effect = max(abs(r - 1.0) for r in jac_ratios)
    if max_effect > 0.5:
        conclusion = f"h IS CONTROL SUBSTRATE: Jacobian intervention max ratio={best_info['ratio']:.2f}x at {best_info['mode']}×{best_info['strength']}."
    elif max_effect > 0.2:
        conclusion = f"MODERATE h INFLUENCE: max effect {max_effect:.2f}. h partially mediates control."
    else:
        conclusion = f"h IS PASSIVE PIPE: max effect {max_effect:.2f}. Hidden state is a transmission channel, not control locus."

    return {"jacobian_max_effect": max_effect, "best": best_info, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_new3_encoder_intervention(n_seeds=8)
    with open("core_mvp_v4/results/new3_encoder_intervention.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("New3:", r["analysis"]["conclusion"])
