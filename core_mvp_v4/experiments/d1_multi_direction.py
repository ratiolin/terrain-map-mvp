"""D1: Multi-Direction Joint Intervention.

Ablates top-k Jacobian right singular vectors simultaneously.
Scans k=1..20. Includes random direction control and
continuous attenuation factor.
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


def _get_jacobian_V(model, n_states=100):
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
    J_mean = np.mean(Js, axis=0).reshape(1, -1)
    _, S, Vt = np.linalg.svd(J_mean, full_matrices=False)
    return Vt.T, S


def _behavioral_ablated(model, env, V_k, k, mode="hard", alpha=0.0, n_steps=500):
    hd = V_k.shape[0]
    if mode == "hard":
        P = np.eye(hd) - V_k[:, :k] @ V_k[:, :k].T
    elif mode == "attenuate":
        I = np.eye(hd)
        V = V_k[:, :k]
        P = I - (1 - alpha) * V @ V.T

    env.reset()
    cost_sum = 0.0
    in_zone = 0
    diffs = []
    for _ in range(n_steps):
        s = env.get_state()
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h = model.encoder(s_t).squeeze(0).detach().numpy()
        a_orig = model.actor(torch.from_numpy(h.astype(np.float32)).unsqueeze(0))
        a_orig = a_orig.squeeze(0).detach().numpy()

        h_mod = P @ h
        a_mod = model.actor(torch.from_numpy(h_mod.astype(np.float32)).unsqueeze(0))
        a_mod = a_mod.squeeze(0).detach().numpy()

        ns, risk, _, _ = env.step(a_mod)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
        diffs.append(float(np.linalg.norm(a_mod - a_orig)))
    return cost_sum / n_steps, in_zone / n_steps, float(np.mean(diffs))


def _random_ortho_directions(hd, k):
    A = np.random.randn(hd, k)
    Q, _ = np.linalg.qr(A)
    return Q


def run_d1_multi_direction(n_seeds=8, d=16, k=2, hd=192,
                           n_episodes=3, episode_length=2000):
    k_list = [1, 2, 3, 4, 5, 7, 10, 15, 20]
    alpha_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        V_jac, S_jac = _get_jacobian_V(model)
        baseline_c, _, _ = _behavioral_ablated(model, env, V_jac, 0, n_steps=500)

        seed_data = {"seed": seed, "baseline_cost": baseline_c,
                     "hard_ablation": {}, "random_ablation": {}, "attenuation": {}}

        for kval in k_list:
            if kval > V_jac.shape[1]:
                continue
            c, z, dev = _behavioral_ablated(model, env, V_jac, kval, "hard")
            seed_data["hard_ablation"][str(kval)] = {
                "cost": c, "cost_ratio": c / (baseline_c + 1e-6),
                "zone": z, "action_dev": dev,
            }

            V_rand = _random_ortho_directions(hd, kval)
            c_r, z_r, dev_r = _behavioral_ablated(model, env, V_rand, kval, "hard")
            seed_data["random_ablation"][str(kval)] = {
                "cost": c_r, "cost_ratio": c_r / (baseline_c + 1e-6),
                "action_dev": dev_r,
            }

        # Continuous attenuation: fix k at 5
        k_fixed = min(5, V_jac.shape[1])
        for alpha in alpha_list:
            c_a, z_a, dev_a = _behavioral_ablated(model, env, V_jac, k_fixed, "attenuate", alpha)
            seed_data["attenuation"][str(alpha)] = {
                "cost": c_a, "cost_ratio": c_a / (baseline_c + 1e-6),
                "action_dev": dev_a,
            }

        results["seeds"].append(seed_data)

    results["analysis"] = _analyze(results, k_list, alpha_list)
    return results


def _analyze(results, k_list, alpha_list):
    jac_ratios = {}
    rand_ratios = {}
    for kval in k_list:
        jac_ratios[kval] = float(np.mean([
            s["hard_ablation"].get(str(kval), {}).get("cost_ratio", 1.0)
            for s in results["seeds"]
        ]))
        rand_ratios[kval] = float(np.mean([
            s["random_ablation"].get(str(kval), {}).get("cost_ratio", 1.0)
            for s in results["seeds"]
        ]))

    max_jac = max(jac_ratios.values())
    max_k = max(jac_ratios, key=jac_ratios.get)

    atten = {}
    for alpha in alpha_list:
        atten[alpha] = float(np.mean([
            s["attenuation"].get(str(alpha), {}).get("cost_ratio", 1.0)
            for s in results["seeds"]
        ]))

    jac_vs_rand = max_jac / rand_ratios[max_k] if max_k in rand_ratios else 1.0

    if max_jac > 1.5:
        conclusion = f"MULTI-DIRECTION WINS: k={max_k} achieves cost_ratio={max_jac:.2f}x."
    elif max_jac > 1.2:
        conclusion = f"WEAK MULTI-EFFECT: max ratio={max_jac:.2f}x at k={max_k}."
    else:
        conclusion = f"HIGHLY REDUNDANT: max ratio={max_jac:.2f}x even at k={max_k}. Control dispersed beyond top-20 Jacobian directions."

    return {"jac_ratios": jac_ratios, "rand_ratios": rand_ratios,
            "max_ratio": max_jac, "max_k": max_k, "jac_vs_rand": jac_vs_rand,
            "attenuation": atten, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_d1_multi_direction(n_seeds=8)
    with open("core_mvp_v4/results/d1_multi_direction.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("D1:", r["analysis"]["conclusion"])
