"""V4-3: Gradient Integration Policy.

h' = h - alpha * g(h) for K steps before producing action.
Scans alpha and K. Compares with original policy.
"""

import json
import numpy as np
import torch; import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model


def _train(model, env, n_ep, ep_len, seed):
    if seed is not None: torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            ns, risk, _, _ = env.step(action.squeeze(0).detach().numpy())
            loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def _cost_of_h(model, env, h, s0):
    a = model.actor(torch.from_numpy(h.astype(np.float32)).unsqueeze(0)).squeeze(0).detach().numpy()
    ns = env.forward_static(s0, a)
    return float(np.linalg.norm(ns[:2]))


def _grad_h(model, env, h, s0, eps=0.01):
    d = len(h); g = np.zeros(d)
    for _ in range(10):
        delta = np.random.randn(d) * eps / np.sqrt(d)
        g += (_cost_of_h(model, env, h + delta, s0) - _cost_of_h(model, env, h - delta, s0)) * delta / (20 * eps**2 / d)
    norm = np.linalg.norm(g)
    return g / norm if norm > 1e-8 else g


def _run_policy(model, env, alpha, K_integrate, n_steps=500):
    env.reset()
    total_cost = 0.0; in_zone = 0
    for _ in range(n_steps):
        s = env.get_state()
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h = model.encoder(s_t).squeeze(0).detach().numpy()
        for _ in range(K_integrate):
            g = _grad_h(model, env, h, s)
            h = h - alpha * g
        a = model.actor(torch.from_numpy(h.astype(np.float32)).unsqueeze(0)).squeeze(0).detach().numpy()
        ns, risk, _, _ = env.step(a)
        total_cost += risk
        if risk < 1.0: in_zone += 1
    return total_cost / n_steps, in_zone / n_steps


def run_v4_3_policy(n_seeds=4, d=16, k=2, hd=192, n_episodes=3, episode_length=2000):
    ep_len = episode_length
    alphas = [0.01, 0.05, 0.1]
    Ks = [1, 3, 5]
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, ep_len, seed)

        orig_cost, orig_zone = _run_policy(model, env, 0, 0)
        seed_data = {"seed": seed, "original": {"cost": orig_cost, "zone": orig_zone}, "sweep": {}}
        for alpha in alphas:
            for K in Ks:
                c, z = _run_policy(model, env, alpha, K)
                seed_data["sweep"][f"a{alpha}_K{K}"] = {"cost": c, "cost_ratio": c / (orig_cost + 1e-6), "zone": z}
        results["seeds"].append(seed_data)

    best_ratio = 1.0; best_cfg = ""
    for alpha in alphas:
        for K in Ks:
            ratios = [s["sweep"][f"a{alpha}_K{K}"]["cost_ratio"] for s in results["seeds"]]
            mr = float(np.mean(ratios))
            if (mr < best_ratio - 0.01):
                best_ratio = mr; best_cfg = f"α={alpha}, K={K}"
    if best_ratio < 0.9:
        conclusion = f"GRADIENT POLICY IMPROVES: {best_cfg} reduces cost to {best_ratio:.2f}x."
    elif best_ratio < 1.0:
        conclusion = f"MARGINAL: {best_cfg} gives {best_ratio:.2f}x."
    else:
        conclusion = f"NO IMPROVEMENT: best ratio={best_ratio:.2f}. Gradient integration not beneficial."
    results["analysis"] = {"best_ratio": best_ratio, "best_config": best_cfg, "conclusion": conclusion}
    return results


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_v4_3_policy(n_seeds=4)
    with open("core_mvp_v4/results/g3_policy.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("V4-3:", r["analysis"]["conclusion"])
