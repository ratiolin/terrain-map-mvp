"""D4: Fisher-Directed Multi-Direction Intervention.

Uses Fisher dominant eigenvectors for multi-direction ablation.
Scans m from 1 to Fisher effective dimension. Compares with
Jacobian/PCA/random directions. Includes adversarial injection.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model
from core_mvp_v4.metrics import compute_k80


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


def _compute_fisher_basis(model, n_samples=500, hd=192, k=2):
    env = MultiDimDoubleWell(d=16, k=k, drift=0.5, seed=0, coupling=0.0)
    env.reset()
    F = np.zeros((hd, hd))
    for _ in range(n_samples):
        s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
        s_t.requires_grad = True
        h_t = model.encoder(s_t)
        a_t = model.actor(h_t)
        for j in range(k):
            grad = torch.autograd.grad(a_t[0, j], h_t, retain_graph=True, allow_unused=True)[0]
            if grad is not None:
                g = grad.squeeze(0).detach().numpy()
                F += np.outer(g, g)
        env.step(model.act_numpy(env.get_state()))
    F /= n_samples
    eigvals, eigvecs = np.linalg.eigh(F)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order], eigvals[order]


def _random_ortho(hd, m):
    A = np.random.randn(hd, m)
    Q, _ = np.linalg.qr(A)
    return Q


def _behavioral_ablated_dir(model, env, V, m, mode="hard", n_steps=500):
    hd = V.shape[0]
    P = np.eye(hd) - V[:, :m] @ V[:, :m].T

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

        if mode == "adversarial":
            proj = V[:, :m] @ (V[:, :m].T @ h)
            h_mod = h - proj - 2.0 * proj
        else:
            h_mod = P @ h

        a_mod = model.actor(torch.from_numpy(h_mod.astype(np.float32)).unsqueeze(0))
        a_mod = a_mod.squeeze(0).detach().numpy()
        ns, risk, _, _ = env.step(a_mod)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
        diffs.append(float(np.linalg.norm(a_mod - a_orig)))
    return cost_sum / n_steps, in_zone / n_steps, float(np.mean(diffs))


def run_d4_fisher_intervention(n_seeds=8, d=16, k=2, hd=192,
                               n_episodes=3, episode_length=2000):
    m_list = [1, 2, 3, 5, 10, 20]
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        V_fisher, S_fisher = _compute_fisher_basis(model, n_samples=300, hd=hd, k=k)
        V_rand_all = _random_ortho(hd, max(m_list))
        baseline_c, _, _ = _behavioral_ablated_dir(model, env, V_fisher, 0, n_steps=500)

        seed_data = {"seed": seed, "baseline_cost": baseline_c,
                     "fisher_ablation": {}, "random_ablation": {},
                     "fisher_adversarial": {}}

        for m in m_list:
            if m > V_fisher.shape[1]:
                continue
            fc, fz, fd = _behavioral_ablated_dir(model, env, V_fisher, m, "hard")
            seed_data["fisher_ablation"][str(m)] = {
                "cost": fc, "cost_ratio": fc / (baseline_c + 1e-6),
                "action_dev": fd,
            }
            rc, rz, rd = _behavioral_ablated_dir(model, env, V_rand_all, m, "hard")
            seed_data["random_ablation"][str(m)] = {
                "cost": rc, "cost_ratio": rc / (baseline_c + 1e-6),
                "action_dev": rd,
            }
            ac, az, ad = _behavioral_ablated_dir(model, env, V_fisher, m, "adversarial")
            seed_data["fisher_adversarial"][str(m)] = {
                "cost": ac, "cost_ratio": ac / (baseline_c + 1e-6),
                "action_dev": ad,
            }

        results["seeds"].append(seed_data)

    results["analysis"] = _analyze(results, m_list)
    return results


def _analyze(results, m_list):
    fisher_ratios = {}
    rand_ratios = {}
    adv_ratios = {}
    for m in m_list:
        fisher_ratios[m] = float(np.mean([
            s["fisher_ablation"].get(str(m), {}).get("cost_ratio", 1.0)
            for s in results["seeds"]
        ]))
        rand_ratios[m] = float(np.mean([
            s["random_ablation"].get(str(m), {}).get("cost_ratio", 1.0)
            for s in results["seeds"]
        ]))
        adv_ratios[m] = float(np.mean([
            s["fisher_adversarial"].get(str(m), {}).get("cost_ratio", 1.0)
            for s in results["seeds"]
        ]))

    max_f = max(fisher_ratios.values())
    max_adv = max(adv_ratios.values())
    max_m = max(fisher_ratios, key=fisher_ratios.get)

    if max_adv > 2.0:
        conclusion = f"FISHER ADVERSARIAL BREAKS THROUGH: max cost_ratio={max_adv:.1f}x with adversarial injection."
    elif max_f > 1.5:
        conclusion = f"FISHER MULTI-DIRECTION: max ratio={max_f:.2f}x at m={max_m}. Control information is multi-dimensional."
    elif max_adv > 1.3:
        conclusion = f"WEAK FISHER: max hard={max_f:.2f}x, max adv={max_adv:.2f}x. Fisher directions partially capture control."
    else:
        conclusion = f"NO BREAKTHROUGH: max hard={max_f:.2f}x, max adv={max_adv:.2f}x. Control beyond linear Fisher structure."

    return {"fisher_max_ratio": max_f, "adv_max_ratio": max_adv,
            "max_m": max_m, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_d4_fisher_intervention(n_seeds=8)
    with open("core_mvp_v4/results/d4_fisher_intervention.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("D4:", r["analysis"]["conclusion"])
