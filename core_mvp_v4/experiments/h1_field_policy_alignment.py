"""H1: Field-Policy Alignment.

Measures cos_sim between gradient g = ∇_h cost and policy Jacobian v_J.
Per-policy-direction and per-environment-condition analysis.
Causal intervention: compare g-directed vs J-directed perturbation effects.
"""

import json
import numpy as np
import torch
import torch.nn as nn

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
    return float(np.linalg.norm(env.forward_static(s0, a)[:2]))


def _grad_h(model, env, h, s0, eps=0.005):
    d = len(h); g = np.zeros(d)
    for _ in range(8):
        delta = np.random.randn(d) * eps / np.sqrt(d)
        g += (_cost_of_h(model, env, h + delta, s0) - _cost_of_h(model, env, h - delta, s0)) * delta / (16 * eps**2 / d)
    return g / (np.linalg.norm(g) + 1e-8)


def _jacobian_dir(model, h):
    h_t = torch.from_numpy(h.astype(np.float32)).unsqueeze(0).requires_grad_(True)
    a_t = model.actor(h_t)
    grads = [torch.autograd.grad(a_t[0, j], h_t, retain_graph=True, allow_unused=True)[0]
             for j in range(a_t.shape[1]) if a_t.shape[1] > 0]
    if not grads: return np.zeros(len(h))
    J = torch.stack([g.squeeze(0) for g in grads if g is not None], dim=0).detach().numpy()
    if J.shape[0] < 2: return J.flatten() / (np.linalg.norm(J) + 1e-8)
    _, _, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt[0] / (np.linalg.norm(Vt[0]) + 1e-8)


def run_h1_field_policy_alignment(n_seeds=4, d=16, k=2, hd=192,
                                   n_episodes=3, episode_length=2000, n_points=500):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        cos_global = []; cos_high_drift = []; cos_low_drift = []
        cos_near_barrier = []; cos_near_center = []
        g_effects = []; j_effects = []; proj_effects = []; rand_effects = []

        for _ in range(n_points):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            g = _grad_h(model, env, h, s)
            v_j = _jacobian_dir(model, h)
            cos_val = float(np.dot(g, v_j))
            cos_global.append(cos_val)

            drift_val = env.drift
            s_ctrl_norm = np.linalg.norm(s[:2])
            if drift_val > 0.8: cos_high_drift.append(cos_val)
            else: cos_low_drift.append(cos_val)
            if s_ctrl_norm > 1.0: cos_near_barrier.append(cos_val)
            else: cos_near_center.append(cos_val)

            c_base = _cost_of_h(model, env, h, s)
            h_g = h - 0.05 * g; c_g = _cost_of_h(model, env, h_g, s)
            h_j = h - 0.05 * v_j; c_j = _cost_of_h(model, env, h_j, s)
            proj = np.dot(g, v_j) * v_j; h_proj = h - 0.05 * proj
            c_proj = _cost_of_h(model, env, h_proj, s)
            r_dir = np.random.randn(len(h)); r_dir /= np.linalg.norm(r_dir) + 1e-8
            h_r = h - 0.05 * r_dir; c_r = _cost_of_h(model, env, h_r, s)

            g_effects.append(c_g - c_base); j_effects.append(c_j - c_base)
            proj_effects.append(c_proj - c_base); rand_effects.append(c_r - c_base)
            env.step(model.act_numpy(s))

        results["seeds"].append({
            "seed": seed,
            "cos_mean": float(np.mean(cos_global)), "cos_median": float(np.median(cos_global)),
            "cos_high_drift": float(np.mean(cos_high_drift)) if cos_high_drift else 0,
            "cos_low_drift": float(np.mean(cos_low_drift)) if cos_low_drift else 0,
            "cos_barrier": float(np.mean(cos_near_barrier)) if cos_near_barrier else 0,
            "cos_center": float(np.mean(cos_near_center)) if cos_near_center else 0,
            "g_effect_mean": float(np.mean(g_effects)), "j_effect_mean": float(np.mean(j_effects)),
            "proj_effect_mean": float(np.mean(proj_effects)), "rand_effect_mean": float(np.mean(rand_effects)),
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    cos_m = float(np.mean([s["cos_mean"] for s in results["seeds"]]))
    g_eff = float(np.mean([s["g_effect_mean"] for s in results["seeds"]]))
    j_eff = float(np.mean([s["j_effect_mean"] for s in results["seeds"]]))
    r_eff = float(np.mean([s["rand_effect_mean"] for s in results["seeds"]]))
    g_vs_r = abs(g_eff) / (abs(r_eff) + 1e-6) if abs(r_eff) > 1e-6 else 1.0

    if cos_m > 0.7 and g_vs_r > 2:
        conclusion = f"STRONG ALIGNMENT: cos={cos_m:.3f}, grad effect {g_vs_r:.1f}x random. Policy ≈ local gradient descent."
    elif cos_m < 0.3:
        conclusion = f"WEAK ALIGNMENT: cos={cos_m:.3f}. Gradient is epiphenomenal; policy uses other structure."
    else:
        conclusion = f"MIXED: cos={cos_m:.3f}, g/r={g_vs_r:.1f}x."
    return {"cos_mean": cos_m, "g_effect": g_eff, "g_vs_random": g_vs_r, "conclusion": conclusion}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_h1_field_policy_alignment(n_seeds=4)
    with open("core_mvp_v4/results/h1_field_policy_alignment.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("H1:", r["analysis"]["conclusion"])
