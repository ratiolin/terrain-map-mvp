"""F-B: Local Control Gradient Field.

Estimates ∇_h cost via finite differences at dense h-points.
Builds vector field, measures smoothness, finds integral curves.
Tests if cost changes monotonically along gradient.
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


def _cost_of_h(model, env, h_point, s_original):
    a = model.actor(torch.from_numpy(h_point.astype(np.float32)).unsqueeze(0))
    a = a.squeeze(0).detach().numpy()
    ns = env.forward_static(s_original, a)
    return float(np.linalg.norm(ns[:2]))


def _estimate_gradient(model, env, h_point, s_origin, eps=0.01, n_samples=5):
    d = h_point.shape[0]
    grad = np.zeros(d)
    for _ in range(n_samples):
        delta = np.random.randn(d) * eps / np.sqrt(d)
        h_plus = h_point + delta
        h_minus = h_point - delta
        cost_plus = _cost_of_h(model, env, h_plus, s_origin)
        cost_minus = _cost_of_h(model, env, h_minus, s_origin)
        grad += (cost_plus - cost_minus) * delta / (2 * n_samples * eps**2 / d)
    norm = np.linalg.norm(grad)
    if norm > 1e-8:
        grad = grad / norm
    return grad, norm


def run_f_gradient_field(n_seeds=8, d=16, k=2, hd=192,
                         n_episodes=3, episode_length=2000, n_points=80):
    results = {"seeds": []}

    for seed in range(min(n_seeds, 6)):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        h_points = []; s_origins = []
        for _ in range(n_points):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            h_points.append(h); s_origins.append(s.copy())
            env.step(model.act_numpy(s))

        grads = []
        for h, s0 in zip(h_points, s_origins):
            g, norm = _estimate_gradient(model, env, h, s0)
            grads.append(g)

        smoothness = []
        for i in range(len(grads)):
            for j in range(i + 1, min(i + 5, len(grads))):
                cos = np.dot(grads[i], grads[j]) / (np.linalg.norm(grads[i]) * np.linalg.norm(grads[j]) + 1e-8)
                smoothness.append(float(cos))

        integral_curve_costs = []
        for i in range(min(10, len(h_points))):
            h_cur = h_points[i].copy()
            costs_along = []
            for _ in range(20):
                g, _ = _estimate_gradient(model, env, h_cur, s_origins[i % len(s_origins)])
                h_cur = h_cur - 0.05 * g
                cost_val = _cost_of_h(model, env, h_cur, s_origins[i % len(s_origins)])
                costs_along.append(cost_val)
            if len(costs_along) >= 2:
                rho, _ = (lambda x, y: (np.corrcoef(x, y)[0, 1], 0))(range(len(costs_along)), costs_along)
                integral_curve_costs.append(abs(rho))

        results["seeds"].append({
            "seed": seed,
            "smoothness_mean": float(np.mean(smoothness)) if smoothness else 0.0,
            "smoothness_std": float(np.std(smoothness)) if smoothness else 0.0,
            "integral_monotonic_frac": float(np.mean([1 if r > 0.7 else 0 for r in integral_curve_costs])) if integral_curve_costs else 0.0,
            "integral_rho_mean": float(np.mean(integral_curve_costs)) if integral_curve_costs else 0.0,
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    smooth = float(np.mean([s["smoothness_mean"] for s in results["seeds"]]))
    mono_frac = float(np.mean([s["integral_monotonic_frac"] for s in results["seeds"]]))
    integ_rho = float(np.mean([s["integral_rho_mean"] for s in results["seeds"]]))

    if integ_rho > 0.7 and mono_frac > 0.5:
        conclusion = f"SMOOTH GRADIENT FIELD: ρ={integ_rho:.3f}, {mono_frac:.0%} monotonic. Control works via local gradients."
    elif smooth > 0.5:
        conclusion = f"GRADIENT SMOOTH BUT NOT MONOTONIC: smoothness={smooth:.3f}. Cost landscape is structured but non-monotonic."
    else:
        conclusion = f"CHAOTIC FIELD: smoothness={smooth:.3f}, ρ={integ_rho:.3f}. No coherent gradient structure."
    return {"smoothness": smooth, "integral_rho": integ_rho, "monotonic_frac": mono_frac,
            "conclusion": conclusion}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_f_gradient_field(n_seeds=8)
    with open("core_mvp_v4/results/f_gradient_field.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("F-B:", r["analysis"]["conclusion"])
