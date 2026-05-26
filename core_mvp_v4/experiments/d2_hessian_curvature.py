"""D2: Hessian Curvature — verify flatness hypothesis.

Computes ∂²cost/∂h² at operating points via autograd.
Analyzes eigenvalue spectrum and alignment with Jacobian directions.
Tests whether Jacobian directions are flat (low curvature).
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


def _cost_fn(model, h):
    a = model.actor(h.unsqueeze(0)).squeeze(0)
    return torch.sum(a ** 2)


def _hessian_eig(model, h_point):
    h = h_point.detach().clone().requires_grad_(True)
    cost = _cost_fn(model, h)
    grad = torch.autograd.grad(cost, h, create_graph=True)[0]

    hd = h.shape[0]
    hess = torch.zeros(hd, hd)
    for i in range(hd):
        g2 = torch.autograd.grad(grad[i], h, retain_graph=True, allow_unused=True)[0]
        if g2 is not None:
            hess[i] = g2.detach()

    eigvals = torch.linalg.eigvalsh(hess).numpy()
    eigvals = eigvals[eigvals > 0] if np.any(eigvals > 0) else eigvals
    cond = float(eigvals.max() / (eigvals.min() + 1e-8)) if len(eigvals) > 1 else 1.0
    return hess.numpy(), eigvals, cond


def _jacobian_at_model(model, h_point):
    h = h_point.detach().clone().requires_grad_(True)
    a = model.actor(h.unsqueeze(0))
    grads = []
    for j in range(a.shape[1]):
        g = torch.autograd.grad(a[0, j], h, retain_graph=True, allow_unused=True)[0]
        if g is not None:
            grads.append(g.squeeze(0).detach().numpy())
    return np.mean(grads, axis=0)


def run_d2_hessian_curvature(n_seeds=8, d=16, k=2, hd=192,
                             n_episodes=3, episode_length=2000, n_points=30):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        h_points = []
        for _ in range(n_points):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach()
            h_points.append(h)
            env.step(model.act_numpy(env.get_state()))

        cond_numbers = []
        jac_hess_alignments = []
        for hp in h_points[:min(n_points, 10)]:
            hess, evals, cond = _hessian_eig(model, hp)
            cond_numbers.append(cond)

            jac_dir = _jacobian_at_model(model, hp)
            jac_dir = jac_dir / (np.linalg.norm(jac_dir) + 1e-8)

            hess_ev = None
            eigs, evecs = np.linalg.eigh(hess)
            if len(eigs) > 0:
                evecs_top = evecs[:, -min(3, len(eigs)):]
                aligns = [abs(np.dot(jac_dir, evecs_top[:, i])) for i in range(evecs_top.shape[1])]
                jac_hess_alignments.append(float(np.mean(aligns)))

        results["seeds"].append({
            "seed": seed,
            "cond_mean": float(np.mean(cond_numbers)) if cond_numbers else 0.0,
            "cond_std": float(np.std(cond_numbers)) if cond_numbers else 0.0,
            "jac_hess_alignment": float(np.mean(jac_hess_alignments)) if jac_hess_alignments else 0.0,
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    cond_m = float(np.mean([s["cond_mean"] for s in results["seeds"]]))
    align_m = float(np.mean([s["jac_hess_alignment"] for s in results["seeds"]]))

    if align_m < 0.1:
        conclusion = f"JACOBIAN IS FLAT: alignment={align_m:.4f} ≈ 0. Condition={cond_m:.1f}. Jacobian directions have near-zero curvature → single-direction ablation can't change cost."
    elif align_m > 0.5:
        conclusion = f"JACOBIAN IS CURVED: alignment={align_m:.3f}. Jacobian directions have significant curvature."
    else:
        conclusion = f"MODERATE: alignment={align_m:.3f}, condition={cond_m:.1f}."

    return {"condition_number": cond_m, "jac_hess_alignment": align_m,
            "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_d2_hessian_curvature(n_seeds=8)
    with open("core_mvp_v4/results/d2_hessian_curvature.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("D2:", r["analysis"]["conclusion"])
