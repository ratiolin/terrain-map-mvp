"""New2: Nonlinear Control Direction Discovery.

Learns δh that maximizes action change ||a(h+δh)-a(h)|| via gradient optimization.
Compares with Jacobian directions and random directions.
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


def _optimize_direction(model, h_point, eps, n_steps=200):
    delta = nn.Parameter(torch.randn(h_point.shape) * 0.01)
    opt = torch.optim.Adam([delta], lr=0.001)
    h_fixed = h_point.detach()
    a_orig = model.actor(h_fixed.unsqueeze(0)).squeeze(0).detach()

    for _ in range(n_steps):
        delta_norm = delta / (torch.norm(delta) + 1e-8)
        h_pert = h_fixed + eps * delta_norm
        a_pert = model.actor(h_pert.unsqueeze(0)).squeeze(0)
        loss = -torch.norm(a_pert - a_orig)
        opt.zero_grad()
        loss.backward()
        opt.step()

    delta_norm = delta.data / (torch.norm(delta.data) + 1e-8)
    h_pert = h_fixed + eps * delta_norm
    a_pert = model.actor(h_pert.unsqueeze(0)).squeeze(0).detach()
    action_change = float(torch.norm(a_pert - a_orig))
    return delta_norm.detach().numpy(), action_change


def _jacobian_response(model, h_point, eps):
    h_orig = h_point.detach()
    a_orig = model.actor(h_orig.unsqueeze(0)).squeeze(0)
    h_orig.requires_grad = True
    a_jac = model.actor(h_orig.unsqueeze(0))
    grads = []
    for j in range(a_jac.shape[1]):
        g = torch.autograd.grad(a_jac[0, j], h_orig, retain_graph=True, allow_unused=True)[0]
        if g is not None:
            grads.append(g.squeeze(0))
    J_dir = torch.stack(grads).mean(dim=0)
    J_dir = J_dir / (torch.norm(J_dir) + 1e-8)
    h_pert = h_orig + eps * J_dir
    a_pert = model.actor(h_pert.unsqueeze(0)).squeeze(0)
    return float(torch.norm(a_pert - a_orig))


def _random_response(model, h_point, eps, n_trials=20):
    changes = []
    for _ in range(n_trials):
        rand_dir = torch.randn(h_point.shape)
        rand_dir = rand_dir / (torch.norm(rand_dir) + 1e-8)
        a_orig = model.actor(h_point.unsqueeze(0)).squeeze(0)
        h_pert = h_point + eps * rand_dir
        a_pert = model.actor(h_pert.unsqueeze(0)).squeeze(0)
        changes.append(float(torch.norm(a_pert - a_orig)))
    return float(np.mean(changes)), float(np.std(changes))


def run_new2_nonlinear_direction(n_seeds=8, d=16, k=2, hd=192,
                                 n_episodes=3, episode_length=2000, n_points=50):
    eps_list = [0.01, 0.05, 0.1]
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        h_points = []
        for _ in range(n_points):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0)
            h_points.append(h)
            env.step(model.act_numpy(s))

        seed_data = {"seed": seed, "epsilons": {}}
        for eps in eps_list:
            opt_changes = []; jac_changes = []; rand_means = []
            angles_opt_jac = []

            for hp in h_points:
                opt_dir, opt_change = _optimize_direction(model, hp, eps)
                jac_change = _jacobian_response(model, hp, eps)
                rand_m, _ = _random_response(model, hp, eps)

                opt_changes.append(opt_change)
                jac_changes.append(jac_change)
                rand_means.append(rand_m)

                J_dir = None
                h_tmp = hp.detach().clone().requires_grad_(True)
                a_tmp = model.actor(h_tmp.unsqueeze(0))
                grads = [torch.autograd.grad(a_tmp[0, j], h_tmp, retain_graph=True, allow_unused=True)[0]
                         for j in range(k)]
                J_vec = torch.stack([g.squeeze(0) for g in grads if g is not None]).mean(dim=0)
                J_vec = J_vec / (torch.norm(J_vec) + 1e-8)
                cos = float(np.abs(np.dot(opt_dir, J_vec.numpy())))
                angles_opt_jac.append(1.0 - cos)

            seed_data["epsilons"][f"eps_{eps}"] = {
                "opt_action_change": float(np.mean(opt_changes)),
                "jac_action_change": float(np.mean(jac_changes)),
                "rand_action_change": float(np.mean(rand_means)),
                "opt_vs_rand_ratio": float(np.mean(opt_changes) / (np.mean(rand_means) + 1e-6)),
                "opt_vs_jac_ratio": float(np.mean(opt_changes) / (np.mean(jac_changes) + 1e-6)),
                "opt_jac_angle": float(np.mean(angles_opt_jac)),
            }

        results["seeds"].append(seed_data)

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    eps_key = "eps_0.1"
    opt_changes = [s["epsilons"][eps_key]["opt_action_change"] for s in results["seeds"]]
    jac_changes = [s["epsilons"][eps_key]["jac_action_change"] for s in results["seeds"]]
    rand_changes = [s["epsilons"][eps_key]["rand_action_change"] for s in results["seeds"]]
    angles = [s["epsilons"][eps_key]["opt_jac_angle"] for s in results["seeds"]]

    opt_m = float(np.mean(opt_changes))
    jac_m = float(np.mean(jac_changes))
    rand_m = float(np.mean(rand_changes))
    ratio_opt_rand = opt_m / (rand_m + 1e-6)
    angle_m = float(np.mean(angles))

    if ratio_opt_rand > 3.0:
        conclusion = f"OPTIMIZED DIRECTION {ratio_opt_rand:.1f}x vs random. Angle w/ Jacobian={angle_m:.3f}."
    elif ratio_opt_rand > 1.5:
        conclusion = f"MODERATE ADVANTAGE: {ratio_opt_rand:.1f}x vs random."
    else:
        conclusion = f"NO ADVANTAGE: optimized direction similar to random ({ratio_opt_rand:.1f}x)."

    return {"opt_change": opt_m, "jac_change": jac_m, "rand_change": rand_m,
            "ratio_opt_rand": ratio_opt_rand, "opt_jac_angle": angle_m,
            "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_new2_nonlinear_direction(n_seeds=8)
    with open("core_mvp_v4/results/new2_nonlinear_direction.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("New2:", r["analysis"]["conclusion"])
