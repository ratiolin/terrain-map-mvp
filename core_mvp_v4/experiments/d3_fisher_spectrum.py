"""D3: Fisher Information Spectrum.

Computes F = JᵀJ (empirical Fisher) from ∂a/∂h.
Eigenvalue decomposition → effective dimensionality of control information.
Compares Fisher dominant directions with Jacobian directions.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model
from core_mvp_v4.metrics import compute_k80, effective_rank, alignment


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


def run_d3_fisher_spectrum(n_seeds=8, d=16, k=2, hd=192,
                           n_episodes=3, episode_length=2000, n_samples=500):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        F = np.zeros((hd, hd))
        Js_raw = []
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
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        k80_f = compute_k80(np.sqrt(np.maximum(eigvals, 0)))
        eff_dim = (np.sum(np.maximum(eigvals, 0))**2) / (np.sum(np.maximum(eigvals, 0)**2) + 1e-10)
        cumsum = np.cumsum(np.maximum(eigvals, 0)) / (np.sum(np.maximum(eigvals, 0)) + 1e-10)
        top3_frac = float(cumsum[min(2, len(cumsum)-1)])

        J_mean_dir = None
        model.eval()
        for _ in range(50):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            s_t.requires_grad = True
            h_t = model.encoder(s_t)
            a_t = model.actor(h_t)
            for j in range(k):
                grad = torch.autograd.grad(a_t[0, j], h_t, retain_graph=True, allow_unused=True)[0]
                if grad is not None:
                    g = grad.squeeze(0).detach().numpy()
                    if J_mean_dir is None:
                        J_mean_dir = g
                    else:
                        J_mean_dir += g
            env.step(model.act_numpy(env.get_state()))
        J_mean_dir /= (50 * k)
        J_mean_dir /= (np.linalg.norm(J_mean_dir) + 1e-8)

        fisher_top_dirs = eigvecs[:, :3]
        aligns = [abs(np.dot(J_mean_dir, fisher_top_dirs[:, i])) for i in range(min(3, fisher_top_dirs.shape[1]))]
        fisher_jac_align = float(np.mean(aligns)) if aligns else 0.0

        results["seeds"].append({
            "seed": seed,
            "fisher_k80": k80_f,
            "fisher_eff_dim": float(eff_dim),
            "fisher_top3_frac": top3_frac,
            "fisher_jac_align": fisher_jac_align,
            "eigvals_top10": eigvals[:10].tolist(),
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    k80_m = float(np.mean([s["fisher_k80"] for s in results["seeds"]]))
    eff_m = float(np.mean([s["fisher_eff_dim"] for s in results["seeds"]]))
    top3 = float(np.mean([s["fisher_top3_frac"] for s in results["seeds"]]))
    align_m = float(np.mean([s["fisher_jac_align"] for s in results["seeds"]]))

    if eff_m <= 3:
        conclusion = f"LOW-DIM FISHER: eff_dim={eff_m:.1f}, top3={top3:.1%}. Control info concentrated. Fisher-Jacobian align={align_m:.3f}."
    elif eff_m <= 10:
        conclusion = f"MODERATE FISHER: eff_dim={eff_m:.1f}. Control info moderately distributed."
    else:
        conclusion = f"HIGH-DIM FISHER: eff_dim={eff_m:.1f}. Control info uniformly spread — explains single-direction ablation failure."

    return {"fisher_k80": k80_m, "fisher_eff_dim": eff_m, "fisher_top3": top3,
            "fisher_jac_align": align_m, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_d3_fisher_spectrum(n_seeds=8)
    with open("core_mvp_v4/results/d3_fisher_spectrum.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("D3:", r["analysis"]["conclusion"])
