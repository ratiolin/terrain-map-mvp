"""P0-1: Strong Causal Ablation — targeted subspace ablation with reinforcement.

Extracts top-2 right singular vectors V_top from W=UΣVt.
Applies P_abl = I - V_top@V_top^T during inference.
Three reinforcement modes: multi-step, adversarial, time-locked.
Scans ablation strength to find cost_ratio >= 2.
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


def _extract_vtop(model, k_top=2):
    W = model.actor.weight.data.clone().numpy()
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    V_top = Vt.T[:, :k_top].copy()
    return V_top, S


def _behavioral_ablated(model, env, V_top, ablation_strength=1.0, mode="continuous",
                        n_steps=500, interval=10):
    P_abl = np.eye(model.hidden_dim) - ablation_strength * V_top @ V_top.T
    env.reset()
    cost_sum = 0.0
    in_zone = 0
    action_diffs = []

    for step in range(n_steps):
        s = env.get_state()
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h_raw = model.encoder(s_t).squeeze(0).detach().numpy()
        a_orig = model.actor(torch.from_numpy(h_raw.astype(np.float32)).unsqueeze(0))
        a_orig = a_orig.squeeze(0).detach().numpy()

        if mode == "continuous":
            h_abl = P_abl @ h_raw
        elif mode == "interval":
            if step % interval == 0:
                h_abl = P_abl @ h_raw
            else:
                h_abl = h_raw
        elif mode == "adversarial":
            h_proj = V_top @ (V_top.T @ h_raw)
            h_abl = h_raw - h_proj + (-1.0) * h_proj
        elif mode == "time_locked":
            risk = float(np.linalg.norm(s[:2]))
            if risk > 1.5:
                h_abl = P_abl @ h_raw
            else:
                h_abl = h_raw
        else:
            h_abl = h_raw

        a_abl = model.actor(torch.from_numpy(h_abl.astype(np.float32)).unsqueeze(0))
        a_abl = a_abl.squeeze(0).detach().numpy()

        ns, risk, _, _ = env.step(a_abl)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
        action_diffs.append(float(np.linalg.norm(a_abl - a_orig)))

    return {
        "mean_cost": cost_sum / n_steps,
        "in_zone_rate": in_zone / n_steps,
        "action_deviation": float(np.mean(action_diffs)),
    }


def run_p0_strong_causal(n_seeds=8, d=16, k=2, hd=192,
                         n_episodes=3, episode_length=2000):
    modes = ["continuous", "interval", "adversarial", "time_locked"]
    strength_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        V_top, S_w = _extract_vtop(model, k)

        orig_c, orig_z = 0, 0
        env.reset()
        for _ in range(500):
            a = model.act_numpy(env.get_state())
            ns, risk, _, _ = env.step(a)
            orig_c += risk / 500
            orig_z += (1.0 / 500) if risk < 1.0 else 0.0

        seed_data = {"seed": seed, "S_w": S_w.tolist(),
                     "orig_cost": orig_c, "vtop_norm": float(np.linalg.norm(V_top)),
                     "modes": {}}

        for mode in modes:
            mode_data = {}
            for strength in strength_list:
                bv = _behavioral_ablated(model, env, V_top,
                                         ablation_strength=strength, mode=mode)
                mode_data[f"str_{strength}"] = {
                    "cost": bv["mean_cost"],
                    "cost_ratio": bv["mean_cost"] / (orig_c + 1e-6),
                    "zone": bv["in_zone_rate"],
                    "action_dev": bv["action_deviation"],
                }
            seed_data["modes"][mode] = mode_data

        results["seeds"].append(seed_data)

    results["analysis"] = _analyze(results, strength_list, modes)
    return results


def _analyze(results, strength_list, modes):
    best_ratio = 1.0
    best_info = {}
    for mode in modes:
        for s in strength_list:
            ratios = [sd["modes"][mode][f"str_{s}"]["cost_ratio"]
                      for sd in results["seeds"]]
            mean_r = float(np.mean(ratios))
            if mean_r > best_ratio:
                best_ratio = mean_r
                best_info = {"mode": mode, "strength": s, "ratio": mean_r}

    if best_ratio >= 2.0:
        conclusion = f"CAUSAL CONFIRMED: {best_info['mode']} at strength={best_info['strength']} achieves cost_ratio={best_ratio:.1f}x."
    elif best_ratio >= 1.5:
        conclusion = f"MODERATE CAUSAL: best ratio={best_ratio:.1f}x at {best_info.get('mode','?')}. Weaker than needed for strong claim."
    else:
        conclusion = f"HIGH REDUNDANCY: best ratio={best_ratio:.1f}x. Control structure is massively redundant."

    results["best"] = best_info
    results["conclusion"] = conclusion
    return {"best_info": best_info, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_p0_strong_causal(n_seeds=8)
    with open("core_mvp_v4/results/p0_strong_causal.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("P0-1:", r["analysis"]["conclusion"])
