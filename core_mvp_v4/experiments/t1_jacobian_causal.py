"""Tier 1-2: Policy Head SVD Destruction — causal validation.

Destroys policy head weight SVD structure and measures behavioral impact.
Methods: remove top-k singular values, shuffle spectrum, randomize directions.
"""

import json
import copy
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


def _behavioral(model, env, n_steps=500):
    env.reset()
    cost_sum = 0.0
    in_zone = 0
    orig_actions = []
    for _ in range(n_steps):
        a = model.act_numpy(env.get_state())
        orig_actions.append(a)
        ns, risk, _, _ = env.step(a)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
    return cost_sum / n_steps, in_zone / n_steps


def _measure_variants(model, env, d, k, W_original):
    results = {}

    U, S, Vt = np.linalg.svd(W_original, full_matrices=False)

    # Remove top-k singular values
    for kval in [1, 2, 3]:
        if kval >= len(S):
            kval = len(S)
        S_mod = S.copy()
        S_mod[:kval] = 0.0
        W_mod = U @ np.diag(S_mod) @ Vt
        model.actor.weight.data = torch.from_numpy(W_mod.astype(np.float32))
        c, z = _behavioral(model, env)
        results[f"remove_top{kval}"] = {"cost": c, "zone": z}

    # Shuffle spectrum
    S_shuffle = S.copy()
    np.random.shuffle(S_shuffle)
    W_mod = U @ np.diag(S_shuffle) @ Vt
    model.actor.weight.data = torch.from_numpy(W_mod.astype(np.float32))
    c, z = _behavioral(model, env)
    results["shuffle_S"] = {"cost": c, "zone": z}

    # Randomize directions
    U_rand = np.random.randn(*U.shape)
    U_rand, _ = np.linalg.qr(U_rand)
    W_mod = U_rand @ np.diag(S) @ Vt
    model.actor.weight.data = torch.from_numpy(W_mod.astype(np.float32))
    c, z = _behavioral(model, env)
    results["randomize_U"] = {"cost": c, "zone": z}

    # Full random replacement
    W_random = np.random.randn(*W_original.shape) * 0.1
    model.actor.weight.data = torch.from_numpy(W_random.astype(np.float32))
    c, z = _behavioral(model, env)
    results["random_full"] = {"cost": c, "zone": z}

    # Restore
    model.actor.weight.data = torch.from_numpy(W_original.astype(np.float32))
    c, z = _behavioral(model, env)
    results["original"] = {"cost": c, "zone": z}

    return results, S


def run_t1_jacobian_causal(n_seeds=8, d=16, k=2, hd=192,
                           n_episodes=3, episode_length=2000):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        W_original = model.actor.weight.data.clone().numpy()
        variants, S_spectrum = _measure_variants(model, env, d, k, W_original)

        results["seeds"].append({
            "seed": seed,
            "variants": variants,
            "S_spectrum": S_spectrum.tolist(),
            "k": k,
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    original_costs = []
    top1_costs = []
    top2_costs = []
    top3_costs = []
    shuffle_costs = []
    rand_costs = []

    for s in results["seeds"]:
        v = s["variants"]
        original_costs.append(v["original"]["cost"])
        if "remove_top1" in v:
            top1_costs.append(v["remove_top1"]["cost"])
        if "remove_top2" in v:
            top2_costs.append(v["remove_top2"]["cost"])
        if "remove_top3" in v:
            top3_costs.append(v["remove_top3"]["cost"])
        shuffle_costs.append(v["shuffle_S"]["cost"])
        rand_costs.append(v["random_full"]["cost"])

    orig_m = float(np.mean(original_costs))
    top1_r = float(np.mean(top1_costs)) / orig_m if top1_costs else 1.0
    top2_r = float(np.mean(top2_costs)) / orig_m if top2_costs else 1.0
    top3_r = float(np.mean(top3_costs)) / orig_m if top3_costs else 1.0
    shuffle_r = float(np.mean(shuffle_costs)) / orig_m
    rand_r = float(np.mean(rand_costs)) / orig_m

    parts = []
    if top1_r > 1.5:
        parts.append(f"Remove top-1 SVD: cost ×{top1_r:.1f} → top singular values are CAUSALLY necessary.")
    else:
        parts.append(f"Remove top-1 SVD: cost ×{top1_r:.1f} → redundant directions exist.")

    if top3_r > 2.0:
        parts.append(f"Remove top-3: cost ×{top3_r:.1f} → near-complete failure.")
    elif top3_r > rand_r * 0.8:
        parts.append(f"Remove top-3 approaches random ({top3_r:.1f} vs {rand_r:.1f}) → 3 directions contain most control.")

    if shuffle_r > 2.0:
        parts.append(f"Shuffle S: cost ×{shuffle_r:.1f} → specific S spectrum required.")
    else:
        parts.append(f"Shuffle S: cost ×{shuffle_r:.1f} → spectrum reshuffling tolerated.")

    return {"top1_ratio": top1_r, "top2_ratio": top2_r, "top3_ratio": top3_r,
            "shuffle_ratio": shuffle_r, "random_ratio": rand_r, "conclusion": " ".join(parts)}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_t1_jacobian_causal(n_seeds=8)
    with open("core_mvp_v4/results/t1_jacobian_causal.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("T1-2:", r["analysis"]["conclusion"])
