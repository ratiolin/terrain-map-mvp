"""P1-3: Information Pathway Decomposition.

Trains three probes to decompose control information:
- f(s): state → action
- f(h): hidden → action
- f(s,h): joint → action

Computes synergy, unique contributions, redundancy.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

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


def _probe_r2(X, y):
    split = int(len(X) * 0.7)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    model = LinearRegression().fit(X_tr, y_tr)
    return float(model.score(X_te, y_te))


def run_p1_information_pathways(n_seeds=8, d=16, k=2, hd=192,
                                n_episodes=3, episode_length=2000, n_samples=1000):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        s_all = []
        h_all = []
        a_all = []
        for _ in range(n_samples):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            a = model.act_numpy(s)
            s_all.append(s)
            h_all.append(h)
            a_all.append(a)
            env.step(a)

        s_arr = np.array(s_all)
        h_arr = np.array(h_all)
        sh_arr = np.concatenate([s_arr, h_arr], axis=1)
        a_arr = np.array(a_all)

        R2_s = np.mean([_probe_r2(s_arr, a_arr[:, j]) for j in range(k)])
        R2_h = np.mean([_probe_r2(h_arr, a_arr[:, j]) for j in range(k)])
        R2_sh = np.mean([_probe_r2(sh_arr, a_arr[:, j]) for j in range(k)])

        synergy = R2_sh - max(R2_s, R2_h)
        unique_s = R2_sh - R2_h
        unique_h = R2_sh - R2_s
        redundancy = max(0, R2_s + R2_h - R2_sh)

        total = unique_s + unique_h + redundancy + synergy
        if total > 1e-6:
            unique_s_frac = unique_s / total
            unique_h_frac = unique_h / total
            synergy_frac = synergy / total
            redundancy_frac = redundancy / total
        else:
            unique_s_frac = unique_h_frac = synergy_frac = redundancy_frac = 0.0

        results["seeds"].append({
            "seed": seed,
            "R2_s": R2_s, "R2_h": R2_h, "R2_sh": R2_sh,
            "unique_s": unique_s, "unique_h": unique_h,
            "synergy": synergy, "redundancy": redundancy,
            "unique_s_frac": unique_s_frac, "unique_h_frac": unique_h_frac,
            "synergy_frac": synergy_frac,
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    us = float(np.mean([s["unique_s_frac"] for s in results["seeds"]]))
    uh = float(np.mean([s["unique_h_frac"] for s in results["seeds"]]))
    sy = float(np.mean([s["synergy_frac"] for s in results["seeds"]]))
    r2_s = float(np.mean([s["R2_s"] for s in results["seeds"]]))
    r2_h = float(np.mean([s["R2_h"] for s in results["seeds"]]))
    r2_sh = float(np.mean([s["R2_sh"] for s in results["seeds"]]))

    parts = []
    if us > 0.7:
        parts.append(f"unique(s)={us:.1%} dominates → h is pure feature preprocessing.")
    elif uh > 0.2:
        parts.append(f"unique(h)={uh:.1%} → encoder actively constructs control-relevant features.")
    if sy > 0.1:
        parts.append(f"synergy={sy:.1%} → h synergistically aids control.")
    if us < 0.5 and uh > 0.3:
        parts.append(f"Shared control: s and h both contribute independently.")

    return {"R2_s": r2_s, "R2_h": r2_h, "R2_sh": r2_sh,
            "unique_s": us, "unique_h": uh, "synergy": sy,
            "conclusion": " ".join(parts) or f"R2_s={r2_s:.3f}, R2_h={r2_h:.3f}, synergy={sy:.3f}"}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_p1_information_pathways(n_seeds=8)
    with open("core_mvp_v4/results/p1_information_pathways.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("P1-3:", r["analysis"]["conclusion"])
