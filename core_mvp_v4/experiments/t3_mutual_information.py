"""Tier 3-6: Mutual Information — I(s;a) vs I(h;a).

Information-theoretic verification that control information does not pass through h.
Uses binning-based mutual information estimation.
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


def _mutual_info(x, y, n_bins=20):
    """Binning-based mutual information estimate."""
    try:
        from sklearn.feature_selection import mutual_info_regression
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        mi = mutual_info_regression(x, y)
        return float(np.mean(mi))
    except Exception:
        x_d = x.ravel() if x.ndim > 1 else x
        y_d = y.ravel() if y.ndim > 1 else y
        hist_2d, _, _ = np.histogram2d(x_d, y_d, bins=n_bins)
        p_xy = hist_2d / hist_2d.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        return float(mi)


def run_t3_mutual_information(n_seeds=8, d=16, k=2, hd=192,
                              n_episodes=3, episode_length=2000, n_traj=1000):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        s_all = []
        h_all = []
        a_all = []
        for _ in range(n_traj):
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
        a_arr = np.array(a_all)

        I_s_a = np.mean([_mutual_info(s_arr, a_arr[:, j]) for j in range(k)])
        I_h_a = np.mean([_mutual_info(h_arr, a_arr[:, j]) for j in range(k)])
        I_s_h = np.mean([_mutual_info(s_arr, h_arr[:, j]) for j in range(min(hd, 5))])

        results["seeds"].append({
            "seed": seed,
            "I_s_a": I_s_a, "I_h_a": I_h_a, "I_s_h": I_s_h,
            "I_ratio_ha_vs_sa": I_h_a / max(I_s_a, 1e-6),
        })

    I_s_a_mean = float(np.mean([s["I_s_a"] for s in results["seeds"]]))
    I_h_a_mean = float(np.mean([s["I_h_a"] for s in results["seeds"]]))
    I_s_h_mean = float(np.mean([s["I_s_h"] for s in results["seeds"]]))
    ratio = I_h_a_mean / max(I_s_a_mean, 1e-6)

    if ratio < 0.1:
        conclusion = f"CONTROL BYPASSES H: I(h;a)={I_h_a_mean:.3f} << I(s;a)={I_s_a_mean:.3f} (ratio={ratio:.4f}). Information bottleneck confirmed."
    elif ratio < 0.5:
        conclusion = f"PARTIAL BYPASS: I(h;a)/I(s;a)={ratio:.3f}. Some control info passes through h."
    else:
        conclusion = f"SIGNIFICANT FLOW: I(h;a)/I(s;a)={ratio:.3f}. h carries substantial control information."

    results["analysis"] = {
        "I_s_a": I_s_a_mean, "I_h_a": I_h_a_mean, "I_s_h": I_s_h_mean,
        "ratio": ratio, "conclusion": conclusion,
    }
    return results


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_t3_mutual_information(n_seeds=8)
    with open("core_mvp_v4/results/t3_mutual_information.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("T3-6:", r["analysis"]["conclusion"])
