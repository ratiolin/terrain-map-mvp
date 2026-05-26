"""H3: Subspace as Diagnostic — Early Warning Indicator.

d=32, long rollout with increasing pressure (drift g or noise sigma).
Tracks when alignment_gt drops vs when behavior collapses.
Calculates warning lead time: delta_t = t_behavior_collapse - t_semantic_collapse.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_1samp

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, compute_jacobian, collect_controllability_data
from core_mvp_v4.metrics import compute_k80, alignment


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
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            loss = nn.functional.mse_loss(risk_pred, risk_t) + 0.1 * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def _quick_eval(model, env, d, k):
    env_state = env.save_state()
    test_h, test_C = collect_controllability_data(model, env, n_samples=100)
    probe = LinearRegression().fit(test_h, test_C)
    R2_val = float(probe.score(test_h, test_C))
    env.restore_state(env_state)

    env_state = env.save_state()
    Js = [compute_jacobian(model, env.get_state())]
    for _ in range(30):
        a = model.act_numpy(env.get_state())
        env.step(a)
        Js.append(compute_jacobian(model, env.get_state()))
    J_mean = np.mean(Js, axis=0)
    _, S_mean, Vt_mean = np.linalg.svd(J_mean, full_matrices=False)
    V = Vt_mean.T
    k80_val = compute_k80(S_mean)
    U_true = np.eye(d)[:, :k]
    k_use = min(k, V.shape[1])
    align_gt = float(alignment(V[:, :k_use], U_true[:, :k_use], k=k_use))
    env.restore_state(env_state)
    return {"align": align_gt, "R2": R2_val, "k80": k80_val}


def _rolling_cost(model, env, window=100):
    costs = []
    for _ in range(window):
        a = model.act_numpy(env.get_state())
        ns, risk, _, _ = env.step(a)
        costs.append(risk)
    return float(np.mean(costs))


def run_h3_diagnostic(n_seeds=8, d=32, k=2, hd=384,
                      n_episodes=3, episode_length=1000, total_steps=10000):
    pressure_interval = 2000
    delta_g = 0.1
    results_per_seed = []

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.0, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        time_series = {"step": [], "align": [], "k80": [], "R2": [], "cost": [], "drift": []}
        env.set_drift(0.0)
        env.reset()

        baseline_cost = _rolling_cost(model, env, 200)
        cost_threshold = baseline_cost * 2.0
        semantic_threshold = 0.5

        t_semantic = None
        t_behavior = None

        for t in range(total_steps):
            if t > 0 and t % pressure_interval == 0:
                current_g = env.drift
                env.set_drift(current_g + delta_g)

            if t % 100 == 0:
                m = _quick_eval(model, env, d, k)
                c = _rolling_cost(model, env, 100)
                time_series["step"].append(t)
                time_series["align"].append(m["align"])
                time_series["k80"].append(m["k80"])
                time_series["R2"].append(m["R2"])
                time_series["cost"].append(c)
                time_series["drift"].append(env.drift)

                if t_semantic is None and m["align"] < semantic_threshold:
                    t_semantic = t
                if t_behavior is None and c > cost_threshold:
                    t_behavior = t

            a = model.act_numpy(env.get_state())
            env.step(a)

        delta_t = (t_behavior - t_semantic) if (t_semantic is not None and t_behavior is not None) else None
        results_per_seed.append({
            "seed": seed,
            "t_semantic": t_semantic, "t_behavior": t_behavior,
            "delta_t": delta_t,
            "time_series": time_series,
        })

    delta_ts = [r["delta_t"] for r in results_per_seed if r["delta_t"] is not None]
    if delta_ts:
        dt_mean = float(np.mean(delta_ts))
        dt_std = float(np.std(delta_ts))
        t_stat, p_val = ttest_1samp(delta_ts, 0.0)
    else:
        dt_mean = 0.0
        dt_std = 0.0
        p_val = 1.0

    if dt_mean > 0 and p_val < 0.05:
        conclusion = (f"EARLY WARNING: semantic degradation leads behavior collapse by "
                      f"{dt_mean:.0f}±{dt_std:.0f} steps (p={p_val:.4f}).")
    elif dt_mean < 0 and p_val < 0.05:
        conclusion = (f"LAGGING INDICATOR: behavior collapses before semantics "
                      f"(delta_t={dt_mean:.0f}, p={p_val:.4f}).")
    else:
        conclusion = (f"SYNCHRONOUS: semantic and behavioral collapse near-simultaneous "
                      f"(delta_t={dt_mean:.0f}, p={p_val:.4f}).")

    return {
        "delta_t_mean": dt_mean, "delta_t_std": dt_std,
        "p_value": float(p_val),
        "n_valid_pairs": len(delta_ts),
        "conclusion": conclusion,
        "seeds": results_per_seed,
    }


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_h3_diagnostic(n_seeds=8)
    with open("core_mvp_v4/results/h3_diagnostic.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("H3:", r["conclusion"])
