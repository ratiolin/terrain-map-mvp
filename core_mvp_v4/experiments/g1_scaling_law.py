"""G1: Attribution Bias Scaling Law — high-dim failure boundary.

Tests split encoder and guided attention at d ∈ [16, 32, 64, 128]
with hidden_dim/d = 12 (sufficient capacity). Compares against baseline.

Finds the dimension where each architecture's alignment_gt drops below 0.7.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import (
    V4Model, SplitV4Model, GuidedAttentionModel,
    collect_controllability_data, compute_jacobian,
)
from core_mvp_v4.metrics import compute_k80, alignment


def _measure(model, env, d, k):
    test_h, test_C = collect_controllability_data(model, env, n_samples=300)
    probe = LinearRegression().fit(test_h, test_C)
    R2 = float(probe.score(test_h, test_C))

    env.reset()
    jac_states = [env.get_state()]
    for _ in range(100):
        a = model.act_numpy(env.get_state())
        env.step(a)
        jac_states.append(env.get_state())
    J_mean = np.mean([compute_jacobian(model, s) for s in jac_states[::3]], axis=0)
    _, S_mean, Vt_mean = np.linalg.svd(J_mean, full_matrices=False)
    V = Vt_mean.T
    k80 = compute_k80(S_mean)
    U_true = np.eye(d)[:, :k]
    k_use = min(k, V.shape[1])
    align_gt = float(alignment(V[:, :k_use], U_true[:, :k_use], k=k_use))
    return {"R2": R2, "alignment_gt": align_gt, "k80": k80}


def _behavioral(model, env, d, k, n_steps=500):
    env.reset()
    cost_sum = 0.0
    in_zone = 0
    for _ in range(n_steps):
        a = model.act_numpy(env.get_state())
        ns, risk, _, _ = env.step(a)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
    return {
        "mean_cost": cost_sum / n_steps,
        "in_zone_rate": in_zone / n_steps,
    }


def _train_vanilla(model, env, n_ep, ep_len, seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            outputs = model(s_t)
            action, h, risk_pred = outputs[0], outputs[1], outputs[2]
            a_np = action.squeeze(0).detach().numpy()
            ns, risk, _, _ = env.step(a_np)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            loss = nn.functional.mse_loss(risk_pred, risk_t) + 0.1 * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def run_g1_scaling_law(n_seeds=8, k=2, n_episodes=3, episode_length=2000):
    d_list = [16, 32, 64]
    results = {"baseline": {}, "split_encoder": {}, "guided_attention": {}, "comparison": {}}

    for d in d_list:
        hd = d * 12

        base_aligns = []; base_r2s = []; base_k80s = []
        base_costs = []; base_zones = []

        split_aligns = []; split_r2s = []; split_k80s = []
        split_costs = []; split_zones = []

        guided_aligns = []; guided_r2s = []; guided_k80s = []
        guided_costs = []; guided_zones = []

        for seed in range(n_seeds):
            env_b = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model_b = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
            _train_vanilla(model_b, env_b, n_episodes, episode_length, seed)
            m = _measure(model_b, env_b, d, k)
            base_aligns.append(m["alignment_gt"]); base_r2s.append(m["R2"]); base_k80s.append(m["k80"])
            bv = _behavioral(model_b, env_b, d, k)
            base_costs.append(bv["mean_cost"]); base_zones.append(bv["in_zone_rate"])

            env_s = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model_s = SplitV4Model(state_dim=d, hidden_dim=hd, action_dim=k, k=k)
            _train_vanilla(model_s, env_s, n_episodes, episode_length, seed)
            m = _measure(model_s, env_s, d, k)
            split_aligns.append(m["alignment_gt"]); split_r2s.append(m["R2"]); split_k80s.append(m["k80"])
            bv = _behavioral(model_s, env_s, d, k)
            split_costs.append(bv["mean_cost"]); split_zones.append(bv["in_zone_rate"])

            env_g = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model_g = GuidedAttentionModel(state_dim=d, hidden_dim=hd, action_dim=k, k=k)
            _train_vanilla(model_g, env_g, n_episodes, episode_length, seed)
            m = _measure(model_g, env_g, d, k)
            guided_aligns.append(m["alignment_gt"]); guided_r2s.append(m["R2"]); guided_k80s.append(m["k80"])
            bv = _behavioral(model_g, env_g, d, k)
            guided_costs.append(bv["mean_cost"]); guided_zones.append(bv["in_zone_rate"])

        for name, aligns, r2s, k80s, costs, zones in [
            ("baseline", base_aligns, base_r2s, base_k80s, base_costs, base_zones),
            ("split_encoder", split_aligns, split_r2s, split_k80s, split_costs, split_zones),
            ("guided_attention", guided_aligns, guided_r2s, guided_k80s, guided_costs, guided_zones),
        ]:
            results[name][str(d)] = {
                "d": d, "hd": hd,
                "align_mean": float(np.mean(aligns)), "align_std": float(np.std(aligns)),
                "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
                "k80_mean": float(np.mean(k80s)), "k80_std": float(np.std(k80s)),
                "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
                "zone_rate_mean": float(np.mean(zones)), "zone_rate_std": float(np.std(zones)),
            }

    analysis = _analyze(results, d_list)
    results["analysis"] = analysis
    return results


def _analyze(results, d_list):
    archs = ["baseline", "split_encoder", "guided_attention"]
    failure_info = {}
    for arch in archs:
        aligns = [results[arch][str(d)]["align_mean"] for d in d_list]
        failure_d = None
        for i, d in enumerate(d_list):
            if aligns[i] < 0.7:
                failure_d = d
                break
        failure_info[arch] = {"failure_d": failure_d, "curve": aligns}

    conclusion = []
    for arch, info in failure_info.items():
        fd = info["failure_d"]
        if fd is None:
            conclusion.append(f"{arch}: no failure up to d={d_list[-1]} (align_min={min(info['curve']):.3f})")
        else:
            conclusion.append(f"{arch}: failure at d={fd}")

    return {"failures": failure_info, "conclusion": "; ".join(conclusion)}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_g1_scaling_law(n_seeds=8)
    with open("core_mvp_v4/results/g1_scaling_law.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("G1:", r["analysis"]["conclusion"])
