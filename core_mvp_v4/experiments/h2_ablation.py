"""H2: Representation Redundancy — Ablation Validation.

Tests whether the geometric controllability subspace is causally necessary for behavior.
If ablation doesn't change behavior, the subspace is epiphenomenal; backup representations exist.

Three ablation modes: hard (project h to null space), soft (directional noise), dim_drop (zero components).
Control: random subspace ablation.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_rel

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, SplitV4Model, compute_jacobian
from core_mvp_v4.metrics import compute_k80


def _train(model, env, n_ep, ep_len, seed):
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


def _behavioral(model, env, d, k, ablation_fn=None, n_steps=500):
    env.reset()
    cost_sum = 0.0
    in_zone = 0
    success = 0
    for _ in range(n_steps):
        s = env.get_state()
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h_raw = model.encoder(s_t) if hasattr(model, 'encoder') else model._compute_hidden(s_t)
        h_raw_np = h_raw.squeeze(0).detach().numpy()

        if ablation_fn is not None:
            h_ablated = ablation_fn(h_raw_np)
            h_t = torch.from_numpy(h_ablated.astype(np.float32)).unsqueeze(0)
            a = model.actor(h_t).squeeze(0).detach().numpy()
        else:
            a = model.act_numpy(s)

        ns, risk, _, _ = env.step(a)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
        if risk < 0.5:
            success += 1
    return {
        "mean_cost": cost_sum / n_steps,
        "in_zone_rate": in_zone / n_steps,
        "success_rate": success / n_steps,
    }


def _compute_subspace(model, env, d, k):
    env.reset()
    Js = []
    for _ in range(50):
        s = env.get_state()
        J = compute_jacobian(model, s)
        Js.append(J)
        a = model.act_numpy(s)
        env.step(a)
    J_mean = np.mean(Js, axis=0)
    U, S, Vt = np.linalg.svd(J_mean, full_matrices=False)
    k80 = compute_k80(S)
    U_ctrl = U[:, :k80].copy()
    return U_ctrl, k80


def _make_ablation_fns(U_ctrl):
    k = U_ctrl.shape[1]
    hd = U_ctrl.shape[0]

    def hard_ablation(h):
        P_null = np.eye(hd) - U_ctrl @ U_ctrl.T
        return P_null @ h

    def soft_ablation_eps(h, eps):
        coeffs = np.random.randn(k) * eps
        noise = U_ctrl @ coeffs
        return h + noise

    def dim_drop(h):
        comp = U_ctrl.T @ h
        h_drop = h - U_ctrl @ comp
        return h_drop

    return hard_ablation, soft_ablation_eps, dim_drop


def run_h2_ablation(n_seeds=8, d=16, k=2, hd=192, n_episodes=3, episode_length=2000):
    eps_list = [0.1, 0.5, 1.0]
    results = {"baseline": [], "split_encoder": []}

    for arch, ModelClass in [("baseline", V4Model), ("split_encoder", SplitV4Model)]:
        arch_results = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            if arch == "split_encoder":
                model = ModelClass(state_dim=d, hidden_dim=hd, action_dim=k, k=k)
            else:
                model = ModelClass(state_dim=d, hidden_dim=hd, action_dim=k)
            _train(model, env, n_episodes, episode_length, seed)

            U_ctrl, k80_val = _compute_subspace(model, env, d, k)
            hard_fn, soft_fn, dim_fn = _make_ablation_fns(U_ctrl)

            U_rand = np.random.randn(*U_ctrl.shape)
            U_rand, _ = np.linalg.qr(U_rand)
            hard_rand_fn, _, _ = _make_ablation_fns(U_rand)

            env2 = env.clone()
            no_ab = _behavioral(model, env2, d, k, n_steps=500)
            env2.reset()
            hard_ab = _behavioral(model, env2, d, k, ablation_fn=hard_fn, n_steps=500)
            env2.reset()
            hard_rand = _behavioral(model, env2, d, k, ablation_fn=hard_rand_fn, n_steps=500)
            env2.reset()
            dim_ab = _behavioral(model, env2, d, k, ablation_fn=dim_fn, n_steps=500)

            soft_abs = {}
            for eps in eps_list:
                env2.reset()
                sf = lambda h, e=eps: soft_fn(h, e)
                soft_abs[f"eps_{eps}"] = _behavioral(model, env2, d, k, ablation_fn=sf, n_steps=500)

            arch_results.append({
                "seed": seed, "k80": k80_val,
                "no_ablation": no_ab,
                "hard_ablation": hard_ab,
                "hard_random": hard_rand,
                "dim_drop": dim_ab,
                "soft_ablation": soft_abs,
            })
        results[arch] = arch_results
        print(f"  {arch}: {n_seeds} seeds done")

    results["analysis"] = _analyze(results, eps_list)
    return results


def _analyze(results, eps_list):
    analysis = {}
    for arch in ["baseline", "split_encoder"]:
        arch_data = results[arch]
        no_costs = [r["no_ablation"]["mean_cost"] for r in arch_data]
        hard_costs = [r["hard_ablation"]["mean_cost"] for r in arch_data]
        rand_costs = [r["hard_random"]["mean_cost"] for r in arch_data]
        dim_costs = [r["dim_drop"]["mean_cost"] for r in arch_data]

        t_hard, p_hard = ttest_rel(hard_costs, no_costs)
        t_rand, p_rand = ttest_rel(rand_costs, no_costs)

        hard_effect = float(np.mean(hard_costs) - np.mean(no_costs))
        rand_effect = float(np.mean(rand_costs) - np.mean(no_costs))

        analysis[arch] = {
            "hard": {"p_value": float(p_hard), "significant": p_hard < 0.01,
                     "effect_size": hard_effect},
            "random": {"p_value": float(p_rand), "significant": p_rand < 0.01,
                       "effect_size": rand_effect},
        }

    b_hard_sig = analysis["baseline"]["hard"]["significant"]
    s_hard_sig = analysis["split_encoder"]["hard"]["significant"]

    if (not b_hard_sig) and (not s_hard_sig):
        conclusion = "REDUNDANCY CONFIRMED: ablation has no significant effect. Subspace is epiphenomenal."
    elif s_hard_sig and not b_hard_sig:
        conclusion = "SPLIT-SENSITIVE: only split encoder shows causal dependency. Attribution bias enhances causal engagement."
    elif b_hard_sig and s_hard_sig:
        conclusion = "CAUSAL: subspace is causally necessary in both architectures."
    else:
        conclusion = "MIXED: baseline shows causality but split doesn't."

    analysis["conclusion"] = conclusion
    return analysis


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_h2_ablation(n_seeds=8)
    with open("core_mvp_v4/results/h2_ablation.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("H2:", r["analysis"]["conclusion"])
