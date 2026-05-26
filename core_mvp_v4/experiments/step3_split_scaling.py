"""Step 3: Split Encoder Generalization.

Uses C2 split-encoder architecture (control/noise separate branches, then fuse).
Fixed hidden_dim=64. Tests d ∈ [4, 8, 16, 32].
Compares against shared-encoder baseline at each d.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import (
    V4Model, SplitV4Model,
    train_with_signal_loss, collect_controllability_data, compute_jacobian,
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


def train_split_model(model, env, n_episodes, ep_length, lambda_signal, seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    probe = nn.Linear(model.hidden_dim, 1)
    params = list(model.parameters()) + list(probe.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    for _ in range(n_episodes):
        env.reset()
        for __ in range(ep_length):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()
            sa = env.forward_static(env.get_state(), a_np)
            sz = env.forward_static(env.get_state(), np.zeros(env.action_dim))
            C_actual = float(np.linalg.norm(sa[:env.k] - sz[:env.k]))
            ns, risk, _, _ = env.step(a_np)
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            C_t = torch.tensor([[C_actual]], dtype=torch.float32)
            C_pred = probe(h)
            loss = (nn.functional.mse_loss(risk_pred, risk_t) +
                    0.1 * torch.mean(action ** 2) +
                    lambda_signal * nn.functional.mse_loss(C_pred, C_t))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            env.get_state()


def run_step3_split_scaling(n_seeds=8, k=2, hd=64,
                            n_episodes=3, episode_length=2000, lambda_signal=1.0):
    d_list = [4, 8, 16, 32]
    results = {"split": {}, "baseline": {}, "comparison": {}}

    for d in d_list:
        split_aligns = []; split_r2s = []; split_k80s = []
        base_aligns = []; base_r2s = []; base_k80s = []

        for seed in range(n_seeds):
            split_env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            split_model = SplitV4Model(state_dim=d, hidden_dim=hd, action_dim=k, k=k)
            train_split_model(split_model, split_env, n_episodes, episode_length, lambda_signal, seed)
            m = _measure(split_model, split_env, d, k)
            split_aligns.append(m["alignment_gt"]); split_r2s.append(m["R2"]); split_k80s.append(m["k80"])

            base_env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            base_model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
            train_with_signal_loss(base_model, base_env, num_episodes=n_episodes,
                                   episode_length=episode_length,
                                   lambda_signal=lambda_signal, seed=seed, alpha=1.0)
            m = _measure(base_model, base_env, d, k)
            base_aligns.append(m["alignment_gt"]); base_r2s.append(m["R2"]); base_k80s.append(m["k80"])

        results["split"][str(d)] = {
            "align_mean": float(np.mean(split_aligns)), "align_std": float(np.std(split_aligns)),
            "R2_mean": float(np.mean(split_r2s)), "R2_std": float(np.std(split_r2s)),
            "k80_mean": float(np.mean(split_k80s)), "k80_std": float(np.std(split_k80s)),
        }
        results["baseline"][str(d)] = {
            "align_mean": float(np.mean(base_aligns)), "align_std": float(np.std(base_aligns)),
            "R2_mean": float(np.mean(base_r2s)), "R2_std": float(np.std(base_r2s)),
            "k80_mean": float(np.mean(base_k80s)), "k80_std": float(np.std(base_k80s)),
        }

    for d in d_list:
        s = str(d)
        results["comparison"][s] = {
            "d": d,
            "split_align": results["split"][s]["align_mean"],
            "baseline_align": results["baseline"][s]["align_mean"],
            "improvement": results["split"][s]["align_mean"] - results["baseline"][s]["align_mean"],
        }

    results["analysis"] = _analyze_split(results, d_list)
    return results


def _analyze_split(results, d_list):
    split_aligns = [results["comparison"][str(d)]["split_align"] for d in d_list]
    base_aligns = [results["comparison"][str(d)]["baseline_align"] for d in d_list]

    failure_d = None
    for i, d in enumerate(d_list):
        if split_aligns[i] < 0.6:
            failure_d = d
            break

    if failure_d is None:
        conclusion = f"Split encoder maintains alignment >= 0.6 across all d up to {d_list[-1]}. No failure found."
    else:
        conclusion = f"Split encoder fails at d={failure_d} (alignment={next(s for d,s in zip(d_list,split_aligns) if d==failure_d):.3f}<0.6)"

    return {
        "split_curve": split_aligns, "baseline_curve": base_aligns,
        "failure_d": failure_d, "conclusion": conclusion,
    }


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_step3_split_scaling(n_seeds=8)
    with open("core_mvp_v4/results/split_encoder_scaling.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    an = r["analysis"]
    print(f"Step 3: failure_d={an['failure_d']}")
    print(f"  {an['conclusion']}")
