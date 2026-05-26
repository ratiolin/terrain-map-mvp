"""Experiment C: Forced Semantics — Interventions to preserve alignment in high dims.

C1: Auxiliary loss sweep — lambda * MSE(linear(h), C(s)) with varying lambda.
C2: Split encoder — separate branches for control and noise dims.
C3: Attention gating — learned attention over input dimensions.

All run at d=8, k=2. Compare against baseline (no intervention).
"""

import json
import numpy as np
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import (
    V4Model, SplitV4Model, AttentionV4Model,
    train_with_signal_loss, collect_controllability_data, compute_jacobian,
)
from core_mvp_v4.metrics import compute_k80, alignment


def _measure(model, env, d, k):
    test_h, test_C = collect_controllability_data(model, env, n_samples=300)
    probe = LinearRegression().fit(test_h, test_C)
    R2 = float(probe.score(test_h, test_C))

    states = []
    env.reset()
    for _ in range(100):
        states.append(env.get_state())
        a = model.act_numpy(env.get_state())
        env.step(a)
    J_mean = np.mean([compute_jacobian(model, s) for s in states[::3]], axis=0)
    _, S_mean, Vt_mean = np.linalg.svd(J_mean, full_matrices=False)
    V = Vt_mean.T
    k80 = compute_k80(S_mean)
    U_true = np.eye(d)[:, :k]
    k_use = min(k, V.shape[1])
    align_gt = float(alignment(V[:, :k_use], U_true[:, :k_use], k=k_use))
    return {"R2": R2, "alignment_gt": align_gt, "k80": k80}


def run_c1_aux_loss_sweep(n_seeds=8, d=8, k=2, hd=32,
                          n_episodes=3, episode_length=2000):
    lambdas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = {}

    for lam in lambdas:
        r2_vals = []; align_vals = []; k80_vals = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
            train_with_signal_loss(model, env, num_episodes=n_episodes,
                                   episode_length=episode_length,
                                   lambda_signal=lam, seed=seed)
            m = _measure(model, env, d, k)
            r2_vals.append(m["R2"]); align_vals.append(m["alignment_gt"]); k80_vals.append(m["k80"])
        results[f"lambda_{lam}"] = {
            "R2_mean": float(np.mean(r2_vals)), "R2_std": float(np.std(r2_vals)),
            "align_mean": float(np.mean(align_vals)), "align_std": float(np.std(align_vals)),
            "k80_mean": float(np.mean(k80_vals)), "k80_std": float(np.std(k80_vals)),
        }
    baseline = results["lambda_0.0"]
    best_key = max(results.keys(), key=lambda x: results[x]["align_mean"])
    results["diagnosis"] = {
        "best_lambda": best_key,
        "best_align": results[best_key]["align_mean"],
        "baseline_align": baseline["align_mean"],
        "improvement": results[best_key]["align_mean"] - baseline["align_mean"],
    }
    return results


def train_split_model(model, env, num_episodes, episode_length,
                      lambda_signal, seed):
    import torch
    import torch.nn as nn
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    signal_probe = nn.Linear(model.hidden_dim, 1)
    params = list(model.parameters()) + list(signal_probe.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    for ep in range(num_episodes):
        state = env.reset()
        for step in range(episode_length):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()

            sa = env.forward_static(state, a_np)
            sz = env.forward_static(state, np.zeros(env.action_dim))
            C_actual = float(np.linalg.norm(sa[:env.k] - sz[:env.k]))

            ns, risk, _, _ = env.step(a_np)

            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            C_t = torch.tensor([[C_actual]], dtype=torch.float32)
            C_pred = signal_probe(h)

            loss = (nn.functional.mse_loss(risk_pred, risk_t) +
                    0.1 * torch.mean(action ** 2) +
                    lambda_signal * nn.functional.mse_loss(C_pred, C_t))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(signal_probe.parameters(), 1.0)
            optimizer.step()
            state = ns


def run_c2_split_encoder(n_seeds=8, d=8, k=2, hd=32,
                         n_episodes=3, episode_length=2000, lambda_signal=1.0):
    r2_vals = []; align_vals = []; k80_vals = []
    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = SplitV4Model(state_dim=d, hidden_dim=hd, action_dim=k, k=k)
        train_split_model(model, env, n_episodes, episode_length, lambda_signal, seed)
        m = _measure(model, env, d, k)
        r2_vals.append(m["R2"]); align_vals.append(m["alignment_gt"]); k80_vals.append(m["k80"])
    return {
        "R2_mean": float(np.mean(r2_vals)), "R2_std": float(np.std(r2_vals)),
        "align_mean": float(np.mean(align_vals)), "align_std": float(np.std(align_vals)),
        "k80_mean": float(np.mean(k80_vals)), "k80_std": float(np.std(k80_vals)),
    }


def train_attention_model(model, env, num_episodes, episode_length,
                          lambda_signal, seed):
    import torch
    import torch.nn as nn
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    signal_probe = nn.Linear(model.hidden_dim, 1)
    params = list(model.parameters()) + list(signal_probe.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    for ep in range(num_episodes):
        state = env.reset()
        for step in range(episode_length):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred, attn = model(s_t)
            a_np = action.squeeze(0).detach().numpy()

            sa = env.forward_static(state, a_np)
            sz = env.forward_static(state, np.zeros(env.action_dim))
            C_actual = float(np.linalg.norm(sa[:env.k] - sz[:env.k]))

            ns, risk, _, _ = env.step(a_np)

            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            C_t = torch.tensor([[C_actual]], dtype=torch.float32)
            C_pred = signal_probe(h)

            loss = (nn.functional.mse_loss(risk_pred, risk_t) +
                    0.1 * torch.mean(action ** 2) +
                    lambda_signal * nn.functional.mse_loss(C_pred, C_t))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(signal_probe.parameters(), 1.0)
            optimizer.step()
            state = ns


def run_c3_attention(n_seeds=8, d=8, k=2, hd=32,
                     n_episodes=3, episode_length=2000, lambda_signal=1.0):
    r2_vals = []; align_vals = []; k80_vals = []
    attn_weights = []
    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = AttentionV4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        train_attention_model(model, env, n_episodes, episode_length, lambda_signal, seed)
        m = _measure(model, env, d, k)
        r2_vals.append(m["R2"]); align_vals.append(m["alignment_gt"]); k80_vals.append(m["k80"])

        env.reset()
        attn_sum = np.zeros(d)
        for _ in range(100):
            s = env.get_state()
            attn_sum += model.attn_numpy(s)
            a = model.act_numpy(s)
            env.step(a)
        attn_weights.append((attn_sum / 100).tolist())

    attn_ctrl_mean = float(np.mean([np.mean(w[:k]) for w in attn_weights]))
    attn_noise_mean = float(np.mean([np.mean(w[k:]) for w in attn_weights]))
    return {
        "R2_mean": float(np.mean(r2_vals)), "R2_std": float(np.std(r2_vals)),
        "align_mean": float(np.mean(align_vals)), "align_std": float(np.std(align_vals)),
        "k80_mean": float(np.mean(k80_vals)), "k80_std": float(np.std(k80_vals)),
        "attention_ctrl_mean": attn_ctrl_mean,
        "attention_noise_mean": attn_noise_mean,
        "attention_bias": attn_ctrl_mean - attn_noise_mean,
    }


def run_exp_c_forced_semantics(n_seeds=8, d=8, k=2, hd=32,
                               n_episodes=3, episode_length=2000, lambda_signal=1.0):
    results = {}

    print("  C1: Auxiliary loss sweep...")
    results["C1_aux_loss"] = run_c1_aux_loss_sweep(
        n_seeds=n_seeds, d=d, k=k, hd=hd,
        n_episodes=n_episodes, episode_length=episode_length,
    )

    print("  C2: Split encoder...")
    results["C2_split_encoder"] = run_c2_split_encoder(
        n_seeds=n_seeds, d=d, k=k, hd=hd,
        n_episodes=n_episodes, episode_length=episode_length,
        lambda_signal=lambda_signal,
    )

    print("  C3: Attention gating...")
    results["C3_attention"] = run_c3_attention(
        n_seeds=n_seeds, d=d, k=k, hd=hd,
        n_episodes=n_episodes, episode_length=episode_length,
        lambda_signal=lambda_signal,
    )

    results["comparison"] = _compare(results)
    return results


def _compare(results):
    c1_diag = results["C1_aux_loss"]["diagnosis"]
    c2 = results["C2_split_encoder"]
    c3 = results["C3_attention"]

    best = max([
        ("C1_aux_loss", c1_diag["best_align"]),
        ("C2_split_encoder", c2["align_mean"]),
        ("C3_attention", c3["align_mean"]),
    ], key=lambda x: x[1])

    return {
        "best_method": best[0],
        "best_alignment": best[1],
        "C1_improvement": c1_diag["improvement"],
        "C2_alignment": c2["align_mean"],
        "C3_alignment": c3["align_mean"],
        "C3_attention_bias": c3["attention_bias"],
    }


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_exp_c_forced_semantics(n_seeds=8)
    with open("core_mvp_v4/results/c_forced_semantics.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    comp = r["comparison"]
    print(f"Experiment C: best_method={comp['best_method']}, best_align={comp['best_alignment']:.4f}")
