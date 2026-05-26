"""G2: Minimal Attribution Bias Search.

Fixed d=16, k=2, hd=192 (ratio 12). Three methods:

A: Learnable input scaling weights (s *= w, w init=1.0)
B: Learnable scaling + L1 regularization on weights
C: Input augmentation with control-dim hint (concat one-hot indicator)

Goal: find minimal structural injection that restores alignment_gt > 0.8.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import (
    V4Model, ScaledInputModel,
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


def train_scaled(model, env, n_ep, ep_len, seed, l1_lambda=0.0):
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
            loss = (nn.functional.mse_loss(risk_pred, risk_t) +
                    0.1 * torch.mean(action**2))
            if l1_lambda > 0 and hasattr(model, 'input_weights'):
                loss += l1_lambda * model.input_weights.abs().sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def _baseline(seed, d, k, hd, n_ep, ep_len):
    env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
    model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
    train_scaled(model, env, n_ep, ep_len, seed)
    m = _measure(model, env, d, k)
    return m


def run_g2_minimal_bias(n_seeds=8, d=16, k=2, hd=192,
                        n_episodes=3, episode_length=2000):
    results = {}

    # Baseline
    base_aligns = []; base_r2s = []; base_k80s = []
    for seed in range(n_seeds):
        m = _baseline(seed, d, k, hd, n_episodes, episode_length)
        base_aligns.append(m["alignment_gt"]); base_r2s.append(m["R2"]); base_k80s.append(m["k80"])
    results["baseline"] = {
        "align_mean": float(np.mean(base_aligns)), "align_std": float(np.std(base_aligns)),
        "R2_mean": float(np.mean(base_r2s)), "R2_std": float(np.std(base_r2s)),
        "k80_mean": float(np.mean(base_k80s)), "k80_std": float(np.std(base_k80s)),
        "extra_params": 0,
    }

    # A: Learnable input scaling
    a_aligns = []; a_r2s = []; a_k80s = []; a_weights = []
    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = ScaledInputModel(state_dim=d, hidden_dim=hd, action_dim=k)
        train_scaled(model, env, n_episodes, episode_length, seed, l1_lambda=0.0)
        m = _measure(model, env, d, k)
        a_aligns.append(m["alignment_gt"]); a_r2s.append(m["R2"]); a_k80s.append(m["k80"])
        a_weights.append(model.weight_numpy())
    results["A_scaling"] = {
        "align_mean": float(np.mean(a_aligns)), "align_std": float(np.std(a_aligns)),
        "R2_mean": float(np.mean(a_r2s)), "R2_std": float(np.std(a_r2s)),
        "k80_mean": float(np.mean(a_k80s)), "k80_std": float(np.std(a_k80s)),
        "extra_params": d,
        "weight_ctrl_mean": float(np.mean([np.mean(w[:k]) for w in a_weights])),
        "weight_noise_mean": float(np.mean([np.mean(w[k:]) for w in a_weights])),
        "weight_ratio": float(np.mean([np.mean(w[:k]) / (np.mean(np.abs(w[k:])) + 1e-6) for w in a_weights])),
    }

    # B: Learnable scaling + L1
    b_aligns = []; b_r2s = []; b_k80s = []; b_weights = []
    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = ScaledInputModel(state_dim=d, hidden_dim=hd, action_dim=k)
        train_scaled(model, env, n_episodes, episode_length, seed, l1_lambda=0.01)
        m = _measure(model, env, d, k)
        b_aligns.append(m["alignment_gt"]); b_r2s.append(m["R2"]); b_k80s.append(m["k80"])
        b_weights.append(model.weight_numpy())
    results["B_sparse_scaling"] = {
        "align_mean": float(np.mean(b_aligns)), "align_std": float(np.std(b_aligns)),
        "R2_mean": float(np.mean(b_r2s)), "R2_std": float(np.std(b_r2s)),
        "k80_mean": float(np.mean(b_k80s)), "k80_std": float(np.std(b_k80s)),
        "extra_params": d,
        "weight_ctrl_mean": float(np.mean([np.mean(w[:k]) for w in b_weights])),
        "weight_noise_mean": float(np.mean([np.mean(w[k:]) for w in b_weights])),
        "weight_ratio": float(np.mean([np.mean(w[:k]) / (np.mean(np.abs(w[k:])) + 1e-6) for w in b_weights])),
    }

    # C: Input hint vector
    c_aligns = []; c_r2s = []; c_k80s = []
    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d+k, hidden_dim=hd, action_dim=k)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        hint = np.zeros(k)
        for i in range(k):
            hint[i] = 1.0
        for _ in range(n_episodes):
            env.reset()
            for __ in range(episode_length):
                s_raw = env.get_state()
                s_aug = np.concatenate([s_raw, hint])
                s_t = torch.from_numpy(s_aug.astype(np.float32)).unsqueeze(0)
                action, h, risk_pred = model(s_t)
                a_np = action.squeeze(0).detach().numpy()
                ns, risk, _, _ = env.step(a_np)
                risk_t = torch.tensor([[risk]], dtype=torch.float32)
                loss = (nn.functional.mse_loss(risk_pred, risk_t) + 0.1 * torch.mean(action**2))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        env.reset()

        class AugmentedModel:
            def __init__(self, inner, hint_vec):
                self.inner = inner
                self.hint_vec = hint_vec
            def act_numpy(self, s):
                s_aug = np.concatenate([s, self.hint_vec])
                return self.inner.act_numpy(s_aug)
            def f_numpy(self, s):
                s_aug = np.concatenate([s, self.hint_vec])
                return self.inner.f_numpy(s_aug)
        model_wrapper = AugmentedModel(model, hint)

        m = _measure(model_wrapper, env, d, k)
        c_aligns.append(m["alignment_gt"]); c_r2s.append(m["R2"]); c_k80s.append(m["k80"])
    results["C_hint"] = {
        "align_mean": float(np.mean(c_aligns)), "align_std": float(np.std(c_aligns)),
        "R2_mean": float(np.mean(c_r2s)), "R2_std": float(np.std(c_r2s)),
        "k80_mean": float(np.mean(c_k80s)), "k80_std": float(np.std(c_k80s)),
        "extra_params": k,
    }

    results["comparison"] = _compare(results)
    return results


def _compare(results):
    baseline_align = results["baseline"]["align_mean"]
    methods = [
        ("A_scaling", results["A_scaling"]["align_mean"], results["A_scaling"]["extra_params"]),
        ("B_sparse_scaling", results["B_sparse_scaling"]["align_mean"], results["B_sparse_scaling"]["extra_params"]),
        ("C_hint", results["C_hint"]["align_mean"], results["C_hint"]["extra_params"]),
    ]

    best_name = None
    best_score = -np.inf
    for name, align, params in methods:
        if align > 0.8 and params < 10:
            score = align / (params + 1)
            if score > best_score:
                best_score = score
                best_name = name

    info = {}
    for name, align, params in methods:
        info[name] = {"align": align, "params": params, "improvement": align - baseline_align}

    if best_name:
        conclusion = f"MINIMAL: {best_name} achieves align={info[best_name]['align']:.3f} with only {info[best_name]['params']} extra params."
    else:
        conclusion = f"NO METHOD achieves align>0.8 with <10 extra params. Best align among methods: {max(m[1] for m in methods):.3f}."

    return {"methods": info, "best": best_name, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_g2_minimal_bias(n_seeds=8)
    with open("core_mvp_v4/results/g2_minimal_bias.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("G2:", r["comparison"]["conclusion"])
