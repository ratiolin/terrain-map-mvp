"""Step 4: Attention Guided — enhanced attention with position encoding.

Adds learnable position embeddings + binary indicator (1 for control dims).
d=8, hd=64, k=2. Compares against baseline C3 attention model.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import (
    AttentionV4Model, GuidedAttentionModel,
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


def train_attn_model(model, env, n_episodes, ep_length, lambda_signal, seed):
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
            outputs = model(s_t)
            action, h, risk_pred = outputs[0], outputs[1], outputs[2]
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


def run_step4_attention_guided(n_seeds=8, d=8, k=2, hd=64,
                               n_episodes=3, episode_length=2000, lambda_signal=1.0):
    results = {"baseline_C3": {}, "guided": {}, "comparison": {}}

    base_aligns = []; base_r2s = []; base_k80s = []
    base_attn_ctrl = []; base_attn_noise = []

    guided_aligns = []; guided_r2s = []; guided_k80s = []
    guided_attn_ctrl = []; guided_attn_noise = []

    for seed in range(n_seeds):
        env_b = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model_b = AttentionV4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        train_attn_model(model_b, env_b, n_episodes, episode_length, lambda_signal, seed)
        m = _measure(model_b, env_b, d, k)
        base_aligns.append(m["alignment_gt"]); base_r2s.append(m["R2"]); base_k80s.append(m["k80"])

        env_b.reset()
        attn_sum = np.zeros(d)
        for _ in range(100):
            s = env_b.get_state()
            attn = model_b.attn_numpy(s)
            attn_sum += attn
            a = model_b.act_numpy(s)
            env_b.step(a)
        attn_avg = attn_sum / 100
        base_attn_ctrl.append(float(np.mean(attn_avg[:k])))
        base_attn_noise.append(float(np.mean(attn_avg[k:])))

        env_g = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model_g = GuidedAttentionModel(state_dim=d, hidden_dim=hd, action_dim=k, k=k)
        train_attn_model(model_g, env_g, n_episodes, episode_length, lambda_signal, seed)
        m = _measure(model_g, env_g, d, k)
        guided_aligns.append(m["alignment_gt"]); guided_r2s.append(m["R2"]); guided_k80s.append(m["k80"])

        env_g.reset()
        attn_sum = np.zeros(d)
        for _ in range(100):
            s = env_g.get_state()
            attn = model_g.attn_numpy(s)
            attn_sum += attn
            a = model_g.act_numpy(s)
            env_g.step(a)
        attn_avg = attn_sum / 100
        guided_attn_ctrl.append(float(np.mean(attn_avg[:k])))
        guided_attn_noise.append(float(np.mean(attn_avg[k:])))

    results["baseline_C3"] = {
        "align_mean": float(np.mean(base_aligns)), "align_std": float(np.std(base_aligns)),
        "R2_mean": float(np.mean(base_r2s)), "R2_std": float(np.std(base_r2s)),
        "k80_mean": float(np.mean(base_k80s)), "k80_std": float(np.std(base_k80s)),
        "attn_ctrl_mean": float(np.mean(base_attn_ctrl)),
        "attn_noise_mean": float(np.mean(base_attn_noise)),
        "attn_bias": float(np.mean(base_attn_ctrl) - np.mean(base_attn_noise)),
    }
    results["guided"] = {
        "align_mean": float(np.mean(guided_aligns)), "align_std": float(np.std(guided_aligns)),
        "R2_mean": float(np.mean(guided_r2s)), "R2_std": float(np.std(guided_r2s)),
        "k80_mean": float(np.mean(guided_k80s)), "k80_std": float(np.std(guided_k80s)),
        "attn_ctrl_mean": float(np.mean(guided_attn_ctrl)),
        "attn_noise_mean": float(np.mean(guided_attn_noise)),
        "attn_bias": float(np.mean(guided_attn_ctrl) - np.mean(guided_attn_noise)),
    }
    results["comparison"] = {
        "guided_improvement": results["guided"]["align_mean"] - results["baseline_C3"]["align_mean"],
        "guided_attn_improvement": results["guided"]["attn_bias"] - results["baseline_C3"]["attn_bias"],
    }

    gb = results["guided"]["attn_bias"]
    if gb > 0.1:
        conclusion = (f"GUIDED SUCCESS: attention bias={gb:.3f} toward control dims. "
                      f"Position encoding + indicator force attention convergence.")
    elif gb > 0.03:
        conclusion = f"MODERATE GUIDANCE: attention bias={gb:.3f}. Some convergence but not strong."
    else:
        conclusion = (f"NO GUIDANCE EFFECT: attention bias={gb:.3f}. "
                      f"Position encoding insufficient to redirect attention.")

    results["conclusion"] = conclusion
    return results


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_step4_attention_guided(n_seeds=8)
    with open("core_mvp_v4/results/attention_guided.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"Step 4: {r['conclusion']}")
