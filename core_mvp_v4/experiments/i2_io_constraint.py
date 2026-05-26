"""I2: Input-Output Direct Constraints.

Measures input sensitivity, Lipschitz ||∂a/∂s||, output PCA.
Compares trained vs random model.
"""

import json, copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model


def _train(model, env, n_ep, ep_len, seed):
    if seed is not None: torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            ns, risk, _, _ = env.step(action.squeeze(0).detach().numpy())
            loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def _input_sensitivity(model, env, d, k, sigma, n_pts=100):
    env.reset()
    act_changes = []; cost_changes = []
    for _ in range(n_pts):
        s = env.get_state()
        a_orig = model.act_numpy(s)
        _, r_orig, _, _ = env.step(a_orig)
        s_noisy = s + np.random.randn(d) * sigma
        a_noisy = model.act_numpy(s_noisy)
        ns, r_noisy, _, _ = env.step(a_noisy)
        act_changes.append(float(np.linalg.norm(a_noisy - a_orig)))
        cost_changes.append(abs(r_noisy - r_orig))
    return float(np.mean(act_changes)), float(np.mean(cost_changes))


def _estimate_lipschitz_s(model, s_list, eps=0.01, k=2):
    ratios = []
    for s in s_list:
        s1 = s + np.random.randn(len(s)) * eps
        a1 = model.act_numpy(s)
        a2 = model.act_numpy(s1)
        ds = np.linalg.norm(s1 - s)
        if ds > 1e-8: ratios.append(np.linalg.norm(a2 - a1) / ds)
    return float(np.mean(ratios))


def run_i2_io_constraint(n_seeds=4, d=16, k=2, hd=192,
                          n_episodes=3, episode_length=2000):
    results = {"trained": {}, "random": {}}

    for model_type in ["trained", "random"]:
        act_sens = []; cost_sens = []; lip_vals = []; act_dims = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
            if model_type == "trained":
                _train(model, env, n_episodes, episode_length, seed)
            env_ref = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=0, coupling=0.0)
            env_ref.reset()

            for sigma in [0.01, 0.05, 0.1]:
                ac, cc = _input_sensitivity(model, env_ref, d, k, sigma)
                act_sens.append(ac); cost_sens.append(cc)

            test_s = []
            for _ in range(50):
                test_s.append(env_ref.get_state())
                env_ref.step(np.zeros(k))
            lip_vals.append(_estimate_lipschitz_s(model, test_s))

            actions = [model.act_numpy(s) for s in test_s]
            pca = PCA().fit(np.array(actions))
            act_dims.append(int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)) + 1)

        results[model_type] = {
            "action_sensitivity_mean": float(np.mean(act_sens)),
            "cost_sensitivity_mean": float(np.mean(cost_sens)),
            "lipschitz_mean": float(np.mean(lip_vals)),
            "output_pca_dim": float(np.mean(act_dims)),
        }

    tr_lip = results["trained"]["lipschitz_mean"]
    rd_lip = results["random"]["lipschitz_mean"]
    out_dim = results["trained"]["output_pca_dim"]

    if tr_lip < rd_lip * 0.5:
        conclusion = f"TRAINING SMOOTHENS: Lip trained={tr_lip:.4f}, random={rd_lip:.4f}. Output dim={out_dim:.0f}."
    elif abs(tr_lip - rd_lip) / (rd_lip + 1e-6) < 0.2:
        conclusion = f"ARCHITECTURE-INHERENT: Lip trained={tr_lip:.4f} ≈ random={rd_lip:.4f}. Smoothness from architecture."
    else:
        conclusion = f"MIXED: trained Lip={tr_lip:.4f}, random={rd_lip:.4f}, output dim={out_dim:.0f}."
    results["analysis"] = {"conclusion": conclusion}
    return results


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_i2_io_constraint(n_seeds=4)
    with open("core_mvp_v4/results/i2_io_constraint.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("I2:", r["analysis"]["conclusion"])
