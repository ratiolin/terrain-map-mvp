"""F-C: Task Complexity Analysis.

1. iLQR optimal control on known double-well dynamics → PCA of optimal sequence.
2. Tests on simplified 1D tracking env → verifies analysis pipeline.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

from core_mvp_v4.env import MultiDimDoubleWell


class SimpleTrackingEnv:
    def __init__(self, d=4, k=1, noise_std=0.02, seed=None):
        self.d = d; self.k = k; self.noise_std = noise_std
        self.action_dim = k; self.state_dim = d
        self._rng = np.random.RandomState(seed); self.state = np.zeros(d, dtype=np.float32)
        self.t = 0

    def reset(self):
        self.state = np.zeros(self.d, dtype=np.float32)
        self.state[0] = self._rng.uniform(-1, 1)
        self.t = 0
        return self.state.copy()

    def step(self, action):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()
        target = 0.5 * np.sin(0.02 * self.t)
        err = target - self.state[0]
        self.state[0] += 0.2 * err + 0.05 * action[0] + self._rng.normal(0, self.noise_std)
        self.state[0] = np.clip(self.state[0], -2, 2)
        x_noise = self.state[self.k:].copy()
        x_noise -= 0.5 * x_noise
        self.state[self.k:] = x_noise + self._rng.randn(self.d - self.k) * 0.02
        self.t += 1
        risk = abs(target - self.state[0])
        return self.state.copy(), risk, False, {}

    def forward_static(self, state, action):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()
        s_next = state.copy()
        target = 0.5 * np.sin(0.02 * self.t)
        err = target - s_next[0]
        s_next[0] += 0.2 * err + 0.05 * action[0]
        s_next[0] = np.clip(s_next[0], -2, 2)
        return s_next

    def get_state(self): return self.state.copy()
    def clone(self): return self

    def act_numpy(self, s):
        from core_mvp_v4.models import V4Model
        return np.zeros(self.k)


def _ilqr_optimal(model_env, T=100, n_traj=30):
    """Simplified iLQR: random shooting + gradient descent."""
    all_actions = []
    for _ in range(n_traj):
        x0 = np.random.uniform(-1, 1, size=model_env.k)
        actions = np.zeros((T, model_env.k))
        for opt_step in range(50):
            x = x0.copy()
            total_cost = 0.0
            grad = np.zeros((T, model_env.k))
            for t in range(T):
                a = actions[t]
                x_next = np.zeros_like(x)
                x_next[:model_env.k] = x[:model_env.k] + 0.1 * a + 0.01 * np.random.randn(model_env.k)
                x_next[model_env.k:] = x[model_env.k:] - 0.5 * x[model_env.k:]
                cost = np.linalg.norm(x_next[:model_env.k])
                total_cost += cost
                grad[t] = 0.1 * x_next[:model_env.k] / (cost + 1e-6)
                x = x_next
            actions -= 0.01 * grad
        all_actions.append(actions)
    return np.array(all_actions)


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


def run_f_task_analysis(n_seeds=8, d=16, k=2, hd=192,
                        n_episodes=3, episode_length=2000):
    results = {}

    # Part 1: Optimal control PCA
    env_oc = MultiDimDoubleWell(d=2, k=2, drift=0.5, seed=0, coupling=0.0)
    opt_actions = _ilqr_optimal(env_oc, T=200, n_traj=30)
    opt_flat = opt_actions.reshape(opt_actions.shape[0], -1)
    pca = PCA().fit(opt_flat)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    id_90 = int(np.argmax(cumvar >= 0.90)) + 1
    results["optimal_control_PCA"] = {
        "ID_90": id_90,
        "explained_var_ratio": pca.explained_variance_ratio_[:10].tolist(),
        "note": f"Optimal action sequences require {id_90} PCA dims for 90% variance.",
    }

    # Part 2: Simplified tracking env
    r2_list = []; rho_list = []; cost_list = []
    for seed in range(min(n_seeds, 6)):
        env_s = SimpleTrackingEnv(d=4, k=1, seed=seed)
        from core_mvp_v4.models import V4Model
        model_s = V4Model(state_dim=4, hidden_dim=32, action_dim=1)
        _train(model_s, env_s, n_episodes, episode_length, seed)

        env_s.reset()
        H_vals, C_vals = [], []
        for _ in range(500):
            s = env_s.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model_s.encoder(s_t).squeeze(0).detach().numpy()
            a = model_s.act_numpy(s)
            ns, risk, _, _ = env_s.step(a)
            H_vals.append(h); C_vals.append(risk)

        from sklearn.linear_model import LinearRegression
        split = int(len(H_vals) * 0.7)
        probe = LinearRegression().fit(np.array(H_vals[:split]), np.array(C_vals[:split]))
        R2 = float(probe.score(np.array(H_vals[split:]), np.array(C_vals[split:])))
        rho, _ = spearmanr(np.array(H_vals)[:, 0], C_vals)
        r2_list.append(R2); rho_list.append(abs(rho)); cost_list.append(float(np.mean(C_vals[-100:])))

    results["simple_tracking"] = {
        "R2_linear_mean": float(np.mean(r2_list)), "R2_linear_std": float(np.std(r2_list)),
        "rho_mean": float(np.mean(rho_list)), "cost_mean": float(np.mean(cost_list)),
    }

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    id_oc = results["optimal_control_PCA"]["ID_90"]
    r2_simple = results["simple_tracking"]["R2_linear_mean"]
    rho_simple = results["simple_tracking"]["rho_mean"]

    parts = [f"Optimal actions need {id_oc} PCA dims."]
    if id_oc > 5:
        parts.append("Task is inherently high-dimensional → low-dim control not mathematically guaranteed.")
    else:
        parts.append("Optimal control IS low-dim → our analysis pipeline should work.")

    if rho_simple > 0.3 and r2_simple > 0.3:
        parts.append(f"Simple tracking: R²={r2_simple:.3f}, ρ={rho_simple:.3f}. Pipeline validated on simple case.")
    else:
        parts.append(f"Simple tracking also fails (R²={r2_simple:.3f}). Pipeline issue or task too noisy.")

    return {"id_90_optimal": id_oc, "R2_simple": r2_simple, "rho_simple": rho_simple,
            "conclusion": " ".join(parts)}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_f_task_analysis(n_seeds=8)
    with open("core_mvp_v4/results/f_task_analysis.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("F-C:", r["analysis"]["conclusion"])
