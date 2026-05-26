"""V4-5: Multi-Agent Gradient Field Consistency.

Compares gradient fields from independently trained agents.
Tests whether gradient knowledge is transferable.
"""

import json
import numpy as np
import torch; import torch.nn as nn
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


def _cost_of_h(model, env, h, s0):
    a = model.actor(torch.from_numpy(h.astype(np.float32)).unsqueeze(0)).squeeze(0).detach().numpy()
    return float(np.linalg.norm(env.forward_static(s0, a)[:2]))


def _grad_h(model, env, h, s0, eps=0.01):
    d = len(h); g = np.zeros(d)
    for _ in range(10):
        delta = np.random.randn(d) * eps / np.sqrt(d)
        g += (_cost_of_h(model, env, h + delta, s0) - _cost_of_h(model, env, h - delta, s0)) * delta / (20 * eps**2 / d)
    norm = np.linalg.norm(g)
    return g / norm if norm > 1e-8 else g


def _collect_gradients(model, env_state, s_list, n_pts):
    env = MultiDimDoubleWell(d=16, k=2, drift=0.5, seed=0, coupling=0.0)
    env.restore_state(env_state)
    G, C = [], []
    for i in range(min(n_pts, len(s_list))):
        s = s_list[i]
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h = model.encoder(s_t).squeeze(0).detach().numpy()
        g = _grad_h(model, env, h, s)
        G.append(g); C.append(_cost_of_h(model, env, h, s))
    return np.array(G), np.array(C)


def run_v4_5_multiagent(n_pairs=3, n_seeds=4, d=16, k=2, hd=192, n_episodes=3, episode_length=2000):
    ep_len = episode_length
    results = []

    np.random.seed(n_seeds); torch.manual_seed(n_seeds)
    shared_env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=0, coupling=0.0)
    shared_env.reset()
    shared_s_list = []
    for _ in range(200):
        shared_s_list.append(shared_env.get_state())
        shared_env.step(np.zeros(k))
    env_snapshot = shared_env.save_state()

    for pair in range(n_pairs):
        seed_a, seed_b = pair * 100 + 1, pair * 100 + 2
        env_a = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed_a, coupling=0.0)
        model_a = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model_a, env_a, n_episodes, ep_len, seed_a)

        env_b = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed_b, coupling=0.0)
        model_b = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model_b, env_b, n_episodes, ep_len, seed_b)

        G_a, C_a = _collect_gradients(model_a, env_snapshot, shared_s_list, 100)
        G_b, C_b = _collect_gradients(model_b, env_snapshot, shared_s_list, 100)

        cos_sims = []
        for i in range(len(G_a)):
            cos = np.dot(G_a[i], G_b[i]) / (np.linalg.norm(G_a[i]) * np.linalg.norm(G_b[i]) + 1e-8)
            cos_sims.append(float(cos))

        transfer_costs = []
        for i in range(min(20, len(G_a))):
            s_test = shared_s_list[i]
            s_t = torch.from_numpy(s_test.astype(np.float32)).unsqueeze(0)
            h_test = model_b.encoder(s_t).squeeze(0).detach().numpy()
            g_test = G_a[i]
            h_mod = h_test - 0.05 * g_test
            a = model_b.actor(torch.from_numpy(h_mod.astype(np.float32)).unsqueeze(0))
            a = a.squeeze(0).detach().numpy()
            ns = env_b.forward_static(s_test, a)
            transfer_costs.append(float(np.linalg.norm(ns[:2])))

        results.append({
            "pair": pair,
            "cos_alignment": float(np.mean(cos_sims)),
            "cos_std": float(np.std(cos_sims)),
            "transfer_cost_mean": float(np.mean(transfer_costs)),
        })

    align = float(np.mean([r["cos_alignment"] for r in results]))

    if align > 0.3:
        conclusion = f"MODERATE CROSS-AGENT ALIGNMENT: cos={align:.3f}. Gradient fields partially transferable."
    elif align > 0.1:
        conclusion = f"WEAK CROSS-AGENT: cos={align:.3f}. Fields have minimal consistency."
    else:
        conclusion = f"NO CROSS-AGENT CONSISTENCY: cos={align:.3f}. Each agent learns different field."
    return {"analysis": {"cross_agent_alignment": align, "conclusion": conclusion}}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_v4_5_multiagent(n_pairs=3)
    with open("core_mvp_v4/results/g5_multiagent.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("V4-5:", r["analysis"]["conclusion"])
