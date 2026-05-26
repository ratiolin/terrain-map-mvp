"""H3: Coordination Layer Separation.

Tests at which layer coordination between agents occurs:
representation (h), gradient field (g), or output (a).
Uses two independently trained agents on shared state set.
"""

import json
import numpy as np
import torch
import torch.nn as nn

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


def _grad_h(model, env, h, s0, eps=0.005):
    d = len(h); g = np.zeros(d)
    for _ in range(6):
        delta = np.random.randn(d) * eps / np.sqrt(d)
        g += (_cost_of_h(model, env, h + delta, s0) - _cost_of_h(model, env, h - delta, s0)) * delta / (12 * eps**2 / d)
    return g / (np.linalg.norm(g) + 1e-8)


def _cka(X, Y):
    X = X - X.mean(axis=0); Y = Y - Y.mean(axis=0)
    Kx = X @ X.T; Ky = Y @ Y.T
    hsic = np.trace(Kx @ Ky)
    norm_x = np.sqrt(np.trace(Kx @ Kx)); norm_y = np.sqrt(np.trace(Ky @ Ky))
    return hsic / (norm_x * norm_y + 1e-8)


def run_h3_coordination_layer(n_pairs=3, d=16, k=2, hd=192,
                               n_episodes=3, episode_length=2000, n_seeds=None):
    results = []

    shared_env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=0, coupling=0.0)
    shared_env.reset()
    shared_s = []
    for _ in range(200):
        shared_s.append(shared_env.get_state())
        shared_env.step(np.zeros(k))

    for pair in range(n_pairs):
        sa, sb = pair * 100 + 1, pair * 100 + 2
        env_a = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=sa, coupling=0.0)
        ma = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(ma, env_a, n_episodes, episode_length, sa)

        env_b = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=sb, coupling=0.0)
        mb = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(mb, env_b, n_episodes, episode_length, sb)

        Ha, Hb, Hc = [], [], []
        Ga, Gb = [], []
        Aa, Ab = [], []
        for si in shared_s:
            s_t = torch.from_numpy(si.astype(np.float32)).unsqueeze(0)
            ha = ma.encoder(s_t).squeeze(0).detach().numpy()
            hb = mb.encoder(s_t).squeeze(0).detach().numpy()
            aa = ma.act_numpy(si); ab = mb.act_numpy(si)
            Ha.append(ha); Hb.append(hb); Hc.append(ha + hb)
            ga = _grad_h(ma, env_a, ha, si); gb = _grad_h(mb, env_b, hb, si)
            Ga.append(ga); Gb.append(gb)
            Aa.append(aa); Ab.append(ab)

        cka_rep = _cka(np.array(Ha), np.array(Hb))
        cos_grad = float(np.mean([np.dot(Ga[i], Gb[i]) / (np.linalg.norm(Ga[i]) * np.linalg.norm(Gb[i]) + 1e-8)
                                   for i in range(len(Ga))]))
        cos_act = float(np.mean([np.dot(Aa[i], Ab[i]) / (np.linalg.norm(Aa[i]) * np.linalg.norm(Ab[i]) + 1e-8)
                                  for i in range(len(Aa))]))

        results.append({
            "pair": pair,
            "CKA_representation": float(cka_rep),
            "cos_gradient_field": float(cos_grad),
            "cos_action_output": float(cos_act),
        })

    cka_m = float(np.mean([r["CKA_representation"] for r in results]))
    cosg_m = float(np.mean([r["cos_gradient_field"] for r in results]))
    cosa_m = float(np.mean([r["cos_action_output"] for r in results]))

    if cosa_m > 0.8 and cka_m < 0.3 and cosg_m < 0.1:
        conclusion = f"OUTPUT-LAYER COORDINATION: action cos={cosa_m:.3f}, CKA={cka_m:.3f}, grad cos={cosg_m:.3f}."
    elif cka_m > 0.5:
        conclusion = f"REPRESENTATION-LAYER COORDINATION: CKA={cka_m:.3f}."
    elif cosg_m > 0.3:
        conclusion = f"GRADIENT-LAYER COORDINATION: grad cos={cosg_m:.3f}."
    else:
        conclusion = f"MIXED: CKA={cka_m:.3f}, grad={cosg_m:.3f}, action={cosa_m:.3f}."
    return {"analysis": {"CKA_rep": cka_m, "cos_grad": cosg_m, "cos_action": cosa_m,
                         "conclusion": conclusion}}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_h3_coordination_layer(n_pairs=3)
    with open("core_mvp_v4/results/h3_coordination_layer.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("H3:", r["analysis"]["conclusion"])
