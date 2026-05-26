"""H2: Gradient Field Evolution.

Saves checkpoints during training, measures field smoothness/Lipschitz
at each checkpoint. Determines if roughness is inherent or emergent.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from core_mvp_v4.env import MultiDimDoubleWell


def _cost_of_h(model, env, h, s0):
    a = model.actor(torch.from_numpy(h.astype(np.float32)).unsqueeze(0)).squeeze(0).detach().numpy()
    return float(np.linalg.norm(env.forward_static(s0, a)[:2]))


def _grad_h(model, env, h, s0, eps=0.005):
    d = len(h); g = np.zeros(d)
    for _ in range(6):
        delta = np.random.randn(d) * eps / np.sqrt(d)
        g += (_cost_of_h(model, env, h + delta, s0) - _cost_of_h(model, env, h - delta, s0)) * delta / (12 * eps**2 / d)
    return g / (np.linalg.norm(g) + 1e-8)


def _field_metrics(model, env, d, n_points=60):
    env_state = env.save_state()
    env.reset()
    H, G, S = [], [], []
    for _ in range(n_points):
        s = env.get_state()
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h = model.encoder(s_t).squeeze(0).detach().numpy()
        g = _grad_h(model, env, h, s)
        H.append(h); G.append(g); S.append(s)
        env.step(model.act_numpy(s))
    H_arr = np.array(H); G_arr = np.array(G)
    nn = NearestNeighbors(n_neighbors=10).fit(H_arr)
    smooth = []; lip = []
    for i in range(len(H_arr)):
        _, nbrs = nn.kneighbors([H_arr[i]], 10)
        for j in nbrs[0][1:]:
            smooth.append(float(np.dot(G_arr[i], G_arr[j]) / (np.linalg.norm(G_arr[i]) * np.linalg.norm(G_arr[j]) + 1e-8)))
            d_ij = np.linalg.norm(H_arr[i] - H_arr[j])
            if d_ij > 1e-6: lip.append(np.linalg.norm(G_arr[i] - G_arr[j]) / d_ij)
    env.restore_state(env_state)
    return float(np.mean(smooth)) if smooth else 0.0, float(np.mean(lip)) if lip else 0.0


def run_h2_field_evolution(n_seeds=4, d=16, k=2, hd=192,
                            n_episodes=3, episode_length=2000, n_ckpts=6):
    from core_mvp_v4.models import V4Model
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        if seed is not None: torch.manual_seed(seed); np.random.seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        total_steps = n_episodes * episode_length
        ckpt_step = max(1, total_steps // n_ckpts)
        ckpt_data = []; costs = []; step_counter = 0

        for ep in range(n_episodes):
            env.reset()
            for st in range(episode_length):
                s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
                action, h, risk_pred = model(s_t)
                a_np = action.squeeze(0).detach().numpy()
                ns, risk, _, _ = env.step(a_np)
                loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

                if step_counter % ckpt_step == 0:
                    sm, lp = _field_metrics(model, env, d, n_points=40)
                    ckpt_data.append({"step": step_counter, "smoothness": sm, "lipschitz": lp})
                    costs.append(float(risk))
                step_counter += 1

        sm_curve = [c["smoothness"] for c in ckpt_data]
        lip_curve = [c["lipschitz"] for c in ckpt_data]
        trend_sm = sm_curve[-1] - sm_curve[0] if len(sm_curve) > 1 else 0.0

        results["seeds"].append({
            "seed": seed, "ckpt_data": ckpt_data,
            "smooth_init": sm_curve[0] if sm_curve else 0,
            "smooth_final": sm_curve[-1] if sm_curve else 0,
            "trend": trend_sm,
        })

    init_sm = float(np.mean([s["smooth_init"] for s in results["seeds"]]))
    final_sm = float(np.mean([s["smooth_final"] for s in results["seeds"]]))
    trend = float(np.mean([s["trend"] for s in results["seeds"]]))

    if abs(trend) < 0.005:
        conclusion = f"BORN ROUGH: init={init_sm:.3f}, final={final_sm:.3f}. Roughness is inherent."
    elif trend > 0.005:
        conclusion = f"IMPROVING: smoothness {init_sm:.3f}→{final_sm:.3f}. Training structures the field."
    else:
        conclusion = f"DEGRADING: smoothness {init_sm:.3f}→{final_sm:.3f}. Structure lost during training."
    results["analysis"] = {"smooth_init": init_sm, "smooth_final": final_sm, "trend": trend,
                           "conclusion": conclusion}
    return results


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_h2_field_evolution(n_seeds=4)
    with open("core_mvp_v4/results/h2_field_evolution.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("H2:", r["analysis"]["conclusion"])
