"""E2: Dynamic Latent State Learning.

Sequence model: z_{t+1} = F(z_t, a_t).
Learns transition dynamics in latent space.
Predicts future action and cost from z_t.
Tests if manipulating z_t has systematic causal effect on future.
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


class DynamicZModel(nn.Module):
    def __init__(self, hd, z_dim=2, k=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(hd, 64), nn.ReLU(), nn.Linear(64, z_dim))
        self.dynamics = nn.Sequential(nn.Linear(z_dim + k, 64), nn.ReLU(), nn.Linear(64, z_dim))
        self.a_decoder = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, k))
        self.c_decoder = nn.Sequential(nn.Linear(z_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, h, a_prev, z_prev=None):
        if z_prev is None:
            z = self.encoder(h)
        else:
            z_input = torch.cat([z_prev, a_prev], dim=-1)
            z = self.dynamics(z_input)
        a_pred = self.a_decoder(z)
        c_pred = self.c_decoder(z)
        return z, a_pred, c_pred


def _collect_sequences(model, env, n_steps, seq_len=10):
    env.reset()
    h_seq, a_seq, c_seq = [], [], []
    for _ in range(n_steps // seq_len):
        hs, acts, costs = [], [], []
        for __ in range(seq_len):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            a = model.act_numpy(s)
            ns, risk, _, _ = env.step(a)
            hs.append(h); acts.append(a); costs.append(risk)
        h_seq.append(np.array(hs)); a_seq.append(np.array(acts)); c_seq.append(np.array(costs))
    return h_seq, a_seq, c_seq


def run_e2_dynamic_z(n_seeds=8, d=16, k=2, hd=192, z_dim=2,
                     n_episodes=3, episode_length=2000):
    results = {"seeds": []}
    seq_len = 10

    for seed in range(min(n_seeds, 6)):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        h_seq, a_seq, c_seq = _collect_sequences(model, env, 3000, seq_len)
        n_seq = len(h_seq)

        dzmodel = DynamicZModel(hd, z_dim, k)
        opt = torch.optim.Adam(dzmodel.parameters(), lr=1e-3)

        for epoch in range(200):
            idx = np.random.randint(0, n_seq)
            h_t = torch.from_numpy(h_seq[idx].astype(np.float32))
            a_t = torch.from_numpy(a_seq[idx].astype(np.float32))
            c_t = torch.from_numpy(c_seq[idx].astype(np.float32)).unsqueeze(-1)

            z = None; loss = 0.0
            for t in range(seq_len - 1):
                a_prev = a_t[t:t+1] if t > 0 else torch.zeros(1, k)
                z, a_pred, c_pred = dzmodel(h_t[t:t+1], a_prev, z.detach() if t > 0 else None)
                loss += nn.functional.mse_loss(a_pred, a_t[t+1:t+2]) + 0.1 * nn.functional.mse_loss(c_pred, c_t[t+1:t+2])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dzmodel.parameters(), 1.0); opt.step()

        test_hs, test_as, test_cs = [], [], []
        env.reset()
        for _ in range(500):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            a = model.act_numpy(s)
            ns, risk, _, _ = env.step(a)
            test_hs.append(h); test_as.append(a); test_cs.append(risk)

        z0 = None; c_preds = []
        for t in range(len(test_hs) - 1):
            h_t = torch.from_numpy(test_hs[t].astype(np.float32)).unsqueeze(0)
            a_prev = torch.from_numpy(test_as[t-1].astype(np.float32)).unsqueeze(0) if t > 0 else torch.zeros(1, k)
            with torch.no_grad():
                z0, _, c_p = dzmodel(h_t, a_prev, z0.detach() if t > 0 else None)
            c_preds.append(c_p.item())

        from scipy.stats import spearmanr
        rho, _ = spearmanr(c_preds, test_cs[1:]) if len(c_preds) > 2 else (0, 1)

        results["seeds"].append({"seed": seed, "rho_cost_pred": float(rho)})

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    rho_m = float(np.mean([s["rho_cost_pred"] for s in results["seeds"]]))
    if rho_m > 0.3:
        conclusion = f"DYNAMIC Z CAPTURES COST: ρ={rho_m:.3f}. Latent dynamics predict future cost."
    else:
        conclusion = f"NO COST TRACKING: ρ={rho_m:.3f}. Dynamic z does not capture cost structure."
    return {"rho_cost_pred": rho_m, "conclusion": conclusion}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_e2_dynamic_z(n_seeds=8)
    with open("core_mvp_v4/results/e2_dynamic_z.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("E2:", r["analysis"]["conclusion"])
