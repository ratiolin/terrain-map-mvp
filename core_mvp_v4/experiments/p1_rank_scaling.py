"""P1-2: Rank-Performance Scaling Law.

Constrains policy head to specific ranks via W = A @ B (rank-limited).
Sweeps r ∈ [1,2,3,4,8,16,full_rank].
Finds optimal rank and measures rank-cost relationship.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell


class RankConstrainedModel(nn.Module):
    def __init__(self, d, hd, k, rank):
        super().__init__()
        self.d = d
        self.hd = hd
        self.k = k
        self.r = rank
        self.encoder = nn.Sequential(
            nn.Linear(d, hd), nn.ReLU(),
            nn.Linear(hd, hd), nn.ReLU(),
        )
        self.A = nn.Linear(rank, k, bias=False)
        self.B = nn.Linear(hd, rank, bias=False)
        self.predictor = nn.Sequential(
            nn.Linear(hd, hd), nn.ReLU(),
            nn.Linear(hd, 1), nn.Softplus(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.encoder(x)
        z = self.B(h)
        a = self.A(z)
        risk = self.predictor(h)
        return a, h, risk

    def act_numpy(self, s):
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            a, _, _ = self(s_t)
            return a.squeeze(0).numpy()

    def f_numpy(self, s):
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            _, h, _ = self(s_t)
            return h.squeeze(0).numpy()

    def effective_rank(self):
        W_eff = (self.A.weight.data @ self.B.weight.data).numpy()
        _, S, _ = np.linalg.svd(W_eff, full_matrices=False)
        return float(np.exp(-np.sum((S / S.sum()) * np.log(S / S.sum() + 1e-10))))


def _train(model, env, n_ep, ep_len, seed):
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
            loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def _behavioral(model, env, n_steps=500):
    env.reset()
    cost_sum = 0.0
    in_zone = 0
    for _ in range(n_steps):
        a = model.act_numpy(env.get_state())
        ns, risk, _, _ = env.step(a)
        cost_sum += risk
        if risk < 1.0:
            in_zone += 1
    return cost_sum / n_steps, in_zone / n_steps


def run_p1_rank_scaling(n_seeds=8, d=16, k=2, hd=192,
                        n_episodes=3, episode_length=2000):
    ranks = [1, 2, 3, 4, 8, 16, 0]
    results = {}

    for r in ranks:
        actual_r = r if r > 0 else k
        costs = []
        eff_ranks = []
        for seed in range(n_seeds):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = RankConstrainedModel(d, hd, k, actual_r)
            _train(model, env, n_episodes, episode_length, seed)
            c, z = _behavioral(model, env)
            costs.append(c)
            eff_ranks.append(model.effective_rank())
        label = f"r={r}" if r > 0 else "full_rank"
        results[label] = {
            "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
            "eff_rank_mean": float(np.mean(eff_ranks)),
            "constraint_rank": r,
        }

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    r_keys = sorted([k for k in results if k.startswith("r=") or k == "full_rank"],
                    key=lambda x: int(x.split("=")[-1]) if "=" in x else 999)
    cost_curve = [results[k]["cost_mean"] for k in r_keys]
    eff_curve = [results[k]["eff_rank_mean"] for k in r_keys]

    best_idx = int(np.argmin(cost_curve))
    best_r = r_keys[best_idx]

    if best_r == "r=2":
        conclusion = f"OPTIMAL RANK=2: r=2 gives lowest cost ({cost_curve[best_idx]:.4f}). System self-organizes to optimal rank."
    elif best_r == "full_rank":
        conclusion = "FULL RANK WINS: higher rank gives monotonically better performance."
    else:
        conclusion = f"OPTIMAL r={best_r}: cost={cost_curve[best_idx]:.4f}. Rank not exactly 2."

    return {"cost_curve": dict(zip(r_keys, cost_curve)),
            "eff_rank_curve": dict(zip(r_keys, eff_curve)),
            "best_rank": best_r, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_p1_rank_scaling(n_seeds=8)
    with open("core_mvp_v4/results/p1_rank_scaling.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("P1-2:", r["analysis"]["conclusion"])
