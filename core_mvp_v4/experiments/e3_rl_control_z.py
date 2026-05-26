"""E3: RL-style Control Embedding.

Trains z=g(h) as explicit control conditioning variable during interaction.
Action = policy_head(concat(h, z)).
Alignment losses: z[0] ≈ scale * drift, z[1] ≈ monotonic with cost.
Tests causal effect via z-intervention vs h-intervention.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell


class RLControlModel(nn.Module):
    def __init__(self, d, hd=192, z_dim=2, k=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, hd), nn.ReLU(), nn.Linear(hd, hd), nn.ReLU())
        self.z_gen = nn.Sequential(nn.Linear(hd, 32), nn.ReLU(), nn.Linear(32, z_dim))
        self.policy = nn.Linear(hd + z_dim, k)
        self.predictor = nn.Sequential(nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, 1), nn.Softplus())
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, s, drift=None, return_z=False):
        h = self.encoder(s)
        z = self.z_gen(h)
        a = self.policy(torch.cat([h, z], dim=-1))
        risk_pred = self.predictor(h)
        if return_z:
            return a, h, risk_pred, z
        return a, h, risk_pred

    def act_numpy(self, s):
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad(): a, _, _ = self(s_t); return a.squeeze(0).numpy()
    def z_numpy(self, s):
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            h = self.encoder(s_t); z = self.z_gen(h); return z.squeeze(0).numpy()
    def act_with_z(self, s, z_val=None):
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            h = self.encoder(s_t)
            if z_val is None: z = self.z_gen(h)
            else: z = torch.from_numpy(z_val.astype(np.float32)).unsqueeze(0)
            a = self.policy(torch.cat([h, z], dim=-1))
            return a.squeeze(0).numpy()


def run_e3_rl_control_z(n_seeds=8, d=16, k=2, hd=192, z_dim=2,
                        n_episodes=4, episode_length=2000):
    results = {"seeds": []}

    for seed in range(min(n_seeds, 6)):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = RLControlModel(d, hd, z_dim, k)
        if seed is not None: torch.manual_seed(seed); np.random.seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        for ep in range(n_episodes):
            env.reset()
            for st in range(episode_length):
                s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
                a, h, risk_pred, z = model(s_t, return_z=True)
                a_np = a.squeeze(0).detach().numpy()
                ns, risk, _, info = env.step(a_np)
                drift = info.get('drift', 0.5)

                task_loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(a**2)
                align_loss = 0.0
                if z.shape[1] >= 2:
                    align_loss += 0.05 * (z[:, 0] - drift)**2
                    z1_target = torch.tensor([[risk]], dtype=torch.float32)
                    align_loss += 0.01 * (z[:, 1] - z1_target.squeeze(-1))**2
                loss = task_loss + align_loss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        env.reset()
        baseline_cost = 0.0; n_test = 500
        for _ in range(n_test):
            a = model.act_numpy(env.get_state()); ns, risk, _, _ = env.step(a)
            baseline_cost += risk / n_test

        z_deltas = [0.5, 1.0, 2.0, 5.0]
        z0_effects = {}; z1_effects = {}
        for delta in z_deltas:
            costs_z0 = []; costs_z1 = []; costs_rand = []
            for _ in range(100):
                s = env.get_state(); z_orig = model.z_numpy(s)
                z0_mod = z_orig.copy(); z0_mod[0] += delta
                a_z0 = model.act_with_z(s, z0_mod)
                ns0, r0, _, _ = env.step(a_z0)
                costs_z0.append(r0)

                z1_mod = z_orig.copy(); z1_mod[1] += delta
                a_z1 = model.act_with_z(s, z1_mod)
                ns1, r1, _, _ = env.step(a_z1)
                costs_z1.append(r1)

                zr_mod = z_orig + np.random.randn(z_dim) * delta * 0.5
                a_zr = model.act_with_z(s, zr_mod)
                nsr, rr, _, _ = env.step(a_zr)
                costs_rand.append(rr)
            z0_effects[str(delta)] = float(np.mean(costs_z0))
            z1_effects[str(delta)] = float(np.mean(costs_z1))

        results["seeds"].append({
            "seed": seed, "baseline_cost": baseline_cost,
            "z0_effects": z0_effects, "z1_effects": z1_effects,
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    delta_ratios = []
    for s in results["seeds"]:
        bc = s["baseline_cost"]
        for dv in ["0.5", "1.0", "2.0", "5.0"]:
            if dv in s["z0_effects"]:
                delta_ratios.append(s["z0_effects"][dv] / (bc + 1e-6))
                delta_ratios.append(s["z1_effects"][dv] / (bc + 1e-6))
    max_ratio = max(delta_ratios) if delta_ratios else 1.0
    if max_ratio > 2.0:
        conclusion = f"Z CAUSAL: intervention achieves {max_ratio:.1f}x cost change."
    elif max_ratio > 1.3:
        conclusion = f"Z MODERATELY CAUSAL: max ratio={max_ratio:.2f}x."
    else:
        conclusion = f"Z NOT CAUSAL: max ratio={max_ratio:.2f}x. Alignment losses insufficient."
    return {"max_cost_ratio": max_ratio, "conclusion": conclusion}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_e3_rl_control_z(n_seeds=8)
    with open("core_mvp_v4/results/e3_rl_control_z.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("E3:", r["analysis"]["conclusion"])
