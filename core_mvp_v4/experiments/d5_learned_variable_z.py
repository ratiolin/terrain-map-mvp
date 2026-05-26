"""D5: Learn Nonlinear Control Variable z = phi(h).

Autoencoder: h → z → a, z_dim=2 or 3.
Trains on rollout data from frozen encoder.
Analyzes z structure: R², Spearman with cost, clustering.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model


def _train(model, env, n_ep, ep_len, seed):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_ep):
        env.reset()
        for __ in range(ep_len):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            ns, risk, _, _ = env.step(action.squeeze(0).detach().numpy())
            loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


class ZAutoencoder(nn.Module):
    def __init__(self, hd, z_dim=2, k=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hd, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, k),
        )
    def forward(self, h):
        z = self.encoder(h)
        a_pred = self.decoder(z)
        return a_pred, z
    def encode_numpy(self, h):
        h_t = torch.from_numpy(h.astype(np.float32))
        with torch.no_grad():
            z = self.encoder(h_t).numpy()
        return z
    def decode_numpy(self, z):
        z_t = torch.from_numpy(z.astype(np.float32))
        with torch.no_grad():
            a = self.decoder(z_t).numpy()
        return a


def run_d5_learned_variable_z(n_seeds=8, d=16, k=2, hd=192, z_dim=2,
                              n_episodes=3, episode_length=2000, rollout_steps=3000):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        H_all = []; A_all = []; S_all = []; C_all = []
        for _ in range(rollout_steps):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            a = model.act_numpy(s)
            H_all.append(h); A_all.append(a); S_all.append(s)
            ns, risk, _, _ = env.step(a)
            C_all.append(risk)

        H_arr = np.array(H_all); A_arr = np.array(A_all)
        C_arr = np.array(C_all); S_arr = np.array(S_all)

        split = int(len(H_arr) * 0.7)
        H_train = torch.from_numpy(H_arr[:split].astype(np.float32))
        A_train = torch.from_numpy(A_arr[:split].astype(np.float32))
        H_test = H_arr[split:]; A_test = A_arr[split:]

        zmodel = ZAutoencoder(hd, z_dim, k)
        opt = torch.optim.Adam(zmodel.parameters(), lr=1e-3)
        for _ in range(500):
            idx = np.random.choice(len(H_train), min(256, len(H_train)), replace=False)
            h_b = H_train[idx]; a_b = A_train[idx]
            a_pred, z_b = zmodel(h_b)
            loss = nn.functional.mse_loss(a_pred, a_b)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            a_test_pred, _ = zmodel(torch.from_numpy(H_test.astype(np.float32)))
            ss_res = torch.sum((torch.from_numpy(A_test.astype(np.float32)) - a_test_pred)**2).item()
            ss_tot = torch.sum((torch.from_numpy(A_test.astype(np.float32)) - A_test.mean())**2).item()
            R2_val = 1 - ss_res / (ss_tot + 1e-8)

        Z_test = zmodel.encode_numpy(H_test)
        from scipy.stats import spearmanr
        rho_z0_cost, _ = spearmanr(Z_test[:, 0], C_arr[split:])
        rho_z1_cost, _ = spearmanr(Z_test[:, 1], C_arr[split:])
        rho_z0_anorm, _ = spearmanr(Z_test[:, 0], np.linalg.norm(A_test, axis=1))
        rho_z1_anorm, _ = spearmanr(Z_test[:, 1], np.linalg.norm(A_test, axis=1))

        from sklearn.metrics import silhouette_score
        drift_bins = np.digitize(S_arr[split:, 0], [-1, 0, 1])
        n_unique = len(set(drift_bins))
        if n_unique >= 2:
            sil = float(silhouette_score(Z_test[:500], drift_bins[:500]))
        else:
            sil = 0.0

        results["seeds"].append({
            "seed": seed,
            "R2": float(R2_val),
            "rho_z0_cost": float(rho_z0_cost), "rho_z1_cost": float(rho_z1_cost),
            "rho_z0_anorm": float(rho_z0_anorm), "rho_z1_anorm": float(rho_z1_anorm),
            "silhouette": sil,
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    R2_m = float(np.mean([s["R2"] for s in results["seeds"]]))
    sil_m = float(np.mean([s["silhouette"] for s in results["seeds"]]))
    rho_means = [max(abs(s["rho_z0_cost"]), abs(s["rho_z1_cost"])) for s in results["seeds"]]
    rho_m = float(np.mean(rho_means))

    if R2_m > 0.9 and rho_m > 0.3:
        conclusion = f"z IS CONTROL VARIABLE: R²={R2_m:.3f}, max |ρ|={rho_m:.3f}. z captures controllable dimensions."
    elif R2_m > 0.7:
        conclusion = f"z PARTIALLY CAPTURES CONTROL: R²={R2_m:.3f}, max |ρ|={rho_m:.3f}."
    else:
        conclusion = f"z UNDERFITS: R²={R2_m:.3f}. Action not recoverable from 2-dim bottleneck."

    return {"R2": R2_m, "silhouette": sil_m, "max_rho_task": rho_m, "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_d5_learned_variable_z(n_seeds=8)
    with open("core_mvp_v4/results/d5_learned_variable_z.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("D5:", r["analysis"]["conclusion"])
