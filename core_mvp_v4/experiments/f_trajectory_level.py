"""F-A: Trajectory-Level Control Variable.

RNN encodes entire trajectory [h_1..h_T] into z_traj.
Predicts total cost from trajectory embedding.
Tests if control variable is trajectory-level, not instantaneous.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

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


def _collect_trajectory(model, env, traj_len):
    env.reset()
    H, A, C = [], [], []
    for _ in range(traj_len):
        s = env.get_state()
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h = model.encoder(s_t).squeeze(0).detach().numpy()
        a = model.act_numpy(s)
        ns, risk, _, _ = env.step(a)
        H.append(h); A.append(a); C.append(risk)
    return np.array(H), np.array(A), np.array(C)


def _extract_features(A_traj, C_traj):
    A_arr = np.array(A_traj)
    C_arr = np.array(C_traj)
    features = {}
    if len(A_arr) > 4:
        fft_vals = np.abs(np.fft.rfft(A_arr[:, 0]))
        features['dom_freq'] = float(np.argmax(fft_vals[1:]) + 1) if len(fft_vals) > 1 else 0
        if np.sum(fft_vals) > 0:
            p = fft_vals / np.sum(fft_vals)
            p = p[p > 0]
            features['spectral_entropy'] = float(-np.sum(p * np.log(p))) if len(p) > 0 else 0
        else:
            features['spectral_entropy'] = 0.0

    switches = 0; prev_low = C_arr[0] < 1.0
    for c in C_arr[1:]:
        current_low = c < 1.0
        if current_low != prev_low: switches += 1
        prev_low = current_low
    features['n_switches'] = switches
    features['mean_cost'] = float(np.mean(C_arr))
    features['std_cost'] = float(np.std(C_arr))
    features['final_cost'] = float(C_arr[-1])
    cumsum = np.cumsum(C_arr)
    features['cost_auc'] = float(cumsum[-1] / len(C_arr))
    return features


class TrajectoryEncoder(nn.Module):
    def __init__(self, hd, z_dim=4, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(hd, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(hidden * 2, z_dim))
        self.predictor = nn.Sequential(nn.Linear(z_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, h_seq):
        out, _ = self.rnn(h_seq)
        final = torch.cat([out[:, -1, :self.rnn.hidden_size], out[:, 0, self.rnn.hidden_size:]], dim=-1)
        z = self.proj(final)
        c_pred = self.predictor(z)
        return z, c_pred


def run_f_trajectory_level(n_seeds=8, d=16, k=2, hd=192,
                           n_episodes=3, episode_length=2000):
    results = {"seeds": []}
    traj_len = 500
    n_trajs = 30

    for seed in range(min(n_seeds, 6)):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        traj_H, traj_feats, traj_costsum = [], [], []
        for _ in range(n_trajs):
            H_arr, A_arr, C_arr = _collect_trajectory(model, env, traj_len)
            traj_H.append(H_arr)
            ft = _extract_features(A_arr, C_arr)
            traj_feats.append(ft)
            traj_costsum.append(float(np.sum(C_arr)))

        h_seqs = torch.from_numpy(np.array(traj_H).astype(np.float32))
        cost_sums = torch.from_numpy(np.array(traj_costsum).astype(np.float32))

        tenc = TrajectoryEncoder(hd, z_dim=4)
        opt = torch.optim.Adam(tenc.parameters(), lr=1e-3)
        for _ in range(300):
            z_b, c_pred = tenc(h_seqs[:20])
            loss = nn.functional.mse_loss(c_pred.squeeze(), cost_sums[:20])
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            z_all, c_pred_all = tenc(h_seqs)
            ss_res = torch.sum((cost_sums - c_pred_all.squeeze())**2).item()
            ss_tot = torch.sum((cost_sums - cost_sums.mean())**2).item()
            R2 = 1 - ss_res / (ss_tot + 1e-8)

        z_np = z_all.numpy()
        rhos = []
        for j in range(z_np.shape[1]):
            rho, _ = spearmanr(z_np[:, j], traj_costsum)
            rhos.append(abs(rho))
        max_rho = max(rhos)

        results["seeds"].append({
            "seed": seed, "R2_cost_pred": float(R2), "max_rho": float(max_rho),
        })

    results["analysis"] = _analyze(results)
    return results


def _analyze(results):
    R2_m = float(np.mean([s["R2_cost_pred"] for s in results["seeds"]]))
    rho_m = float(np.mean([s["max_rho"] for s in results["seeds"]]))
    if rho_m > 0.6:
        conclusion = f"TRAJECTORY-LEVEL z WORKS: max ρ={rho_m:.3f}, R²={R2_m:.3f}. Cost is trajectory-compressible."
    elif rho_m > 0.3:
        conclusion = f"PARTIAL: max ρ={rho_m:.3f}. Some trajectory-level structure."
    else:
        conclusion = f"NO TRAJECTORY VARIABLE: max ρ={rho_m:.3f}, R²={R2_m:.3f}. Cost not trackable at trajectory level."
    return {"R2_cost": R2_m, "max_rho": rho_m, "conclusion": conclusion}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_f_trajectory_level(n_seeds=8)
    with open("core_mvp_v4/results/f_trajectory_level.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("F-A:", r["analysis"]["conclusion"])
