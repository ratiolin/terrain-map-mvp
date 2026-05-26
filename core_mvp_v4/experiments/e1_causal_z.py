"""E1: Causal Constrained Control Variable Learning.

Trains z=φ(h) with structural constraints beyond pure action prediction:
- monotonicity: encourage consistent ∂cost/∂z_i sign
- Jacobian nuclear norm: low-rank ∂a/∂z  
- drift decoupling: include drift in decoder, measure z-drift correlation
- contrastive: same-cost samples map to close z

Compares all constraint types against baseline.
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


class CausalZModel(nn.Module):
    def __init__(self, hd, z_dim=2, k=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(hd, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, z_dim))
        self.decoder = nn.Sequential(nn.Linear(z_dim + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, k))

    def forward(self, h, drift=0.5):
        z = self.encoder(h)
        drift_t = torch.ones(h.shape[0], 1, device=h.device) * drift
        a = self.decoder(torch.cat([z, drift_t], dim=-1))
        return a, z


def _collect_data(model, env, n_steps):
    env.reset()
    H, A, C, D = [], [], [], []
    for _ in range(n_steps):
        s = env.get_state()
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h = model.encoder(s_t).squeeze(0).detach().numpy()
        a = model.act_numpy(s)
        ns, risk, _, info = env.step(a)
        H.append(h); A.append(a); C.append(risk); D.append(info.get('drift', 0.5))
    return np.array(H), np.array(A), np.array(C), np.array(D)


def _train_z(zmodel, H_arr, A_arr, C_arr, D_arr, constraint, epochs=300, batch=256, lam=0.1):
    H_t = torch.from_numpy(H_arr.astype(np.float32)); A_t = torch.from_numpy(A_arr.astype(np.float32))
    C_t = torch.from_numpy(C_arr.astype(np.float32)); D_t = torch.from_numpy(D_arr.astype(np.float32))
    opt = torch.optim.Adam(zmodel.parameters(), lr=1e-3)
    losses = []
    for _ in range(epochs):
        idx = np.random.choice(len(H_t), min(batch, len(H_t)), replace=False)
        h_b, a_b, c_b, d_b = H_t[idx], A_t[idx], C_t[idx], D_t[idx]
        a_pred, z_b = zmodel(h_b, d_b.mean().item())
        pred_loss = nn.functional.mse_loss(a_pred, a_b)
        extra_loss = 0.0

        if constraint == "monotonicity" and z_b.shape[1] >= 1:
            with torch.enable_grad():
                z0 = z_b[:, 0].detach().clone().requires_grad_(True)
                a_z0_pred = zmodel.decoder(torch.cat([z0.unsqueeze(-1), z_b[:, 1:], torch.ones(h_b.shape[0], 1) * d_b.mean().item()], dim=-1))
                cost_est = torch.sum(a_z0_pred**2, dim=1)
            extra_loss = lam * torch.mean(torch.clamp(c_b - cost_est.detach(), min=0)**2)

        elif constraint == "jacobian_nuclear":
            z_b_grad = z_b.detach().clone().requires_grad_(True)
            a_z = zmodel.decoder(torch.cat([z_b_grad, torch.ones(h_b.shape[0], 1) * d_b.mean().item()], dim=-1))
            Js = []
            for j in range(a_z.shape[1]):
                g = torch.autograd.grad(a_z[:, j].sum(), z_b_grad, retain_graph=True, allow_unused=True)[0]
                if g is not None: Js.append(g)
            if len(Js) >= 2:
                J_stack = torch.stack(Js, dim=0)
                extra_loss = lam * torch.norm(J_stack.reshape(J_stack.shape[0], -1), p='nuc')

        elif constraint == "decoupling":
            rho = torch.corrcoef(torch.stack([z_b[:, 0], d_b]))[0, 1]
            extra_loss = lam * (rho**2 if not torch.isnan(rho) else 0.0)

        elif constraint == "contrastive":
            c_sorted, idx_sorted = torch.sort(c_b)
            for i in range(0, len(c_sorted) - 2):
                z_close = z_b[idx_sorted[i:i+2]]
                z_far_idx = min(i + 10, len(idx_sorted) - 1)
                z_far = z_b[idx_sorted[z_far_idx:z_far_idx+1]]
                if len(z_far) > 0:
                    extra_loss += lam * max(0, 0.5 - torch.norm(z_close.mean(0) - z_far.mean(0)))

        loss = pred_loss + extra_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(zmodel.parameters(), 1.0); opt.step()
        losses.append(loss.item())
    return float(np.mean(losses[-50:])) if len(losses) >= 50 else 0.0


def run_e1_causal_z(n_seeds=8, d=16, k=2, hd=192, z_dim=2,
                    n_episodes=3, episode_length=2000):
    constraints = ["baseline", "monotonicity", "jacobian_nuclear", "decoupling", "contrastive"]
    results = {}

    for constraint in constraints:
        r2_vals = []; rho_vals = []; rho_drift = []
        for seed in range(min(n_seeds, 6)):
            env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
            model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
            _train(model, env, n_episodes, episode_length, seed)
            H_arr, A_arr, C_arr, D_arr = _collect_data(model, env, 2000)

            zmodel = CausalZModel(hd, z_dim, k)
            _train_z(zmodel, H_arr, A_arr, C_arr, D_arr, constraint)

            split = int(len(H_arr) * 0.7)
            H_test = torch.from_numpy(H_arr[split:].astype(np.float32))
            A_test = A_arr[split:]
            C_test = C_arr[split:]
            with torch.no_grad():
                a_pred, z_test = zmodel(H_test, D_arr[split:].mean())
                ss_res = torch.sum((torch.from_numpy(A_test.astype(np.float32)) - a_pred)**2).item()
                ss_tot = torch.sum((torch.from_numpy(A_test.astype(np.float32)) - A_test.mean())**2).item()
                R2 = 1 - ss_res / (ss_tot + 1e-8)
            z_np = z_test.numpy()
            rho_c, _ = spearmanr(z_np[:, 0], C_test) if z_np.shape[0] > 2 else (0, 1)
            rho_d, _ = spearmanr(z_np[:, 0], D_arr[split:]) if z_np.shape[0] > 2 else (0, 1)
            r2_vals.append(R2); rho_vals.append(abs(rho_c)); rho_drift.append(abs(rho_d))

        results[constraint] = {
            "R2_mean": float(np.mean(r2_vals)), "R2_std": float(np.std(r2_vals)),
            "rho_cost_mean": float(np.mean(rho_vals)), "rho_cost_std": float(np.std(rho_vals)),
            "rho_drift_mean": float(np.mean(rho_drift)),
        }

    results["analysis"] = _analyze(results, constraints)
    return results


def _analyze(results, constraints):
    best = max(constraints, key=lambda c: results[c]["rho_cost_mean"] if results[c]["R2_mean"] > 0.8 else 0)
    best_rho = results[best]["rho_cost_mean"]
    base_rho = results["baseline"]["rho_cost_mean"]

    if best_rho > 0.3 and best_rho > base_rho * 2:
        conclusion = f"CAUSAL ALIGNMENT: {best} achieves |ρ|={best_rho:.3f} vs baseline {base_rho:.3f}."
    elif best_rho > 0.15:
        conclusion = f"WEAK ALIGNMENT: best {best} ρ={best_rho:.3f}. Constraints have partial effect."
    else:
        conclusion = f"NO ALIGNMENT: best ρ={best_rho:.3f}. Causal constraints cannot align z with cost."
    return {"best_constraint": best, "best_rho": best_rho, "baseline_rho": base_rho, "conclusion": conclusion}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_e1_causal_z(n_seeds=8)
    with open("core_mvp_v4/results/e1_causal_z.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("E1:", r["analysis"]["conclusion"])
