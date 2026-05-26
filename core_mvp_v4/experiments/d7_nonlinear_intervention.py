"""D7: Nonlinear Control Variable Intervention.

Intervenes in z-space (translate, scale, gradient-directed).
Measures action change and cost change per intervention mode.
Scans intervention amplitude, builds dose-response curves.
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
        z = self.encoder(h); a = self.decoder(z); return a, z
    def encode_numpy(self, h):
        h_t = torch.from_numpy(h.astype(np.float32))
        with torch.no_grad(): z = self.encoder(h_t).numpy()
        return z
    def decode_numpy(self, z):
        z_t = torch.from_numpy(z.astype(np.float32))
        with torch.no_grad(): a = self.decoder(z_t).numpy()
        return a


def _intervene_z(zmodel, policy_head, h_orig, mode="translate", dim=0, delta=0.5):
    z_orig = zmodel.encode_numpy(h_orig.reshape(1, -1)).squeeze()
    a_orig = policy_head(torch.from_numpy(h_orig.astype(np.float32)).unsqueeze(0))
    a_orig = a_orig.squeeze(0).detach().numpy()

    z_mod = z_orig.copy()
    if mode == "translate":
        z_mod[dim] += delta
    elif mode == "scale":
        z_mod[dim] *= (1 + delta)
    elif mode == "gradient":
        z_t = torch.from_numpy(z_orig.astype(np.float32)).requires_grad_(True)
        a_pred = zmodel.decoder(z_t)
        grads = []
        for j in range(a_pred.shape[0]):
            g = torch.autograd.grad(a_pred[j], z_t, retain_graph=True, allow_unused=True)[0]
            if g is not None: grads.append(g.detach().numpy())
        grad_dir = np.mean(grads, axis=0)
        grad_dir = grad_dir / (np.linalg.norm(grad_dir) + 1e-8)
        z_mod = z_orig + delta * grad_dir
    elif mode == "random":
        r_dir = np.random.randn(*z_orig.shape)
        r_dir = r_dir / (np.linalg.norm(r_dir) + 1e-8)
        z_mod = z_orig + delta * r_dir

    a_mod = zmodel.decode_numpy(z_mod.reshape(1, -1)).squeeze()
    action_change = float(np.linalg.norm(a_mod - a_orig))
    return action_change, z_orig, z_mod, a_orig, a_mod


def run_d7_nonlinear_intervention(n_seeds=8, d=16, k=2, hd=192, z_dim=2,
                                   n_episodes=3, episode_length=2000):
    deltas = [0.1, 0.5, 1.0, 2.0]
    modes = ["translate", "scale", "gradient", "random"]
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        H_all = []; A_all = []
        for _ in range(2000):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            a = model.act_numpy(s)
            H_all.append(h); A_all.append(a)
            env.step(a)
        H_arr = np.array(H_all); A_arr = np.array(A_all)

        zmodel = ZAutoencoder(hd, z_dim, k)
        opt = torch.optim.Adam(zmodel.parameters(), lr=1e-3)
        H_t = torch.from_numpy(H_arr.astype(np.float32)); A_t = torch.from_numpy(A_arr.astype(np.float32))
        for _ in range(300):
            idx = np.random.choice(len(H_t), min(256, len(H_t)), replace=False)
            a_pred, _ = zmodel(H_t[idx])
            loss = nn.functional.mse_loss(a_pred, A_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()

        test_idx = np.random.choice(len(H_arr), 50, replace=False)
        seed_data = {"seed": seed, "modes": {}}

        for mode in modes:
            mode_data = {}
            for delta_val in deltas:
                act_changes = []
                for idx in test_idx:
                    h_orig = H_arr[idx]
                    for dim in range(z_dim):
                        ac, _, _, _, _ = _intervene_z(zmodel, model.actor, h_orig,
                                                      mode=mode, dim=dim, delta=delta_val)
                        act_changes.append(ac)
                mode_data[f"delta_{delta_val}"] = {
                    "action_change_mean": float(np.mean(act_changes)),
                    "action_change_max": float(np.max(act_changes)),
                }
            seed_data["modes"][mode] = mode_data
        results["seeds"].append(seed_data)

    results["analysis"] = _analyze(results, modes, deltas)
    return results


def _analyze(results, modes, deltas):
    ratios = {}
    for mode in modes:
        for dv in deltas:
            means = [s["modes"][mode][f"delta_{dv}"]["action_change_mean"] for s in results["seeds"]]
            ratios[f"{mode}_{dv}"] = float(np.mean(means))

    grad_max = max((ratios.get(f"gradient_{dv}", 0), dv) for dv in deltas)
    rand_max = max((ratios.get(f"random_{dv}", 0), dv) for dv in deltas)
    grad_vs_rand = grad_max[0] / (rand_max[0] + 1e-6)

    if grad_vs_rand > 3.0:
        conclusion = f"z IS CAUSAL: gradient intervention {grad_vs_rand:.1f}x vs random at Δ={grad_max[1]}."
    elif grad_vs_rand > 1.5:
        conclusion = f"z MODERATELY CAUSAL: {grad_vs_rand:.1f}x advantage."
    else:
        conclusion = f"z NOT CAUSAL: gradient={grad_max[0]:.4f} vs random={rand_max[0]:.4f}. No advantage."

    return {"grad_vs_rand": grad_vs_rand, "gradient_max": grad_max[0], "random_max": rand_max[0],
            "conclusion": conclusion}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_d7_nonlinear_intervention(n_seeds=8)
    with open("core_mvp_v4/results/d7_nonlinear_intervention.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("D7:", r["analysis"]["conclusion"])
