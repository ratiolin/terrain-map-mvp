"""Direction A: Nonlinear Decodability of Control Information.

Tests whether control information is nonlinearly encoded in hidden states.

A1: MLP probe vs linear probe comparison.
A2: Layer-wise decoding (at each encoder layer).
A3: Subspace vs full space information content.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, compute_jacobian
from core_mvp_v4.metrics import compute_k80, alignment


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
            risk_t = torch.tensor([[risk]], dtype=torch.float32)
            loss = nn.functional.mse_loss(risk_pred, risk_t) + 0.1 * torch.mean(action**2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


class LayerwiseExtractor:
    def __init__(self, model):
        self.model = model
        self.layer_outputs = {}

    def extract(self, s):
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        x = s_t
        layer_idx = 0
        outputs = []
        for module in self.model.encoder:
            x = module(x)
            if isinstance(module, nn.ReLU):
                outputs.append(x.squeeze(0).detach().numpy())
                layer_idx += 1
        outputs.append(x.squeeze(0).detach().numpy())
        return outputs


def run_a_nonlinear_decoding(n_seeds=8, d=16, k=2, hd=192,
                             n_episodes=3, episode_length=2000):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        # Collect hidden states and controllability labels
        env.reset()
        h_collected = []
        C_collected = []
        layer_outputs = {0: [], 1: []}
        extractor = LayerwiseExtractor(model)

        for _ in range(500):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            h_collected.append(h)

            a = model.act_numpy(s)
            sa = env.forward_static(s, a)
            sz = env.forward_static(s, np.zeros(env.k))
            C_val = float(np.linalg.norm(sa[:k] - sz[:k]))
            C_collected.append(C_val)

            layers = extractor.extract(s)
            for i, lh in enumerate(layers[:2]):
                if i not in layer_outputs:
                    layer_outputs[i] = []
                layer_outputs[i].append(lh)

            ns, _, _, _ = env.step(a)

        h_array = np.array(h_collected)
        C_array = np.array(C_collected).reshape(-1, 1).ravel()

        split = int(len(h_array) * 0.7)
        h_train, h_test = h_array[:split], h_array[split:]
        C_train, C_test = C_array[:split], C_array[split:]

        # A1: linear vs MLP probe
        lin = LinearRegression().fit(h_train, C_train)
        lin_r2 = float(lin.score(h_test, C_test))

        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=seed)
        try:
            mlp.fit(h_train, C_train)
            mlp_r2 = float(mlp.score(h_test, C_test))
        except Exception:
            mlp_r2 = 0.0

        # A2: layer-wise decoding
        layer_r2 = {}
        for li, lh_list in layer_outputs.items():
            if len(lh_list) == 0:
                continue
            lh_arr = np.array(lh_list)
            lh_train, lh_test = lh_arr[:split], lh_arr[split:]
            layer_probe = LinearRegression().fit(lh_train, C_train)
            layer_r2[f"layer_{li}"] = float(layer_probe.score(lh_test, C_test))

        # A3: subspace vs full space (use U in hidden space)
        J_geo = np.mean([compute_jacobian(model, env.get_state()) for _ in range(30)], axis=0)
        U_geo, S_geo, Vt_geo = np.linalg.svd(J_geo, full_matrices=False)
        k80_geo = compute_k80(S_geo)
        U_geo_ctrl = U_geo[:, :k80_geo]

        h_proj = h_array @ U_geo_ctrl @ U_geo_ctrl.T
        h_proj_train, h_proj_test = h_proj[:split], h_proj[split:]

        sub_lin = LinearRegression().fit(h_proj_train, C_train)
        sub_lin_r2 = float(sub_lin.score(h_proj_test, C_test))
        sub_mlp = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=500, random_state=seed)
        try:
            sub_mlp.fit(h_proj_train, C_train)
            sub_mlp_r2 = float(sub_mlp.score(h_proj_test, C_test))
        except Exception:
            sub_mlp_r2 = 0.0

        results["seeds"].append({
            "seed": seed,
            "A1_linear_R2": lin_r2, "A1_mlp_R2": mlp_r2,
            "A2_layer_R2": layer_r2,
            "A3_full_linear_R2": lin_r2, "A3_subspace_linear_R2": sub_lin_r2,
            "A3_subspace_mlp_R2": sub_mlp_r2,
            "k80_geo": k80_geo,
        })

    results["analysis"] = _analyze_a(results)
    return results


def _analyze_a(results):
    lin_r2s = [s["A1_linear_R2"] for s in results["seeds"]]
    mlp_r2s = [s["A1_mlp_R2"] for s in results["seeds"]]
    sub_lin_r2s = [s["A3_subspace_linear_R2"] for s in results["seeds"]]
    sub_mlp_r2s = [s["A3_subspace_mlp_R2"] for s in results["seeds"]]

    lin_mean = float(np.mean(lin_r2s))
    mlp_mean = float(np.mean(mlp_r2s))
    sub_lin_mean = float(np.mean(sub_lin_r2s))
    sub_mlp_mean = float(np.mean(sub_mlp_r2s))

    mlp_vs_lin = mlp_mean / max(lin_mean, 1e-6)
    sub_vs_full = sub_lin_mean / max(lin_mean, 1e-6)

    parts = []
    if mlp_mean > lin_mean * 1.5:
        parts.append(f"MLP R2={mlp_mean:.3f} >> linear R2={lin_mean:.3f} → nonlinear encoding.")
    else:
        parts.append(f"MLP R2={mlp_mean:.3f} ≈ linear R2={lin_mean:.3f} → linear encoding sufficient.")

    if sub_lin_mean < lin_mean * 0.5:
        parts.append(f"Subspace R2={sub_lin_mean:.3f} << full R2={lin_mean:.3f} → control info dispersed beyond geometric subspace.")
    else:
        parts.append(f"Subspace R2={sub_lin_mean:.3f} retains most of full R2={lin_mean:.3f}.")

    return {"linear_R2": lin_mean, "mlp_R2": mlp_mean, "sub_linear_R2": sub_lin_mean,
            "sub_mlp_R2": sub_mlp_mean, "mlp_ratio": mlp_vs_lin, "sub_ratio": sub_vs_full,
            "conclusion": " ".join(parts)}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_a_nonlinear_decoding(n_seeds=8)
    with open("core_mvp_v4/results/a_nonlinear_decoding.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("A:", r["analysis"]["conclusion"])
