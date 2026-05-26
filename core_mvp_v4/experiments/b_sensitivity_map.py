"""Direction B: Representation Perturbation Sensitivity Map.

Finds which directions in hidden space most affect behavioral output.

B1: Random direction perturbation (baseline sensitivity).
B2: PCA principal component direction perturbation.
B3: Jacobian ∂a/∂h singular vector perturbation.
B4: Principal angles between behavior-sensitive directions and geometric subspace.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

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


def _action_diff(model, env, perturbation_fn, n_samples=50, eps=0.1):
    env.reset()
    diffs = []
    for _ in range(n_samples):
        s = env.get_state()
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h_orig = model.encoder(s_t).squeeze(0).detach().numpy()
        a_orig = model.actor(torch.from_numpy(h_orig.astype(np.float32)).unsqueeze(0))
        a_orig = a_orig.squeeze(0).detach().numpy()

        h_noisy = perturbation_fn(h_orig, eps)
        a_noisy = model.actor(torch.from_numpy(h_noisy.astype(np.float32)).unsqueeze(0))
        a_noisy = a_noisy.squeeze(0).detach().numpy()

        diffs.append(float(np.linalg.norm(a_noisy - a_orig)))

        ns, _, _, _ = env.step(a_orig)
    return float(np.mean(diffs))


def run_b_sensitivity_map(n_seeds=8, d=16, k=2, hd=192,
                          n_episodes=3, episode_length=2000):
    eps_list = [0.01, 0.05, 0.1, 0.2]
    n_pc = 5
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        # Collect hidden states for PCA
        env.reset()
        h_collected = []
        for _ in range(300):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = model.encoder(s_t).squeeze(0).detach().numpy()
            h_collected.append(h)
            a = model.act_numpy(s)
            env.step(a)
        h_array = np.array(h_collected)
        pca = PCA(n_components=n_pc)
        pca.fit(h_array)
        pc_components = pca.components_.T

        # B1: random direction sensitivity
        b1_random = {}
        for eps in eps_list:
            def rand_perturb(h, e=eps):
                u = np.random.randn(len(h))
                u /= np.linalg.norm(u) + 1e-8
                return h + e * u
            b1_random[f"eps_{eps}"] = _action_diff(model, env, rand_perturb, n_samples=50, eps=eps)

        # B2: PCA direction sensitivity
        b2_pca = {}
        for i in range(n_pc):
            pc_vec = pc_components[:, i]
            def pc_perturb(h, v=pc_vec, e=0.1):
                return h + e * v
            b2_pca[f"PC{i+1}"] = _action_diff(model, env, pc_perturb, n_samples=50, eps=0.1)

        # B2-baseline: random directions of same dimensionality
        rand_dirs = []
        for i in range(n_pc):
            u = np.random.randn(hd)
            u /= np.linalg.norm(u) + 1e-8
            rand_dirs.append(u)
        b2_random_dir = {}
        for i, rd in enumerate(rand_dirs):
            def rd_perturb(h, v=rd, e=0.1):
                return h + e * v
            b2_random_dir[f"rand{i+1}"] = _action_diff(model, env, rd_perturb, n_samples=50, eps=0.1)

        # B3: Jacobian ∂a/∂h sensitivity
        env.reset()
        dh_a = []
        for _ in range(50):
            s = env.get_state()
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            s_t.requires_grad = True
            h_t = model.encoder(s_t)
            a_t = model.actor(h_t)
            jac = []
            for j in range(k):
                grad = torch.autograd.grad(a_t[0, j], h_t, retain_graph=True,
                                           create_graph=False, allow_unused=True)[0]
                jac.append(grad.squeeze(0).detach().numpy())
            dh_a.append(np.mean(jac, axis=0))
            env.step(model.act_numpy(s))
        dh_a_mean = np.mean(dh_a, axis=0)
        dh_a_norm = dh_a_mean / (np.linalg.norm(dh_a_mean) + 1e-8)
        jac_sensitivity = _action_diff(model, env,
                                       lambda h, e=0.1: h + e * dh_a_norm,
                                       n_samples=50, eps=0.1)

        # B4: alignment between behavior-sensitive dirs and geometric subspace
        J_geo = np.mean([compute_jacobian(model, env.get_state()) for _ in range(30)], axis=0)
        for _ in range(30):
            a = model.act_numpy(env.get_state())
            env.step(a)
            J_geo += compute_jacobian(model, env.get_state())
        J_geo /= 31
        U_geo, S_geo, Vt_geo = np.linalg.svd(J_geo, full_matrices=False)
        k80_geo = compute_k80(S_geo)
        U_geo_ctrl = U_geo[:, :k80_geo]

        behavior_sensitive = pc_components[:k, :]
        k_use = min(k, U_geo_ctrl.shape[1])
        if k_use > 0 and behavior_sensitive.shape[1] == U_geo_ctrl.shape[0]:
            principal_angle = float(np.abs(
                np.linalg.norm(behavior_sensitive @ U_geo_ctrl[:, :k_use]) / np.sqrt(k_use)
            ))
        else:
            principal_angle = 0.0

        results["seeds"].append({
            "seed": seed,
            "B1_random": b1_random,
            "B2_pca_sensitivity": b2_pca,
            "B2_random_dir_sensitivity": b2_random_dir,
            "B3_jacobian_sensitivity": jac_sensitivity,
            "B4_principal_angle": principal_angle,
            "k80_geo": k80_geo,
        })

    results["analysis"] = _analyze_b(results)
    return results


def _analyze_b(results):
    pc_sens = np.mean([np.mean(list(s["B2_pca_sensitivity"].values()))
                       for s in results["seeds"]])
    rand_sens = np.mean([np.mean(list(s["B2_random_dir_sensitivity"].values()))
                         for s in results["seeds"]])
    jac_sens = np.mean([s["B3_jacobian_sensitivity"] for s in results["seeds"]])
    random_base = np.mean([s["B1_random"]["eps_0.1"] for s in results["seeds"]])
    p_align = np.mean([s["B4_principal_angle"] for s in results["seeds"]])
    pc_vs_rand = "PC > random" if pc_sens > rand_sens * 1.2 else "PC ≈ random"
    jac_vs_pc = "Jacobian > PC" if jac_sens > pc_sens * 1.2 else "Jacobian ≈ PC"

    if p_align > 0.7:
        angle_desc = "BEHAVIOR AND GEOMETRY ALIGNED (align > 0.7)"
    elif p_align < 0.3:
        angle_desc = "BEHAVIOR AND GEOMETRY ORTHOGONAL (align < 0.3)"
    else:
        angle_desc = f"MODERATE ALIGNMENT (align ≈ {p_align:.3f})"

    conclusion = f"{angle_desc}. {pc_vs_rand}. {jac_vs_pc}."
    return {
        "pc_vs_random": {"pc_sens": float(pc_sens), "rand_sens": float(rand_sens)},
        "jac_sens": float(jac_sens), "random_base": float(random_base),
        "behavior_geom_alignment": float(p_align),
        "conclusion": conclusion,
    }


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_b_sensitivity_map(n_seeds=8)
    with open("core_mvp_v4/results/b_sensitivity_map.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("B:", r["analysis"]["conclusion"])
