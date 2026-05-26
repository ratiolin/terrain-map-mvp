"""Tier 1-1: Jacobian Spectrum — ∂a/∂h low-dimensional structure.

Computes SVD of J = ∂a/∂h across states, analyzes spectrum,
cross-time stability, and principal angles with ∂h/∂s subspace.
"""

import json
import numpy as np
import torch
import torch.nn as nn

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, compute_jacobian
from core_mvp_v4.metrics import compute_k80, effective_rank, alignment


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


def _compute_da_dh(model, s, k):
    s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
    s_t.requires_grad = True
    h_t = model.encoder(s_t)
    a_t = model.actor(h_t)
    J = np.zeros((k, model.hidden_dim))
    for j in range(k):
        grad = torch.autograd.grad(a_t[0, j], h_t, retain_graph=True,
                                   create_graph=False, allow_unused=True)[0]
        if grad is not None:
            J[j] = grad.squeeze(0).detach().numpy()
    return J


def run_t1_jacobian_spectrum(n_seeds=8, d=16, k=2, hd=192,
                             n_episodes=3, episode_length=2000, n_states=500):
    results = {"seeds": []}

    for seed in range(n_seeds):
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
        _train(model, env, n_episodes, episode_length, seed)

        env.reset()
        J_list = []
        J_geo_list = []
        for _ in range(n_states):
            s = env.get_state()
            J = _compute_da_dh(model, s, k)
            J_list.append(J)
            J_geo = compute_jacobian(model, s)
            J_geo_list.append(J_geo)
            a = model.act_numpy(s)
            env.step(a)

        k80s = []
        eff_ranks = []
        Vt_list = []
        Vt_geo_list = []
        S_spectra = []

        for J in J_list:
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            k80s.append(compute_k80(S))
            eff_ranks.append(effective_rank(S))
            Vt_list.append(Vt)
            S_spectra.append(S)

        for Jg in J_geo_list:
            _, Sg, Vtg = np.linalg.svd(Jg, full_matrices=False)
            Vt_geo_list.append(Vtg)

        cross_align = []
        for i in range(0, len(Vt_list) - 1, max(1, len(Vt_list) // 100)):
            for j in range(i + 1, min(i + 10, len(Vt_list))):
                k_use = min(k80s[i], k80s[j], k)
                if k_use > 0:
                    V1 = Vt_list[i].T[:, :k_use]
                    V2 = Vt_list[j].T[:, :k_use]
                    cross_align.append(alignment(V1, V2, k=k_use))

        principal_angles_da_dh_vs_dh_ds = []
        for i in range(0, len(Vt_list), max(1, len(Vt_list) // 50)):
            kda = min(k80s[i], k)
            if kda > 0 and i < len(Vt_geo_list):
                _, Sg, Vtg = np.linalg.svd(J_geo_list[i], full_matrices=False)
                kg = min(compute_k80(Sg), k)
                if kg > 0:
                    V_da = Vt_list[i].T[:, :kda]
                    V_dh = Vtg.T[:, :kg]
                    k_align = min(kda, kg, V_da.shape[0], V_dh.shape[0])
                    if k_align > 0:
                        pa = 1.0 - alignment(V_da[:k_align, :k_align].T if V_da.shape[0] >= k_align else V_da,
                                            V_dh[:k_align, :k_align].T if V_dh.shape[0] >= k_align else V_dh,
                                            k=k_align)
                        principal_angles_da_dh_vs_dh_ds.append(float(pa))

        results["seeds"].append({
            "seed": seed,
            "k80_mean": float(np.mean(k80s)), "k80_std": float(np.std(k80s)),
            "eff_rank_mean": float(np.mean(eff_ranks)), "eff_rank_std": float(np.std(eff_ranks)),
            "cross_time_alignment": float(np.mean(cross_align)) if cross_align else 0.0,
            "cross_time_alignment_std": float(np.std(cross_align)) if cross_align else 0.0,
            "principal_angle_da_vs_dh": float(np.mean(principal_angles_da_dh_vs_dh_ds)) if principal_angles_da_dh_vs_dh_ds else 0.0,
            "avg_S_norm": np.mean(S_spectra, axis=0).tolist() if S_spectra else [],
        })

    analysis = _analyze(results)
    results["analysis"] = analysis
    return results


def _analyze(results):
    k80s = [s["k80_mean"] for s in results["seeds"]]
    effs = [s["eff_rank_mean"] for s in results["seeds"]]
    aligns = [s["cross_time_alignment"] for s in results["seeds"]]
    angles = [s["principal_angle_da_vs_dh"] for s in results["seeds"]]

    k80_m = float(np.mean(k80s))
    eff_m = float(np.mean(effs))
    align_m = float(np.mean(aligns))
    angle_m = float(np.mean(angles))

    parts = []
    if k80_m <= 3:
        parts.append(f"LOW-DIM: k80={k80_m:.1f}≤3 → control is low-rank Jacobian structure.")
    else:
        parts.append(f"HIGH-DIM: k80={k80_m:.1f}>3 → control not strongly low-rank.")

    if align_m > 0.8:
        parts.append(f"Cross-time alignment={align_m:.3f}>0.8 → stable structure.")
    else:
        parts.append(f"Cross-time alignment={align_m:.3f} → moderate stability.")

    if angle_m > 0.7:
        parts.append(f"∂a/∂h ⟂ ∂h/∂s (angle={angle_m:.3f}) → control and representation decoupled.")
    else:
        parts.append(f"∂a/∂h aligned with ∂h/∂s (angle={angle_m:.3f}) → coupled.")

    return {"k80": k80_m, "eff_rank": eff_m, "cross_time_alignment": align_m,
            "da_vs_dh_angle": angle_m, "conclusion": " ".join(parts)}


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_t1_jacobian_spectrum(n_seeds=8)
    with open("core_mvp_v4/results/t1_jacobian_spectrum.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("T1-1:", r["analysis"]["conclusion"])
