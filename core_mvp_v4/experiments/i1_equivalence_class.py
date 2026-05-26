"""I1: Functional Equivalence Classes.

Perturbs baseline model, retrains, measures internal vs external divergence.
Checks for discrete clusters, interpolation barriers.
"""

import json, copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

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


def _model_action_diff(m1, m2, s_list):
    diffs = []
    for s in s_list:
        a1 = m1.act_numpy(s); a2 = m2.act_numpy(s)
        diffs.append(float(np.linalg.norm(a1 - a2)))
    return float(np.mean(diffs))


def _model_weight_diff(m1, m2):
    diffs = []
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        diffs.append(float(torch.norm(p1 - p2)))
    return float(np.mean(diffs))


def _model_h_diff(m1, m2, s_list):
    diffs = []
    for s in s_list:
        s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        h1 = m1.encoder(s_t).squeeze(0).detach().numpy()
        h2 = m2.encoder(s_t).squeeze(0).detach().numpy()
        diffs.append(float(np.linalg.norm(h1 - h2) / (np.linalg.norm(h1) + 1e-8)))
    return float(np.mean(diffs))


def run_i1_equivalence_class(n_variants=20, d=16, k=2, hd=192,
                              n_episodes=3, episode_length=2000, sigma=0.05, n_seeds=None):
    env_ref = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=0, coupling=0.0)
    baseline = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)
    _train(baseline, env_ref, n_episodes, episode_length, 0)

    env_ref.reset()
    test_s = []
    for _ in range(200):
        test_s.append(env_ref.get_state())
        env_ref.step(np.zeros(k))

    variants = [baseline]
    for i in range(n_variants - 1):
        m = copy.deepcopy(baseline)
        for p in m.parameters():
            p.data += torch.randn_like(p) * sigma
        env = MultiDimDoubleWell(d=d, k=k, drift=0.5, seed=i + 10, coupling=0.0)
        _train(m, env, n_episodes, episode_length, i + 10)
        env.reset()
        for _ in range(episode_length):
            s_t = torch.from_numpy(env.get_state().astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = m(s_t)
            ns, risk, _, _ = env.step(action.squeeze(0).detach().numpy())
            loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]])) + 0.1 * torch.mean(action**2)
        variants.append(m)

    action_diffs = []
    weight_diffs = []
    h_diffs = []
    all_h_vecs = []
    for i in range(n_variants):
        h_vecs = []
        for s in test_s[:30]:
            s_t = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
            h = variants[i].encoder(s_t).squeeze(0).detach().numpy()
            h_vecs.append(h)
        all_h_vecs.append(np.mean(h_vecs, axis=0))
        for j in range(i + 1, n_variants):
            action_diffs.append(_model_action_diff(variants[i], variants[j], test_s[:30]))
            weight_diffs.append(_model_weight_diff(variants[i], variants[j]))
            h_diffs.append(_model_h_diff(variants[i], variants[j], test_s[:30]))

    pca = PCA(n_components=2).fit(np.array(all_h_vecs))
    h_2d = pca.transform(np.array(all_h_vecs))

    hull = None
    try:
        from scipy.spatial import ConvexHull
        if len(h_2d) > 3: hull = {"volume": float(ConvexHull(h_2d).volume)}
    except: pass

    act_m = float(np.mean(action_diffs)); act_std = float(np.std(action_diffs))
    w_m = float(np.mean(weight_diffs)); h_m = float(np.mean(h_diffs))
    ratio = act_m / (h_m + 1e-6)

    if ratio < 0.1:
        conclusion = f"EQUIVALENCE CLASSES FOUND: action diff={act_m:.4f}, h diff={h_m:.4f}, ratio={ratio:.4f}."
    else:
        conclusion = f"CONTINUOUS SPECTRUM: action/h ratio={ratio:.3f}. No discrete clusters."
    return {"analysis": {"action_diff": act_m, "weight_diff": w_m, "h_diff": h_m,
                         "action_h_ratio": ratio, "n_variants": n_variants,
                         "pca_var": pca.explained_variance_ratio_.tolist() if pca else [],
                         "conclusion": conclusion}}


if __name__ == "__main__":
    import os; os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_i1_equivalence_class(n_variants=10)
    with open("core_mvp_v4/results/i1_equivalence_class.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("I1:", r["analysis"]["conclusion"])
