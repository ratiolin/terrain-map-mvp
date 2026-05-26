"""Phase Transition Order + Critical Dimension Validation.

Exp1: 101-point α sweep at dim=32 with derivative analysis to classify
      first-order (jump) vs second-order (continuous) transitions.
Exp3: Dimension scan [8,16,24,32,48,64,96] with criticality score.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def train_and_analyze(alpha, hidden_dim=32, T=2500, N=1200, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    schedule = [(625, (0.1, 0.3)), (625, (1.0, 2.0)),
                (625, (0.1, 0.3)), (625, (1.0, 2.0))]
    env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
    pol_A = PolicyNetwork(hidden_dim=hidden_dim)
    pol_B = PolicyNetwork(hidden_dim=hidden_dim)
    params = list(pol_A.parameters()) + list(pol_B.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    enc_A = list(pol_A.backbone.parameters())
    enc_B = list(pol_B.backbone.parameters())
    env.reset()
    for _ in range(T):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        aA, aB = float(pol_A(x)[0].item()), float(pol_B(x)[0].item())
        env.step(aA, aB)
        loss = torch.tensor(abs(float(env.state[0])), requires_grad=True)
        opt.zero_grad()
        loss.backward(retain_graph=True)
        for pA, pB in zip(enc_A, enc_B):
            if pA.grad is not None and pB.grad is not None:
                gA, gB = pA.grad.clone(), pB.grad.clone()
                mix = alpha * 0.5 * (gA + gB)
                pA.grad = (1.0 - alpha) * gA + mix
                pB.grad = (1.0 - alpha) * gB + mix
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

    H, a_list, b_list = [], [], []
    env.reset()
    for _ in range(N):
        x = torch.tensor(env.state.copy(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a_s_A, _, h = pol_A(x); a_s_B, _, _ = pol_B(x)
        env.step(float(a_s_A.item()), float(a_s_B.item()))
        H.append(h.numpy().flatten())
        a_list.append(float(a_s_A.item())); b_list.append(float(a_s_B.item()))

    Hnp = np.array(H)
    WA = pol_A.head_shape.weight.detach().numpy().flatten()
    WB = pol_B.head_shape.weight.detach().numpy().flatten()
    _, S, Vt = np.linalg.svd(Hnp, full_matrices=False)
    Z = Hnp @ Vt.T
    wA, wB = WA @ Vt.T, WB @ Vt.T
    contribs = np.array([wA[i] * wB[i] * float(np.var(Z[:, i])) for i in range(len(S))])
    p = np.abs(contribs) / (np.abs(contribs).sum() + 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    corr = float(np.corrcoef(np.array(a_list), np.array(b_list))[0, 1])
    top_energy = float(np.max(np.abs(contribs)) / (np.abs(contribs).sum() + 1e-12))
    k = int(np.argmax(np.abs(contribs)))
    sign_dom = int(np.sign(wA[k] * wB[k]))

    return {"alpha": float(alpha), "entropy": entropy, "corr": corr,
            "top_pc_energy": round(top_energy, 4), "sign_dominant": sign_dom,
            "pc1_var": round(float(S[0]**2/(S**2).sum()), 3)}


def compute_criticality(sweep, collapse_thresh=0.01):
    H_arr = np.array([r["entropy"] for r in sweep])
    C_arr = np.array([r["corr"] for r in sweep])
    n_collapses = int((H_arr < collapse_thresh).sum())
    sign_changes = 0
    for i in range(len(C_arr) - 1):
        if C_arr[i] * C_arr[i+1] < 0:
            sign_changes += 1
    entropy_range = float(H_arr.max() - H_arr.min())
    score = n_collapses + sign_changes + entropy_range
    return n_collapses, sign_changes, entropy_range, round(score, 2)


def main():
    print("=" * 60)
    print("  PHASE TRANSITION ANALYSIS")
    print("=" * 60)

    # ===== Exp1: High-resolution α sweep at dim=32 =====
    print("\n  Exp1: 101-α sweep (dim=32)")
    alphas_101 = np.linspace(0, 1, 101)
    sweep_32 = []
    for i, alpha in enumerate(alphas_101):
        sweep_32.append(train_and_analyze(alpha, hidden_dim=32, T=2500, N=1200))
        if i % 10 == 0:
            r = sweep_32[-1]
            print(f"    α={alpha:.3f}: H={r['entropy']:.4f} corr={r['corr']:+.4f} "
                  f"top={r['top_pc_energy']:.3f} sign={r['sign_dominant']:+d}")

    H32 = np.array([r["entropy"] for r in sweep_32])
    C32 = np.array([r["corr"] for r in sweep_32])
    dH = np.gradient(H32, alphas_101)
    dC = np.gradient(C32, alphas_101)

    dH_peaks = np.where(np.abs(dH) > np.percentile(np.abs(dH), 98))[0]
    dC_peaks = np.where(np.abs(dC) > np.percentile(np.abs(dC), 98))[0]

    print(f"    dH spike locations (α): {alphas_101[dH_peaks].round(3).tolist()}")
    print(f"    dC spike locations (α): {alphas_101[dC_peaks].round(3).tolist()}")

    collapse_regions = []
    for i in range(len(H32)):
        if H32[i] < 0.01:
            collapse_regions.append(round(float(alphas_101[i]), 3))
    print(f"    collapse α: {collapse_regions}")

    sign_flip_αs = []
    for i in range(len(C32) - 1):
        if C32[i] * C32[i+1] < 0:
            sign_flip_αs.append(round(float(alphas_101[i]), 3))
    print(f"    sign flips at α: {sign_flip_αs}")

    results = {
        "exp1_dim32_101": {
            "sweep": [{"alpha": r["alpha"], "entropy": round(r["entropy"], 4),
                       "corr": round(r["corr"], 4),
                       "top_pc_energy": r["top_pc_energy"],
                       "sign_dominant": r["sign_dominant"]} for r in sweep_32],
            "derivatives": {
                "dH_max_alpha": alphas_101[dH_peaks].round(3).tolist() if len(dH_peaks) else [],
                "dC_max_alpha": alphas_101[dC_peaks].round(3).tolist() if len(dC_peaks) else [],
            },
            "collapse_alphas": collapse_regions,
            "sign_flip_alphas": sign_flip_αs,
        }
    }

    # ===== Exp3: Dimension scan =====
    print("\n  Exp3: dimension scan [8,16,24,32,48,64,96]")
    dims = [8, 16, 24, 32, 48, 64, 96]
    alphas_51 = np.linspace(0, 1, 51)
    dim_sweeps = {}

    for dim in dims:
        sweep = []
        for alpha in alphas_51:
            sweep.append(train_and_analyze(alpha, hidden_dim=dim, T=2500, N=1200))
        dim_sweeps[f"dim_{dim}"] = sweep
        nc, ns, er, score = compute_criticality(sweep)
        print(f"    dim={dim:3d}: collapses={nc} sign_flips={ns} "
              f"entropy_range={er:.2f} criticality={score:.2f}")
        results[f"dim_{dim}_sweep"] = {
            "dim": dim, "n_collapses": nc, "n_sign_flips": ns,
            "entropy_range": round(er, 2), "criticality_score": score,
        }

    out = Path("results_final/phase_transition_analysis.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out}")

    # Summary
    scores = [results[f"dim_{d}_sweep"]["criticality_score"] for d in dims]
    peak_dim = dims[np.argmax(scores)]
    print(f"\n  criticality peak at dim={peak_dim} (score={max(scores):.2f})")
    print(f"  dims sorted by criticality: "
          f"{[d for _, d in sorted(zip(scores, dims), reverse=True)]}")


if __name__ == "__main__":
    main()
