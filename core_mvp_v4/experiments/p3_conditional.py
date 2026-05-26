"""P3-Conditional: Conditional Trigger Drift (Part 2 revision).

Precondition: S0 passed, P1-Refocus identified semantic-preserving d ceiling.

Trains in high-controllability regime (low g), then switches to
low-controllability regime (high g). Observes subspace rotation.

Measures:
- alignment_gt before/after switch
- R² before/after switch
- Principal angle change time series
"""

import json
import numpy as np
from sklearn.linear_model import LinearRegression

from core_mvp_v4.env import MultiDimDoubleWell
from core_mvp_v4.models import V4Model, train_with_signal_loss, collect_controllability_data, compute_jacobian
from core_mvp_v4.metrics import compute_k80, R2_probe, alignment


def run_p3_conditional(n_seeds=8, d=4, k=2, sigma_obs=0.0,
                       hd=16, n_episodes=3, episode_length=2000, lambda_signal=0.5,
                       g_low=0.3, g_high=1.5):
    results = {"seeds": []}

    for seed in range(n_seeds):
        seed_data = {}

        env = MultiDimDoubleWell(d=d, k=k, drift=g_low, seed=seed, coupling=0.0)
        model = V4Model(state_dim=d, hidden_dim=hd, action_dim=k)

        train_with_signal_loss(
            model, env,
            num_episodes=n_episodes, episode_length=episode_length,
            lambda_signal=lambda_signal, seed=seed,
        )

        env.reset()
        pre_h, pre_C = collect_controllability_data(
            model, env, n_samples=200, sigma_obs=sigma_obs,
        )
        pre_probe = LinearRegression().fit(pre_h, pre_C)
        pre_R2 = float(pre_probe.score(pre_h, pre_C))
        seed_data["pre_switch_R2"] = pre_R2

        pre_J = np.mean([compute_jacobian(model, env.get_state()) for _ in range(50)], axis=0)
        _, _, Vt_pre = np.linalg.svd(pre_J, full_matrices=False)
        V_pre = Vt_pre.T
        k_pre = min(k, V_pre.shape[1])
        U_true = np.eye(d)[:, :k]
        pre_alignment = alignment(V_pre[:, :k_pre], U_true[:, :k_pre], k=k_pre)
        seed_data["pre_switch_alignment_gt"] = float(pre_alignment)

        env.set_drift(g_high)
        env.reset()

        post_h, post_C = collect_controllability_data(
            model, env, n_samples=200, sigma_obs=sigma_obs,
        )
        post_probe = LinearRegression().fit(post_h, post_C)
        post_R2 = float(post_probe.score(post_h, post_C))
        seed_data["post_switch_R2"] = post_R2

        post_J = np.mean([compute_jacobian(model, env.get_state()) for _ in range(50)], axis=0)
        _, _, Vt_post = np.linalg.svd(post_J, full_matrices=False)
        V_post = Vt_post.T
        k_post = min(k, V_post.shape[1])
        post_alignment = alignment(V_post[:, :k_post], U_true[:, :k_post], k=k_post)
        seed_data["post_switch_alignment_gt"] = float(post_alignment)

        angle_changes = []
        prev_V = V_pre
        env.set_drift(g_high)
        env.reset()
        for t_step in range(200):
            if t_step % 10 == 0:
                J = compute_jacobian(model, env.get_state())
                _, _, Vt = np.linalg.svd(J, full_matrices=False)
                V_curr = Vt.T
                k_align = min(k, V_curr.shape[1], prev_V.shape[1])
                angle = 1.0 - alignment(V_curr[:, :k_align], prev_V[:, :k_align], k=k_align)
                angle_changes.append(float(angle))
                prev_V = V_curr
            a = model.act_numpy(env.get_state())
            env.step(a)

        seed_data["angle_changes"] = angle_changes
        seed_data["angle_change_mean"] = float(np.mean(angle_changes)) if angle_changes else 0.0
        seed_data["angle_change_max"] = float(np.max(angle_changes)) if angle_changes else 0.0

        results["seeds"].append(seed_data)

    pre_aligns = [s["pre_switch_alignment_gt"] for s in results["seeds"]]
    post_aligns = [s["post_switch_alignment_gt"] for s in results["seeds"]]
    pre_r2s = [s["pre_switch_R2"] for s in results["seeds"]]
    post_r2s = [s["post_switch_R2"] for s in results["seeds"]]

    results["aggregate"] = {
        "pre_alignment": f"{float(np.mean(pre_aligns)):.4f} ± {float(np.std(pre_aligns)):.4f}",
        "post_alignment": f"{float(np.mean(post_aligns)):.4f} ± {float(np.std(post_aligns)):.4f}",
        "pre_R2": f"{float(np.mean(pre_r2s)):.4f} ± {float(np.std(pre_r2s)):.4f}",
        "post_R2": f"{float(np.mean(post_r2s)):.4f} ± {float(np.std(post_r2s)):.4f}",
    }

    results["conclusion"] = _conclude(results)
    return results


def _conclude(results):
    pre_aligns = [s["pre_switch_alignment_gt"] for s in results["seeds"]]
    post_aligns = [s["post_switch_alignment_gt"] for s in results["seeds"]]
    pre_R2s = [s["pre_switch_R2"] for s in results["seeds"]]
    post_R2s = [s["post_switch_R2"] for s in results["seeds"]]

    align_drop = np.mean(pre_aligns) - np.mean(post_aligns)
    R2_drop = np.mean(pre_R2s) - np.mean(post_R2s)
    angle_mag = float(np.mean([s["angle_change_mean"] for s in results["seeds"]]))

    if align_drop > 0.15:
        return (
            f"CONDITIONAL TRIGGER CONFIRMED: alignment_gt dropped by {align_drop:.3f}, "
            f"R2 dropped by {R2_drop:.3f}, principal angle change {angle_mag:.3f}. "
            "Conditional change triggers subspace reorganization."
        )
    elif abs(align_drop) < 0.05:
        return (
            f"NO TRIGGER: subspace stable (|delta_align|={abs(align_drop):.3f}<0.05). "
            "Subspace is condition-invariant or g difference insufficient to trigger change."
        )
    elif align_drop < 0:
        return (
            f"REVERSE: alignment improved after switch ({-align_drop:.3f}). "
            "High-g regime may induce stronger structural constraint on the subspace."
        )
    else:
        return (
            f"WEAK TRIGGER: moderate alignment drop {align_drop:.3f}, "
            f"R2 drop {R2_drop:.3f}. Partial reorganization observed."
        )


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_p3_conditional(n_seeds=8)
    with open("core_mvp_v4/results/p3_conditional_trigger.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    agg = r.get("aggregate", {})
    print(f"P3-Conditional: pre_align={agg.get('pre_alignment')}, post_align={agg.get('post_alignment')}")
    print(f"  {r.get('conclusion')}")
