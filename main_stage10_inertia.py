"""Inertia sweep — does τ_response shift the window as predicted?"""
import numpy as np

from experiment10 import run_experiment_grid
from analysis_stage10 import (
    extract_R_data, fit_threshold, compute_stability_curve,
)


INERTIA_LABELS = {
    0.0: "fast (no inertia)",
    0.5: "medium",
    0.8: "slow (high inertia)",
}

DRIFTS = [0.005, 0.02, 0.08]
KAPPAS = [1.0, 2.0, 4.0]


def run_one_level(inertia):
    print(f"\n{'#'*60}")
    print(f"# INERTIA = {inertia}  ({INERTIA_LABELS.get(inertia, '')})")
    print(f"{'#'*60}")

    results = run_experiment_grid(
        kappa_list=KAPPAS,
        drift_list=DRIFTS,
        seeds=(42, 43, 44),
        expert_hidden=2,
        K_budget=4,
        gating_hidden=8,
        train_steps=1200,
        test_steps=300,
        inertia=inertia,
    )

    for key, triplet in results.items():
        dr_s, sv, sa = compute_stability_curve({key: triplet})

    drifts, tau_resp, tau_drift, R_values, S_adv_values = extract_R_data(results)
    tau_star, _ = fit_threshold(R_values, S_adv_values, drifts)

    print(f"\n  {'drift':>8} {'tau_resp':>10} {'tau_drift':>10} {'R':>10} {'S_adv':>10}")
    print(f"  {'-'*50}")
    for i in range(len(drifts)):
        print(f"  {drifts[i]:>8.3f} {tau_resp[i]:>10.1f} {tau_drift[i]:>10.1f} "
              f"{R_values[i]:>10.4f} {S_adv_values[i]:>10.4f}")

    print(f"\n  tau_high = {tau_star:.4f}")
    print(f"  R values: {[f'{r:.4f}' for r in R_values]}")
    print(f"  S_adv values: {[f'{s:.4f}' for s in S_adv_values]}")

    peak_idx = int(np.argmax(S_adv_values))
    print(f"  Peak at drift={drifts[peak_idx]}  S_adv={S_adv_values[peak_idx]:.4f}")

    return {
        "inertia": inertia,
        "drifts": drifts,
        "tau_resp": tau_resp,
        "tau_drift": tau_drift,
        "R_values": R_values,
        "S_adv_values": S_adv_values,
        "tau_high": tau_star,
    }


if __name__ == "__main__":
    all_levels = {}

    for inertia in [0.0, 0.5, 0.8]:
        all_levels[inertia] = run_one_level(inertia)

    print("\n" + "=" * 60)
    print("WINDOW SHIFT SUMMARY")
    print("=" * 60)
    print(f"{'inertia':>10} {'tau_resp(0.02)':>16} {'tau_high':>10} {'R(0.02)':>10} {'S_adv(0.02)':>12} {'peak_drift':>10}")
    print("-" * 72)

    theory = "预测: inertia↑ → τ_response↑ → R↑ → 窗口左移"
    for inertia in [0.0, 0.5, 0.8]:
        L = all_levels[inertia]
        peak_idx = int(np.argmax(L["S_adv_values"]))
        print(f"{inertia:>10.1f} {L['tau_resp'][1]:>16.1f} {L['tau_high']:>10.4f} "
              f"{L['R_values'][1]:>10.4f} {L['S_adv_values'][1]:>12.4f} "
              f"{L['drifts'][peak_idx]:>10.3f}")

    print(f"\n{theory}")
    print(f"\n解释: ")
    print(f"  τ_response[0.02] 从 {all_levels[0.0]['tau_resp'][1]:.1f} (fast) "
          f"→ {all_levels[0.8]['tau_resp'][1]:.1f} (slow)")
    print(f"  对于同一 drift=0.02, τ_drift=50 不变, 所以 ")
    print(f"  R(0.02) = τ_response/50 从 {all_levels[0.0]['R_values'][1]:.4f} "
          f"→ {all_levels[0.8]['R_values'][1]:.4f}")
    print(f"  R 增大跨越了 τ_high → S_adv 从 >1 变为 <?")
    print(f"  S_adv(0.02): {all_levels[0.0]['S_adv_values'][1]:.4f} "
          f"→ {all_levels[0.5]['S_adv_values'][1]:.4f} "
          f"→ {all_levels[0.8]['S_adv_values'][1]:.4f}")
