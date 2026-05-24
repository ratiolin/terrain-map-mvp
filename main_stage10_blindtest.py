"""Step 5-6: Blind test — predict from R, then verify with experiment."""
import numpy as np

from experiment10 import run_experiment_grid, run_triplet
from analysis_stage10 import (
    print_matrix, stability_analysis, compute_stability_curve,
    extract_R_data, fit_threshold, interpolate_tau_response, run_R_analysis,
)


if __name__ == "__main__":
    print("=" * 60)
    print("BLIND TEST: R-based prediction of S_adv")
    print("=" * 60)

    print("\n[Phase 1] Run baseline grid (0.005, 0.02, 0.08)")
    all_results = run_experiment_grid(
        kappa_list=[0.5, 1.0, 2.0, 4.0],
        drift_list=[0.005, 0.02, 0.08],
        seeds=(42, 43, 44),
        expert_hidden=2,
        K_budget=4,
        gating_hidden=8,
        train_steps=1200,
        test_steps=300,
    )
    print_matrix(all_results)
    stability_analysis(all_results)

    drifts_known, tau_resp, tau_drift, R_values, S_adv_values, tau_star = run_R_analysis(all_results)

    print("\n" + "=" * 60)
    print("[Phase 2] Blind prediction (no experiment run yet)")
    print("=" * 60)

    blind_drifts = [0.01, 0.035]
    predictions = {}
    for d_blind in blind_drifts:
        tau_r_est = interpolate_tau_response(drifts_known, tau_resp, d_blind)
        tau_d = 1.0 / d_blind
        R_est = tau_r_est / tau_d

        if S_adv_values[-1] < 1.0:
            pred = "S_adv < 1 (collapse)" if R_est > tau_star else "S_adv > 1 (advantage)"
        else:
            pred = "S_adv > 1 (advantage)" if R_est < tau_star else "S_adv < 1 (collapse)"

        predictions[d_blind] = pred
        print(f"  drift={d_blind:.3f}  tau_resp_est={tau_r_est:.1f}  "
              f"tau_drift={tau_d:.1f}  R={R_est:.4f}  "
              f"tau*={tau_star:.4f}  -> {pred}")

    print("\n" + "=" * 60)
    print("[Phase 3] Run blind experiments to verify")
    print("=" * 60)

    blind_results = {}
    for d_blind in blind_drifts:
        print(f"\n--- Blind test: drift={d_blind} ---")
        blind_kappas = [1.0, 2.0]
        for kappa in blind_kappas:
            key = f"k{kappa}_d{d_blind}"
            triplet = run_triplet(
                kappa=kappa, drift=d_blind,
                K_budget=4, expert_hidden=2, gating_hidden=8,
                train_steps=1200, test_steps=300,
                seeds=(42, 43, 44),
            )
            _, _, S_adv_values_blind = compute_stability_curve({key: triplet})
            if key not in blind_results:
                blind_results[key] = {}
            blind_results[key]["S_adv"] = S_adv_values_blind[0]
            print(f"  kappa={kappa:.1f}  S_adv={S_adv_values_blind[0]:.4f}")

    print("\n" + "=" * 60)
    print("[Phase 4] Verification")
    print("=" * 60)

    for d_blind in blind_drifts:
        s_adv_vals = []
        for kappa in [1.0, 2.0]:
            key = f"k{kappa}_d{d_blind}"
            if key in blind_results:
                s_adv_vals.append(blind_results[key]["S_adv"])
        s_adv_mean = float(np.mean(s_adv_vals)) if s_adv_vals else 0.0
        actual = "S_adv > 1" if s_adv_mean > 1.0 else "S_adv < 1"
        match = (s_adv_mean > 1.0) == ('advantage' in predictions[d_blind])
        print(f"  drift={d_blind:.3f}  predicted: {predictions[d_blind]}  "
              f"actual S_adv={s_adv_mean:.4f} ({actual})  "
              f"{'MATCH' if match else 'MISMATCH'}")
