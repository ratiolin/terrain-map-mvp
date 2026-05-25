import json
import pickle
import numpy as np

from core_mvp_v2.run_mvp import (run_mpv, run_lambda_scan, run_band_scan,
                                   run_geq, run_perturbation, run_multistability)


def produce_all(regime_visible=True):
    print("=" * 60)
    print(f"CORE MVP V2 — STRUCTURE GAIN REWARD (regime_visible={regime_visible})")
    print("=" * 60)

    # Step 1: beta scan
    beta_results = run_lambda_scan(eta=0.5, regime_visible=regime_visible)

    # Step 2: key test — β=0 vs β>0
    print(f"\n--- Band scan (β=0, no reward) ---")
    band0, g0, S0, _ = run_band_scan(eta=0.5, n_points=21, structure_beta=0.0,
                                       regime_visible=regime_visible)

    print(f"\n--- Band scan (β=1.0, with reward) ---")
    band1, g1, S1, _ = run_band_scan(eta=0.5, n_points=21, structure_beta=1.0,
                                       regime_visible=regime_visible)

    # g-equivalence at β=0
    print(f"\n--- G-equivalence (β=0) ---")
    geq0 = run_geq(g_target=1.0, structure_beta=0.0, regime_visible=regime_visible)

    # perturbation at β=0
    pert0 = run_perturbation(g=0.5, structure_beta=0.0, regime_visible=regime_visible)

    # multi-stability at β=0
    multi_g = np.linspace(0.0, 3.0, 7)
    multi0 = run_multistability(multi_g, structure_beta=0.0, regime_visible=regime_visible)

    # Judgment
    struct_types_b0 = [r["structure_type"] for r in band0]
    n_multi_b0 = sum(1 for t in struct_types_b0 if t == "multi")
    n_pattern_b0 = sum(1 for t in struct_types_b0 if t == "pattern")
    n_single_b0 = sum(1 for t in struct_types_b0 if t == "single")

    struct_types_b1 = [r["structure_type"] for r in band1]
    n_multi_b1 = sum(1 for t in struct_types_b1 if t == "multi")

    multi_only_with_reward = (n_multi_b0 == 0 and n_multi_b1 > 0)
    multi_exists_without_reward = (n_multi_b0 > 0)

    if multi_exists_without_reward:
        verdict = "GENUINE STRUCTURE — multi-expert exists at β=0"
    elif multi_only_with_reward:
        verdict = "ARTIFICIAL STRUCTURE — multi-expert only at β>0"
    else:
        verdict = "NO STRUCTURE — neither β=0 nor β>0 produces multi-expert"

    print(f"\n{'='*60}")
    print(f"JUDGMENT")
    print(f"{'='*60}")
    print(f"  β=0: multi={n_multi_b0}, pattern={n_pattern_b0}, single={n_single_b0}")
    print(f"  β>0: multi={n_multi_b1}")
    print(f"  Verdict: {verdict}")

    # Save
    phase_diagram = {
        "beta_scan": [{k: r[k] for k in ["S_adv", "structure"]} for r in beta_results],
        "beta_zero_scan": {
             "g_axis": g0.tolist(),
             "S_adv_curve": S0.tolist(),
             "structure_types": struct_types_b0,
        },
        "beta_one_scan": {
             "g_axis": g1.tolist(),
             "S_adv_curve": S1.tolist(),
             "structure_types": struct_types_b1,
        },
        "geq_beta0": geq0,
        "multi_stability_beta0": {str(k): v for k, v in multi0.items()},
        "judgment": {
            "n_multi_beta0": n_multi_b0,
            "n_pattern_beta0": n_pattern_b0,
            "n_single_beta0": n_single_b0,
            "n_multi_beta1": n_multi_b1,
            "verdict": verdict,
        },
    }

    with open("core_mvp_v2/results/phase_diagram.json", "w") as f:
        json.dump(phase_diagram, f, indent=2, default=str)

    raw_logs = {
        "beta_scan": beta_results,
        "band_scan_beta0": band0,
        "band_scan_beta1": band1,
        "geq_beta0": geq0,
        "perturbation_beta0": [(t, r) for t, r in pert0],
        "multi_stability_beta0": multi0,
    }

    with open("core_mvp_v2/results/structure_logs.pkl", "wb") as f:
        pickle.dump(raw_logs, f)

    print(f"\nSaved: core_mvp_v2/results/phase_diagram.json")
    print(f"Saved: core_mvp_v2/results/structure_logs.pkl")


if __name__ == "__main__":
    produce_all()
