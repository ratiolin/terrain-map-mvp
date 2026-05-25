#!/usr/bin/env python
import json
import numpy as np

from core_mvp_v2.run_mvp import (run_mpv, run_lambda_scan, run_band_scan,
                                   run_geq, run_perturbation, run_multistability)


def main():
    print("=" * 60)
    print("TERRAIN MAP MVP — Structure Emergence System")
    print("=" * 60)

    results = {}

    print("\n--- 1. Baseline ---")
    r, s = run_mpv(seed=0, regime_visible=True)
    results["baseline"] = {"S_adv": r["S_adv"], "structure": r["structure"]}

    print("\n--- 2. Structure Gain Scan ---")
    beta_results = run_lambda_scan(eta=0.5, regime_visible=True)
    results["beta_scan"] = [{
        "beta": 0.0, "S_adv": beta_results[0]["S_adv"],
        "gain": beta_results[0]["structure"]["structure_gain_posthoc"]
    }]

    print("\n--- 3. Band Scan ---")
    band, g, S, ent = run_band_scan(eta=0.5, n_points=16, structure_beta=0.0,
                                      regime_visible=True)
    results["band_scan"] = {
        "g_axis": g.tolist(),
        "S_adv_curve": S.tolist(),
        "entropy_curve": ent.tolist(),
    }

    print("\n--- 4. g-Equivalence ---")
    geq = run_geq(g_target=1.0, structure_beta=0.0, regime_visible=True)
    results["g_equivalence"] = geq

    print("\n--- 5. Perturbation Test ---")
    pert = run_perturbation(g=0.5, structure_beta=0.0, regime_visible=True)
    S_advs = [r["S_adv"] for _, r in pert]
    results["perturbation"] = {
        "n_tests": len(pert),
        "S_adv_mean": float(np.mean(S_advs)),
        "S_adv_std": float(np.std(S_advs)),
    }

    with open("results_final/run_output.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to results_final/run_output.json")
    print("Done.")


if __name__ == "__main__":
    main()
