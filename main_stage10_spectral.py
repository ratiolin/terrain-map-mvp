import json
import numpy as np

from experiment10 import run_experiment_spectral_grid
from analysis_stage10 import judge, print_matrix


def save_drift_data(all_results, output_dir="spectral_data"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for key, triplet in all_results.items():
        k, w = key.replace("k", "").split("_w")
        for tag in ["det", "rand_noctx", "rand_ctx"]:
            if tag not in triplet:
                continue
            seeds = triplet[tag].get("seeds", [])
            for si, seed_result in enumerate(seeds):
                dh = seed_result.get("drift_history", [])
                if dh:
                    tag_clean = tag.replace("_", "-")
                    fname = f"{output_dir}/drift_k{k}_w{w}_{tag_clean}_s{si}.npy"
                    np.save(fname, np.array(dh))
                ph = seed_result.get("pred_history", [])
                th = seed_result.get("targ_history", [])
                if ph and th:
                    tag_clean = tag.replace("_", "-")
                    error = np.array(ph) - np.array(th)
                    np.save(f"{output_dir}/error_k{k}_w{w}_{tag_clean}_s{si}.npy", error)

    print(f"\nDrift time series saved to {output_dir}/")


def collect_performance(all_results):
    results = []
    for key, triplet in all_results.items():
        k, w = key.replace("k", "").split("_w")
        kappa = float(k)
        omega = float(w)
        det = triplet.get("det", {})
        perf = det.get("mse_gap_mean", 0.0)
        test_large = det.get("test_large_mean", 0.0)
        oracle = det.get("oracle_mse_mean", 0.0)
        routing_gap = det.get("routing_gap_mean", 0.0)
        results.append({
            "kappa": kappa,
            "omega": omega,
            "mse_gap_mean": perf,
            "test_large_mean": test_large,
            "oracle_mse_mean": oracle,
            "routing_gap_mean": routing_gap,
            "verdict": judge(triplet),
        })
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 10 SPECTRAL: Drift Spectrum Analysis")
    print("=" * 60)

    omega_list = np.linspace(0.01, 1.0, 6)
    all_results = run_experiment_spectral_grid(
        kappa_list=[1.0, 2.0],
        omega_list=omega_list,
        seeds=(42, 43, 44),
        expert_hidden=2,
        K_budget=4,
        gating_hidden=8,
        train_steps=1200,
        test_steps=300,
        inertia=0.0,
    )

    print_matrix(all_results)

    save_drift_data(all_results)

    performance_data = collect_performance(all_results)
    with open("spectral_performance.json", "w") as f:
        json.dump(performance_data, f, indent=2)
    print("\nPerformance data saved to spectral_performance.json")

    print()
    print("=" * 60)
    print("STAGE 10 SPECTRAL COMPLETE")
