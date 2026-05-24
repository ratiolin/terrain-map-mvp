import json
import os
import numpy as np

from experiment10 import run_experiment_memory_sweep
from analysis_stage10 import judge, print_matrix


def save_data(all_results, output_dir="memory_data"):
    os.makedirs(output_dir, exist_ok=True)
    for key, triplet in all_results.items():
        parts = key.split("_")
        k_val = [p for p in parts if p.startswith("k")][0][1:]
        w_val = [p for p in parts if p.startswith("w")][0][1:]
        mem_val = None
        if "_k" in key:
            mem_val = key.split("_k")[1]
        elif "_a" in key:
            mem_val = key.split("_a")[1]

        for tag in ["det", "rand_noctx", "rand_ctx"]:
            if tag not in triplet:
                continue
            seeds = triplet[tag].get("seeds", [])
            for si, seed_result in enumerate(seeds):
                dh = seed_result.get("drift_history", [])
                if dh:
                    tag_clean = tag.replace("_", "-")
                    prefix = f"{output_dir}/mem{mem_val}_k{k_val}_w{w_val}_{tag_clean}_s{si}"
                    np.save(f"{prefix}_drift.npy", np.array(dh))
                ph = seed_result.get("pred_history", [])
                th = seed_result.get("targ_history", [])
                if ph and th:
                    tag_clean = tag.replace("_", "-")
                    prefix = f"{output_dir}/mem{mem_val}_k{k_val}_w{w_val}_{tag_clean}_s{si}"
                    error = np.array(ph) - np.array(th)
                    np.save(f"{prefix}_error.npy", error)

    print(f"\nData saved to {output_dir}/")


def collect_performance(all_results):
    results = []
    for key, triplet in all_results.items():
        det = triplet.get("det", {})
        perf = det.get("mse_gap_mean", 0.0)
        results.append({
            "key": key,
            "mse_gap_mean": perf,
            "test_large_mean": det.get("test_large_mean", 0.0),
            "oracle_mse_mean": det.get("oracle_mse_mean", 0.0),
            "verdict": judge(triplet),
        })
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 10 MEMORY: Finite/Decay Memory Model Transfer")
    print("=" * 60)

    omega_list = np.linspace(0.01, 1.0, 6)

    all_results = run_experiment_memory_sweep(
        kappa=1.0,
        omega_list=omega_list,
        memory_k_list=[1, 5, 20],
        seeds=(42,),
        expert_hidden=2,
        K_budget=4,
        gating_hidden=8,
        train_steps=1200,
        test_steps=300,
    )

    print_matrix(all_results)

    save_data(all_results)

    perf = collect_performance(all_results)
    with open("memory_performance.json", "w") as f:
        json.dump(perf, f, indent=2)
    print("\nPerformance saved to memory_performance.json")

    print()
    print("=" * 60)
    print("STAGE 10 MEMORY COMPLETE")
