from experiment10 import run_experiment_grid, run_triplet
from analysis_stage10 import diagnose_triplet, print_diagnosis


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 10 DIAGNOSIS: drift=0.08 Focused Analysis")
    print("=" * 60)

    kappa_list = [1.0, 2.0, 4.0]
    drift = 0.08
    seeds = (42, 43, 44)

    all_results = {}
    for kappa in kappa_list:
        key = f"k{kappa}_d{drift}"
        print(f"\n{'='*60}")
        print(f"TRIPLET: kappa={kappa}, drift={drift}")
        print(f"{'='*60}")
        triplet = run_triplet(
            kappa=kappa, drift=drift,
            K_budget=4, expert_hidden=2, gating_hidden=8,
            train_steps=1200, test_steps=300,
            seeds=seeds,
        )
        all_results[key] = triplet

    print_diagnosis(all_results)

    print()
    print("=" * 60)
    print("STAGE 10 DIAGNOSIS COMPLETE")
