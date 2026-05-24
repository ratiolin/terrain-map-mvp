from experiment10 import run_experiment_grid
from analysis_stage10 import judge, print_matrix, has_structure


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 10: Structure Emergence — Flip Mode × Context")
    print("=" * 60)

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

    print()
    print("=" * 60)
    print("STAGE 10 COMPLETE")
