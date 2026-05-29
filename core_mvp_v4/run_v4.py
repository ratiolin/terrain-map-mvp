"""core_mvp_v4: High-dimensional closed-loop experiments.

Entry point for running all 5 experiment layers.
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D_DIMS = [4, 8, 16, 32]
K = 2
SEEDS = list(range(8))
DEFAULT_EPISODES = 5
DEFAULT_STEPS = 2000
TOTAL_DRIFT_STEPS = 10000


def main():
    parser = argparse.ArgumentParser(description="core_mvp_v4 experiments")
    parser.add_argument("--layer", type=str, nargs="*", default=None,
                        choices=["L1", "L2", "L3", "L4", "L5", "all"],
                        help="Which layers to run. Default: all")
    parser.add_argument("--d-dims", type=int, nargs="*", default=D_DIMS,
                        help=f"State dimensions to test. Default: {D_DIMS}")
    parser.add_argument("--seeds", type=int, default=8,
                        help=f"Number of random seeds per experiment. Default: 8")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help=f"Training episodes. Default: {DEFAULT_EPISODES}")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Steps per episode. Default: {DEFAULT_STEPS}")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: d=[4], seeds=2, episodes=2, steps=500")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Output directory. Default: core_mvp_v4/results/")

    args = parser.parse_args()

    if args.quick:
        d_dims = [4]
        seeds = list(range(2))
        episodes = 2
        steps = 500
        drift_steps = 2000
    else:
        d_dims = args.d_dims
        seeds = list(range(args.seeds))
        episodes = args.episodes
        steps = args.steps
        drift_steps = TOTAL_DRIFT_STEPS

    results_dir = args.results_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    config = {
        "k": K,
        "d_dims": d_dims,
        "n_seeds": len(seeds),
        "episodes": episodes,
        "steps_per_episode": steps,
        "drift_total_steps": drift_steps,
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    layers = args.layer or ["all"]
    if "all" in layers:
        layers = ["L1", "L2", "L3", "L4", "L5"]

    print(f"\n{'='*60}")
    print(f"core_mvp_v4: High-Dimensional Closed-Loop Experiments")
    print(f"{'='*60}")
    print(f"  Controlled dimensions (k): {K}")
    print(f"  State dimensions (d): {d_dims}")
    print(f"  Seeds: {seeds}")
    print(f"  Training: {episodes} episodes x {steps} steps")
    print(f"  Results: {results_dir}")
    print(f"  Layers: {layers}")
    print(f"{'='*60}\n")

    for layer in layers:
        print(f"\n{'#'*60}")
        print(f"# {layer}")
        print(f"{'#'*60}")

        if layer == "L1":
            from core_mvp_v4.experiments.layer1_closed_loop import run_layer1
            run_layer1(results_dir, d_dims=d_dims, seeds=seeds,
                       episodes=episodes, steps=steps)
        elif layer == "L2":
            from core_mvp_v4.experiments.layer2_adapt_shaping import run_layer2
            run_layer2(results_dir, d_dims=d_dims, seeds=seeds,
                       total_steps=drift_steps)
        elif layer == "L3":
            from core_mvp_v4.experiments.layer3_endogenous import run_layer3
            run_layer3(results_dir, d_dims=d_dims, seeds=seeds,
                       total_steps=drift_steps)
        elif layer == "L4":
            from core_mvp_v4.experiments.layer4_controllability import run_layer4
            run_layer4(results_dir, d_dims=d_dims, seeds=seeds,
                       episodes=episodes, steps=steps)
        elif layer == "L5":
            from core_mvp_v4.experiments.layer5_equivalence import run_layer5
            run_layer5(results_dir, d=16, n_seeds=len(seeds),
                       episodes=episodes, steps=steps)

    from core_mvp_v4.experiments.layer_summary import summarize_all_layers
    summarize_all_layers(results_dir)

    print(f"\nAll experiments complete. Results in {results_dir}/")


if __name__ == "__main__":
    main()
