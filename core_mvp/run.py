"""core_mvp — Unified entry point.

Usage:
  uv run python core_mvp/run.py --layer 1
  uv run python core_mvp/run.py --layer 2
  uv run python core_mvp/run.py --layer 3
  uv run python core_mvp/run.py --layer 4
  uv run python core_mvp/run.py --layer all
  uv run python core_mvp/run.py --quick
"""

import os, sys, argparse, json, time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)

RESULTS = os.path.join(_here, "results")

D_DIMS = [2, 5, 10, 20, 50, 100]
SEEDS = 5
EPISODES = 5
STEPS = 2000


def main():
    p = argparse.ArgumentParser(description="core_mvp experiments")
    p.add_argument("--layer", type=str, nargs="*", default=["all"],
                   choices=["1", "2", "3", "4", "all"])
    p.add_argument("--phase", type=str, default="all", choices=["A", "B", "all"])
    p.add_argument("--d-dims", type=int, nargs="*", default=D_DIMS)
    p.add_argument("--seeds", type=int, default=SEEDS)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if args.quick:
        d_dims = [2, 10]; seeds = 3; eps = 2; stp = 500
    else:
        d_dims = args.d_dims; seeds = args.seeds; eps = args.episodes; stp = args.steps

    layers = args.layer
    if "all" in layers:
        layers = ["1", "2", "3", "4"]

    print(f"\n{'='*60}")
    print(f"core_mvp  |  layers={layers}  |  seeds={seeds}")
    print(f"{'='*60}")

    for layer in layers:
        print(f"\n### Layer {layer} ###")
        t0 = time.time()

        if layer == "1":
            from core_mvp.layers.layer1_final import run_phase_a, run_phase_b
            od = os.path.join(RESULTS, "layer1")
            if args.phase in ("A", "all"):
                run_phase_a(d_dims, list(range(seeds)), eps, stp, 1000, od)
            if args.phase in ("B", "all"):
                run_phase_b(d_dims, list(range(seeds)), eps, stp, 1000, od)

        elif layer == "2":
            from core_mvp.layers.layer2_final import run_layer2
            run_layer2(seeds=list(range(seeds)))

        elif layer == "3":
            from core_mvp.layers.layer3_final import run_layer3
            run_layer3(n_seeds=seeds)

        elif layer == "4":
            from core_mvp.layers.layer4_final import run_delay, run_minimal, run_complexity, run_deprivation, run_truefalse, run_dual_path
            out4 = f"{RESULTS}/layer4"
            os.makedirs(out4, exist_ok=True)
            for fn, name in [(run_delay, "delay"), (run_minimal, "minimal"),
                              (run_complexity, "complexity"), (run_deprivation, "deprivation"),
                              (run_truefalse, "truefalse"), (run_dual_path, "dual_path")]:
                print(f"\n  L4-{name}")
                fn(seeds, stp)

        print(f"  Layer {layer} done in {time.time()-t0:.1f}s")

    print(f"\nDone. Results in {RESULTS}/")


if __name__ == "__main__":
    main()
