"""
Binary search for eta_max(drift) — the collapse boundary.

For each drift, find the inertia eta where S_adv crosses below 1.0,
using binary search in [0, eta_ceiling].

Data classification:
  - observed: collapse found within [0, eta_ceiling] → exact eta_max
  - censored:  no collapse even at eta_ceiling    → eta_max > eta_ceiling
"""
import json
import numpy as np

from experiment10 import run_triplet
from analysis_stage10 import judge


ETA_CEILING = 2.0
TOLERANCE = 0.1
DRIFT_LIST = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12]
KAPPA = 1.0
SEEDS = (42,)


def evaluate_at_eta(drift, eta):
    """Run triplet experiment at given drift and inertia, return S_adv."""
    triplet = run_triplet(
        kappa=KAPPA, drift=drift,
        K_budget=4, expert_hidden=2, gating_hidden=8,
        train_steps=1200, test_steps=300,
        seeds=SEEDS,
        inertia=eta,
    )
    det = triplet["det"]
    S_adv = det.get("S_adv", 0.0)
    if S_adv == 0.0:
        v = det["variance_mean"]
        c = det["consistency_mean"]
        d = det["dwell_time_mean"]
        from metrics import compute_stability
        S_ensemble = compute_stability(v, c, d)
        S_single = det["S_single_mean"]
        S_adv = S_ensemble / (S_single + 1e-8)
    return float(S_adv), triplet


def binary_search_eta_max(drift, eta_min=0.0, eta_max=ETA_CEILING):
    """Binary search for the inertia where S_adv crosses 1.0."""
    lo, hi = eta_min, eta_max
    n_evals = 0

    # First: check if even at eta_ceiling we survive (censored)
    s_hi, _ = evaluate_at_eta(drift, hi)
    n_evals += 1
    if s_hi > 1.0:
        return {"drift": drift, "eta_max": None, "eta_min": hi,
                "censored": True, "S_at_ceiling": s_hi,
                "n_evals": n_evals, "bound": f"> {hi}"}

    # First: check if already collapsed at eta_min
    s_lo, _ = evaluate_at_eta(drift, lo)
    n_evals += 1
    if s_lo <= 1.0:
        return {"drift": drift, "eta_max": lo, "eta_min": lo,
                "censored": False, "S_at_bound": s_lo,
                "n_evals": n_evals, "bound": f"= {lo:.4f}"}

    while hi - lo > TOLERANCE:
        mid = (lo + hi) / 2.0
        s_mid, _ = evaluate_at_eta(drift, mid)
        n_evals += 1
        if s_mid > 1.0:
            lo = mid
        else:
            hi = mid
        print(f"  binary: [{lo:.3f}, {hi:.3f}] S={s_mid:.4f}")

    return {"drift": drift, "eta_max": lo, "eta_min": lo,
            "censored": False, "S_at_bound": s_lo if s_lo > 1 else s_mid,
            "n_evals": n_evals, "bound": f"= {lo:.4f}"}


if __name__ == "__main__":
    print("=" * 60)
    print("BINARY SEARCH: eta_max(drift) collapse boundary")
    print(f"ETA_CEILING = {ETA_CEILING}, tolerance = {TOLERANCE}")
    print("=" * 60)

    results = []
    observed = []
    censored = []

    for drift in DRIFT_LIST:
        print(f"\n{'='*60}")
        print(f"DRIFT = {drift}")
        print(f"{'='*60}")
        result = binary_search_eta_max(drift)
        results.append(result)

        if result["censored"]:
            censored.append(result)
            print(f"  → CENSORED: eta_max > {ETA_CEILING} (S={result['S_at_ceiling']:.4f})")
        else:
            observed.append(result)
            print(f"  → OBSERVED: eta_max = {result['eta_max']:.4f} (n_evals={result['n_evals']})")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'drift':>8} {'eta_max':>10} {'type':>12} {'n_evals':>8}")
    print("-" * 45)
    for r in results:
        typ = "CENSORED" if r["censored"] else "OBSERVED"
        val = f"> {ETA_CEILING}" if r["censored"] else f"{r['eta_max']:.4f}"
        print(f"{r['drift']:>8.4f} {val:>10} {typ:>12} {r['n_evals']:>8}")

    print(f"\nObserved: {len(observed)}  Censored: {len(censored)}")

    export = {
        "eta_ceiling": ETA_CEILING,
        "tolerance": TOLERANCE,
        "observed": [{"drift": r["drift"], "eta_max": r["eta_max"], "S": r.get("S_at_bound", 0)}
                      for r in observed],
        "censored": [{"drift": r["drift"], "eta_min": r["eta_min"], "S_ceiling": r.get("S_at_ceiling", 0)}
                      for r in censored],
    }
    with open("binary_search_results.json", "w") as f:
        json.dump(export, f, indent=2)
    print("\nSaved to binary_search_results.json")
