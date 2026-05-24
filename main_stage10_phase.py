"""Phase diagram sweep: (drift, inertia) → S_adv → binary label."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment10 import run_triplet
from metrics import compute_stability


DRIFTS = [0.003, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08]
INERTIAS = [0.0, 0.2, 0.4, 0.6, 0.8]
KAPPA = 2.0
SEEDS = (42, 43)
TRAIN_STEPS = 800
TEST_STEPS = 200


def run_one_point(drift, inertia):
    key = f"d{drift}_i{inertia}"
    print(f"  [{key}] ", end="", flush=True)
    triplet = run_triplet(
        kappa=KAPPA, drift=drift,
        K_budget=4, expert_hidden=2, gating_hidden=8,
        train_steps=TRAIN_STEPS, test_steps=TEST_STEPS,
        seeds=SEEDS, inertia=inertia,
    )
    det = triplet["det"]
    v = det["variance_mean"]
    c = det["consistency_mean"]
    d = det["dwell_time_mean"]
    S_ensemble = compute_stability(v, c, d)
    S_single = det["S_single_mean"]
    S_adv = S_ensemble / (S_single + 1e-8)

    K_final_vals = [s["K_final"] for s in det["seeds"]]
    K_final_mean = float(np.mean(K_final_vals))
    structure_exists = K_final_mean > 1

    print(f"S_adv={S_adv:.4f}  K={K_final_mean:.1f}  struct={'Y' if structure_exists else 'N'}")
    return S_adv, K_final_mean, int(structure_exists)


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE DIAGRAM SWEEP: (drift, inertia) → S_adv")
    print("=" * 60)
    print(f"Grid: {len(DRIFTS)} drifts x {len(INERTIAS)} inertias = {len(DRIFTS)*len(INERTIAS)} points")
    print()

    results = {}
    structure_map = {}
    for inertia in INERTIAS:
        for drift in DRIFTS:
            s, k, struct = run_one_point(drift, inertia)
            results[f"{drift}:{inertia}"] = s
            structure_map[f"{drift}:{inertia}"] = {"S_adv": s, "K": k, "struct": struct}

    with open("phase_diagram_raw.json", "w") as f:
        json.dump({
            "drifts": DRIFTS,
            "inertias": INERTIAS,
            "results": results,
            "structure_map": structure_map,
        }, f)

    Z_binary = np.zeros((len(INERTIAS), len(DRIFTS)))
    Z_ternary = np.zeros((len(INERTIAS), len(DRIFTS)))
    for i, eta in enumerate(INERTIAS):
        for j, d in enumerate(DRIFTS):
            key = f"{d}:{eta}"
            s = results[key]
            struct = structure_map[key]["struct"]
            Z_binary[i, j] = 1.0 if s > 1.0 else 0.0
            if struct == 0:
                Z_ternary[i, j] = 0
            elif s > 1.0:
                Z_ternary[i, j] = 2
            else:
                Z_ternary[i, j] = 1

    print("\n" + "=" * 60)
    print("TERNARY PHASE DIAGRAM")
    print("  0 = no structure")
    print("  1 = structure, S_adv <= 1")
    print("  2 = structure, S_adv > 1")
    print("=" * 60)
    header = f"{'eta/drift':>10}" + "".join(f"{d:>8.3f}" for d in DRIFTS)
    print(header)
    print("-" * (10 + 8 * len(DRIFTS)))
    for i, eta in enumerate(INERTIAS):
        row = f"{eta:>10.1f}" + "".join(f"{int(Z_ternary[i,j]):>8}" for j in range(len(DRIFTS)))
        print(row)

    print("\nBINARY (original):")
    print(header)
    print("-" * (10 + 8 * len(DRIFTS)))
    for i, eta in enumerate(INERTIAS):
        row = f"{eta:>10.1f}" + "".join(f"{int(Z_binary[i,j]):>8}" for j in range(len(DRIFTS)))
        print(row)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    cmap = plt.cm.RdYlGn
    im = ax.pcolormesh(DRIFTS, INERTIAS, Z_binary, cmap=cmap, vmin=0, vmax=1,
                       edgecolors='k', linewidth=0.5)
    ax.set_xlabel("drift")
    ax.set_ylabel("inertia")
    ax.set_title("Binary: S_adv > 1")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(["S<=1", "S>1"])
    for i, eta in enumerate(INERTIAS):
        for j, d in enumerate(DRIFTS):
            ax.text(d + 0.002, eta + 0.05, f"{int(Z_binary[i,j])}",
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white' if Z_binary[i,j] < 0.5 else 'black')

    ax2 = axes[1]
    from matplotlib.colors import ListedColormap
    cmap3 = ListedColormap(['lightgray', 'salmon', 'forestgreen'])
    im2 = ax2.pcolormesh(DRIFTS, INERTIAS, Z_ternary, cmap=cmap3, vmin=0, vmax=2,
                         edgecolors='k', linewidth=0.5)
    ax2.set_xlabel("drift")
    ax2.set_ylabel("inertia")
    ax2.set_title("Ternary: 0=no struct, 1=struct no adv, 2=struct+adv")
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
    cbar2.set_ticklabels(["0: no struct", "1: struct, S<=1", "2: struct, S>1"])
    for i, eta in enumerate(INERTIAS):
        for j, d in enumerate(DRIFTS):
            v = int(Z_ternary[i,j])
            color = 'black' if v == 0 else 'white'
            ax2.text(d + 0.002, eta + 0.05, f"{v}",
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color=color)

    plt.tight_layout()
    plt.savefig("phase_diagram.png", dpi=150)
    print("\nPhase diagrams saved to phase_diagram.png")

    print("\n" + "=" * 60)
    print("ANALYSIS WITHIN STRUCTURE MASK (struct exists)")
    print("=" * 60)

    mask = (Z_ternary >= 1)
    if mask.sum() > 0:
        rows, cols = np.where(mask)
        for ri, ci in zip(rows, cols):
            eta = INERTIAS[ri]
            d = DRIFTS[ci]
            v = int(Z_ternary[ri, ci])
            label = "ADVANTAGE" if v == 2 else "collapse"
            print(f"  drift={d:.3f}  eta={eta:.1f}  →  {label} (S_adv={results[f'{d}:{eta}']:.4f})")

        print()
        adv_mask = (Z_ternary == 2)
        if adv_mask.sum() > 0:
            ar, ac = np.where(adv_mask)
            print(f"Advantage region ({int(adv_mask.sum())} points):")
            for ri, ci in zip(ar, ac):
                print(f"  drift={DRIFTS[ci]:.3f}  eta={INERTIAS[ri]:.1f}")

            d_min = min(DRIFTS[ci] for ci in ac)
            d_max = max(DRIFTS[ci] for ci in ac)
            eta_min_adv = min(INERTIAS[ri] for ri in ar)
            eta_max_adv = max(INERTIAS[ri] for ri in ar)
            print(f"\n  Advantage span: drift=[{d_min:.3f}, {d_max:.3f}], "
                  f"eta=[{eta_min_adv:.1f}, {eta_max_adv:.1f}]")

    print("\n" + "=" * 60)
    print("STRUCTURE PRESENCE (across all drift, eta)")
    print("=" * 60)
    struct_fraction = (Z_ternary >= 1).sum() / Z_ternary.size
    print(f"  Points with structure: {(Z_ternary >= 1).sum()}/{Z_ternary.size} ({struct_fraction:.0%})")
    struct_adv_fraction = (Z_ternary == 2).sum() / Z_ternary.size
    print(f"  Structure AND advantage: {(Z_ternary == 2).sum()}/{Z_ternary.size} ({struct_adv_fraction:.0%})")
    struct_no_adv = (Z_ternary == 1).sum()
    print(f"  Structure but NO advantage: {struct_no_adv}/{Z_ternary.size}")

    if struct_no_adv > 0:
        ar1, ac1 = np.where(Z_ternary == 1)
        print(f"  These collapse points:")
        for ri, ci in zip(ar1, ac1):
            print(f"    drift={DRIFTS[ci]:.3f}  eta={INERTIAS[ri]:.1f}  "
                  f"S_adv={results[f'{DRIFTS[ci]}:{INERTIAS[ri]}']:.4f}  "
                  f"K={structure_map[f'{DRIFTS[ci]}:{INERTIAS[ri]}']['K']:.1f}")
