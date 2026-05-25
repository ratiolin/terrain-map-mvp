"""
Validate unified_stability predictor U = alignment_norm * feasibility
against ground-truth performance across all (g, lambda) points.
Pure numpy — no scipy dependency.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def pearsonr(x, y):
    xm = x - x.mean()
    ym = y - y.mean()
    r = (xm * ym).sum() / (np.sqrt((xm ** 2).sum()) * np.sqrt((ym ** 2).sum()) + 1e-12)
    # p-value via t-distribution approx
    n = len(x)
    t = r * np.sqrt((n - 2) / (1 - r ** 2 + 1e-12))
    # Rough 2-tailed p using normal approx
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return r, p


def spearmanr(x, y):
    rx = np.argsort(np.argsort(x)).astype(float) + 1
    ry = np.argsort(np.argsort(y)).astype(float) + 1
    return pearsonr(rx, ry)


def normalized(a):
    mn, mx = a.min(), a.max()
    if mx - mn < 1e-8:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "gradient_alignment_2d.json")) as f:
        d = json.load(f)

    g_vals = d["g_values"]
    l_vals = d["lambda_values"]
    ng, nl = len(g_vals), len(l_vals)

    alignment = np.array(d["alignment_map"])
    cost = np.zeros((ng, nl))
    sadv = np.zeros((ng, nl))
    inzone = np.zeros((ng, nl))
    slope = np.zeros((ng, nl))

    for i, g in enumerate(g_vals):
        for j, lam in enumerate(l_vals):
            key = f"{g},{lam}"
            r = d["results"][key]
            cost[i, j] = r["mean_cost"]
            sadv[i, j] = r["S_adv"]
            inzone[i, j] = r["in_zone_rate"]
            slope[i, j] = r["policy_slope"]

    # ── Step 1: feasibility ──
    inzone_n = normalized(inzone)
    sadv_n = normalized(sadv)
    feasibility = np.minimum(inzone_n, sadv_n)

    # ── Step 2: normalized alignment ──
    align_n = normalized(alignment)

    # ── Step 3: unified stability ──
    U = align_n * feasibility

    # ── Step 4: ground truth performance ──
    perf = inzone / (1.0 + cost)

    # ── Flatten ──
    u_flat = U.flatten()
    p_flat = perf.flatten()
    a_flat = alignment.flatten()
    f_flat = feasibility.flatten()
    c_flat = cost.flatten()
    i_flat = inzone.flatten()

    # ── Step 5-6: correlations ──
    print("=" * 70)
    print("UNIFIED STABILITY VALIDATION")
    print("=" * 70)
    print()
    print(f"  n_points = {len(u_flat)}  ({ng} g × {nl} lambda)")
    print()

    rU, pU = pearsonr(u_flat, p_flat)
    rA, pA = pearsonr(a_flat, p_flat)
    rF, pF = pearsonr(f_flat, p_flat)
    rsU, psU = spearmanr(u_flat, p_flat)
    rsA, psA = spearmanr(a_flat, p_flat)
    rsF, psF = spearmanr(f_flat, p_flat)

    print("Pearson correlation:")
    print(f"  r(U, perf)          = {rU:+.4f}  (p={pU:.2e})")
    print(f"  r(alignment, perf)  = {rA:+.4f}  (p={pA:.2e})")
    print(f"  r(feasibility, perf)= {rF:+.4f}  (p={pF:.2e})")
    print()
    print("Spearman (rank) correlation:")
    print(f"  r(U, perf)          = {rsU:+.4f}  (p={psU:.2e})")
    print(f"  r(alignment, perf)  = {rsA:+.4f}  (p={psA:.2e})")
    print(f"  r(feasibility, perf)= {rsF:+.4f}  (p={psF:.2e})")
    print()

    # ── Step 7: baseline comparison ──
    print("Baseline comparison:")
    best_pearson = max(rU, rA, rF)
    best_spearman = max(rsU, rsA, rsF)
    print(f"  Best Pearson:  {best_pearson:+.4f}")
    print(f"  Best Spearman: {best_spearman:+.4f}")
    print()

    # Multiple performance measures
    perf_iz = inzone.flatten()
    perf_nc = -cost.flatten()
    perf_ratio = inzone.flatten() / (1.0 + cost.flatten())

    rU_iz, _ = pearsonr(u_flat, perf_iz)
    rU_nc, _ = pearsonr(u_flat, perf_nc)
    rA_iz, _ = pearsonr(a_flat, perf_iz)
    rA_nc, _ = pearsonr(a_flat, perf_nc)
    rF_iz, _ = pearsonr(f_flat, perf_iz)
    rF_nc, _ = pearsonr(f_flat, perf_nc)

    print("Across performance measures:")
    print(f"  {'':>18} {'U':>8} {'alignment':>10} {'feasibility':>12}")
    print(f"  {'in_zone_rate':>18} {rU_iz:>+8.4f} {rA_iz:>+10.4f} {rF_iz:>+12.4f}")
    print(f"  {'-cost':>18} {rU_nc:>+8.4f} {rA_nc:>+10.4f} {rF_nc:>+12.4f}")
    print(f"  {'in_zone/(1+cost)':>18} {rU:>+8.4f} {rA:>+10.4f} {rF:>+12.4f}")

    # ── Step 8: scatter plots ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("U = alignment_norm × feasibility  vs  performance", fontsize=13)

    for ax, x, label in zip(axes,
                            [u_flat, a_flat, f_flat],
                            ["Unified Stability U", "Gradient Alignment", "Feasibility"]):
        ax.scatter(x, p_flat, alpha=0.5, s=20, c=p_flat, cmap="viridis")
        r, _ = pearsonr(x, p_flat)
        ax.set_xlabel(label)
        ax.set_ylabel("in_zone / (1 + cost)")
        ax.set_title(f"r = {r:+.4f}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    sp = os.path.join(OUTPUT_DIR, "unified_stability_scatter.png")
    plt.savefig(sp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Scatter plot saved to {sp}")

    # ── Step 9: monotonicity check ──
    order = np.argsort(u_flat)
    p_sorted = p_flat[order]
    diffs = np.diff(p_sorted)
    n_increasing = np.sum(diffs > 0)
    n_total = len(diffs)
    mono_ratio = n_increasing / n_total if n_total > 0 else 0

    print(f"\nMonotonicity: {n_increasing}/{n_total} = {mono_ratio:.3f} of sorted differences are positive")
    print(f"  (1.0 = perfectly monotonic, 0.5 = random)")

    # ── Step 10: bin analysis ──
    bins = [0, 20, 40, 60, 80, 100]
    percentiles = np.percentile(u_flat, bins)
    bin_stats = []
    for k in range(len(percentiles) - 1):
        lo, hi = percentiles[k], percentiles[k + 1]
        mask = (u_flat >= lo) & (u_flat <= hi)
        if mask.sum() > 0:
            bin_stats.append({
                "bin": k + 1,
                "range": f"[{lo:.3f}, {hi:.3f}]",
                "n": int(mask.sum()),
                "mean_perf": float(p_flat[mask].mean()),
                "std_perf": float(p_flat[mask].std()),
                "mean_cost": float(c_flat[mask].mean()),
            })

    print(f"\nBin analysis (U percentiles):")
    print(f"  {'Bin':>5} {'Range':>20} {'n':>5} {'mean_perf':>10} {'std_perf':>10} {'mean_cost':>10}")
    for bs in bin_stats:
        print(f"  {bs['bin']:>5} {bs['range']:>20} {bs['n']:>5} "
              f"{bs['mean_perf']:>10.4f} {bs['std_perf']:>10.4f} {bs['mean_cost']:>10.4f}")

    # ── Step 11: 2D heatmap comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("U vs Performance — Structure Check", fontsize=13)

    im0 = axes[0].imshow(U.T, aspect="auto", origin="lower",
                          extent=[g_vals[0], g_vals[-1], l_vals[0], l_vals[-1]],
                          cmap="viridis")
    axes[0].set_title("Unified Stability U")
    axes[0].set_xlabel("g"); axes[0].set_ylabel("lambda")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(perf.T, aspect="auto", origin="lower",
                          extent=[g_vals[0], g_vals[-1], l_vals[0], l_vals[-1]],
                          cmap="viridis")
    axes[1].set_title("Performance = in_zone/(1+cost)")
    axes[1].set_xlabel("g"); axes[1].set_ylabel("lambda")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    hp = os.path.join(OUTPUT_DIR, "unified_stability_heatmaps.png")
    plt.savefig(hp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved to {hp}")

    # ── Step 12: extreme point check ──
    u_perc_80 = np.percentile(u_flat, 80)
    p_perc_20 = np.percentile(p_flat, 20)
    high_u_low_p = np.where((u_flat >= u_perc_80) & (p_flat <= p_perc_20))[0]
    low_u_high_p = np.where((u_flat <= np.percentile(u_flat, 20)) & (p_flat >= np.percentile(p_flat, 80)))[0]

    print(f"\nExtreme points:")
    print(f"  High-U Low-Perf (failure): {len(high_u_low_p)} points")
    if len(high_u_low_p) > 0:
        for idx in high_u_low_p[:5]:
            gi, li = divmod(idx, nl)
            print(f"    g={g_vals[gi]:.2f} λ={l_vals[li]:.3f}  U={u_flat[idx]:.3f} perf={p_flat[idx]:.4f}  cost={c_flat[idx]:.3f} in_zone={inzone.flatten()[idx]:.3f}")
    print(f"  Low-U High-Perf (miss):  {len(low_u_high_p)} points")
    if len(low_u_high_p) > 0:
        for idx in low_u_high_p[:5]:
            gi, li = divmod(idx, nl)
            print(f"    g={g_vals[gi]:.2f} λ={l_vals[li]:.3f}  U={u_flat[idx]:.3f} perf={p_flat[idx]:.4f}  cost={c_flat[idx]:.3f} in_zone={inzone.flatten()[idx]:.3f}")

    # ── Step 13: verdict ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    u_win_pearson = rU >= rA and rU >= rF
    u_win_spearman = rsU >= rsA and rsU >= rsF
    mono_ok = mono_ratio > 0.65
    bin_monotonic = all(
        bin_stats[i]["mean_perf"] <= bin_stats[i + 1]["mean_perf"]
        for i in range(len(bin_stats) - 1)
    )

    supported = u_win_pearson or u_win_spearman

    print(f"  U best Pearson:  {'YES' if u_win_pearson else 'NO'}  (U={rU:+.4f} vs A={rA:+.4f} vs F={rF:+.4f})")
    print(f"  U best Spearman: {'YES' if u_win_spearman else 'NO'}  (U={rsU:+.4f} vs A={rsA:+.4f} vs F={rsF:+.4f})")
    print(f"  Monotonic:       {'YES' if mono_ok else 'NO'}  ({mono_ratio:.3f} > 0.65)")
    print(f"  Bin monotonic:   {'YES' if bin_monotonic else 'NO'}")
    print()

    if supported:
        print("  UNIFIED METRIC SUPPORTED")
        print(f"  U can predict performance ordering better than raw alignment or feasibility alone.")
    else:
        print("  REJECTED")
        print(f"  U does not outperform baselines in predicting performance.")
        print(f"  Feasibility alone (r={rF:+.4f}) is the stronger predictor.")

    # ── Save verdict ──
    result = {
        "experiment": "unified_stability_validation",
        "n_points": len(u_flat),
        "correlations": {
            "pearson": {"U": float(rU), "alignment": float(rA), "feasibility": float(rF)},
            "spearman": {"U": float(rsU), "alignment": float(rsA), "feasibility": float(rsF)},
        },
        "monotonicity": float(mono_ratio),
        "bin_monotonic": bool(bin_monotonic),
        "verdict": "supported" if supported else "rejected",
        "bin_analysis": bin_stats,
    }

    vp = os.path.join(OUTPUT_DIR, "unified_stability_verdict.json")
    with open(vp, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Verdict saved to {vp}")


if __name__ == "__main__":
    main()
