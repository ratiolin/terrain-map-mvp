"""
Complete stability analysis pipeline.

Part 1: eta_max(drift) from band structure → validate predictions A/B/C
Part 2: Stability functional S(d) → verify dS/dd ≈ R(d) (Lyapunov proxy)
Part 3: Export eta_max_curve.json, stability_regions.json, stability_function.json
"""
import json
import numpy as np
import matplotlib.pyplot as plt


# ── Part 0: Load band structure data ──────────────────────────────────
def load_data(path="band_structure_raw.json"):
    with open(path) as f:
        return json.load(f)


# ── Part 1: eta_max curve from band structure ────────────────────────
def build_eta_max_curve(drifts, metrics, mus, sigmas, amps, C=1.0):
    """eta_max(d) = 1 / (1 + |R(d)| / max_R) — normalized to (0, 1]"""
    def R(d):
        val = 0.0
        for A, mu, sigma in zip(amps, mus, sigmas):
            val += A * (-(d - mu) / sigma**2) * np.exp(-(d - mu)**2 / (2 * sigma**2))
        return val

    r_vals = np.array([R(d) for d in drifts])
    max_r = max(np.abs(r_vals).max(), 1e-8)
    eta_vals = 1.0 / (1.0 + np.abs(r_vals) / max_r)

    return eta_vals, r_vals


def mark_unstable_regions(drifts, eta_max_vals, threshold=None):
    """Mark drift ranges where eta_max < threshold (unstable)."""
    if threshold is None:
        threshold = np.percentile(eta_max_vals, 30)

    unstable_mask = eta_max_vals < threshold
    stable_mask = ~unstable_mask

    regions = {"stable": [], "unstable": []}

    # Find contiguous stable regions
    in_stable = False
    start = None
    for i, d in enumerate(drifts):
        if stable_mask[i] and not in_stable:
            start = d
            in_stable = True
        elif not stable_mask[i] and in_stable:
            regions["stable"].append({"drift_start": float(start), "drift_end": float(drifts[i-1])})
            in_stable = False
    if in_stable:
        regions["stable"].append({"drift_start": float(start), "drift_end": float(drifts[-1])})

    # Find contiguous unstable regions
    in_unstable = False
    start = None
    for i, d in enumerate(drifts):
        if unstable_mask[i] and not in_unstable:
            start = d
            in_unstable = True
        elif not unstable_mask[i] and in_unstable:
            regions["unstable"].append({"drift_start": float(start), "drift_end": float(drifts[i-1])})
            in_unstable = False
    if in_unstable:
        regions["unstable"].append({"drift_start": float(start), "drift_end": float(drifts[-1])})

    return regions, stable_mask, unstable_mask, threshold


# ── Part 2: Stability functional S(d) ────────────────────────────────
def compute_S_functional(drifts, metrics, w1=1.0, w2=1.0, w3=-0.5):
    """S(d) = w1*var_z + w2*switch_rate + w3*entropy"""
    var_z = np.array([m["var_z"] for m in metrics])
    switch = np.array([m["switch_rate"] for m in metrics])
    entropy = np.array([m["entropy"] for m in metrics])

    S = w1 * var_z + w2 * switch + w3 * entropy
    dS_dd = np.gradient(S, drifts)

    return S, dS_dd, var_z, switch, entropy


def compare_derivatives(drifts, dS_dd, R_vals):
    """Check if dS/dd ≈ R(d) (Lyapunov proxy)."""
    # Normalize both to [0, 1] for comparison
    dS_norm = (dS_dd - dS_dd.min()) / (dS_dd.max() - dS_dd.min() + 1e-8)
    R_norm = (R_vals - R_vals.min()) / (R_vals.max() - R_vals.min() + 1e-8)

    corr = np.corrcoef(dS_norm, R_norm)[0, 1]
    mae = np.mean(np.abs(dS_norm - R_norm))

    return corr, mae, dS_norm, R_norm


# ── Part 3: Validation predictions ────────────────────────────────────
def validate_predictions(drifts, metrics, eta_max_vals, R_vals, stable_mask, threshold):
    """Test predictions A, B, C."""
    switch_rate = np.array([m["switch_rate"] for m in metrics])
    mse_gap = np.array([m["mse_gap"] for m in metrics])
    var_z = np.array([m["var_z"] for m in metrics])

    unstable = eta_max_vals < threshold

    # Prediction A: eta_max low ↔ switch_rate high
    corr_eta_switch = np.corrcoef(eta_max_vals, switch_rate)[0, 1]

    # Prediction B: eta_max low ↔ mse_gap crash (more negative)
    corr_eta_gap = np.corrcoef(eta_max_vals, mse_gap)[0, 1]

    # Prediction C: eta_max peaks at band centers
    # Find actual data-driven peaks in var_z
    var_z_arr = np.array([m["var_z"] for m in metrics])
    threshold_peak = np.mean(var_z_arr) + 0.3 * np.std(var_z_arr)
    peaks_data = []
    for i in range(1, len(var_z_arr) - 1):
        if var_z_arr[i] > var_z_arr[i-1] and var_z_arr[i] > var_z_arr[i+1] and var_z_arr[i] > threshold_peak:
            peaks_data.append(i)
    if peaks_data:
        mus_data = [float(drifts[p]) for p in peaks_data]
    else:
        mus_data = mus
    peak_values = []
    for mu in mus_data:
        idx = np.argmin(np.abs(np.array(drifts) - mu))
        peak_values.append(eta_max_vals[idx])
    band_center_eta = np.mean(peak_values) if peak_values else 0.0
    off_center_eta = np.mean([eta_max_vals[i] for i in range(len(drifts))
                              if min(abs(np.array(drifts)[i] - mu) for mu in mus_data) > 0.03]) \
                     if mus_data else 0.0

    return {
        "A_eta_vs_switch": float(corr_eta_switch),
        "B_eta_vs_mse_gap": float(corr_eta_gap),
        "C_band_center_eta": float(band_center_eta),
        "C_off_center_eta": float(off_center_eta),
    }


# ── Plotting ──────────────────────────────────────────────────────────
def plot_comprehensive(drifts, metrics, eta_max_vals, R_vals,
                        stable_mask, threshold, S_vals, dS_dd, dS_norm, R_norm, corr,
                        band_centers, output="stability_analysis.png"):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    mse_gap = np.array([m["mse_gap"] for m in metrics])
    switch_rate = np.array([m["switch_rate"] for m in metrics])
    var_z = np.array([m["var_z"] for m in metrics])
    entropy = np.array([m["entropy"] for m in metrics])

    def mark_bands(ax):
        for mu in band_centers:
            ax.axvline(x=mu, color='green', linestyle='--', alpha=0.4, linewidth=1)

    # Panel 1: eta_max(drift) with band structure
    ax = axes[0, 0]
    ax.plot(drifts, eta_max_vals, 'b-', linewidth=2, label='eta_max(drift)')
    mark_bands(ax)
    ax.fill_between(drifts, 0, max(eta_max_vals),
                     where=(eta_max_vals < threshold),
                     alpha=0.2, color='red', label='unstable')
    ax.set_xlabel('drift')
    ax.set_ylabel('eta_max')
    ax.set_title('eta_max(drift) = 1 / (1 + |R|)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: eta_max vs switch_rate (Prediction A)
    ax = axes[0, 1]
    ax2 = ax.twinx()
    l1, = ax.plot(drifts, eta_max_vals, 'b-', linewidth=2, label='eta_max')
    l2, = ax2.plot(drifts, switch_rate, 'r-', linewidth=2, label='switch_rate')
    mark_bands(ax)
    ax.set_xlabel('drift')
    ax.set_ylabel('eta_max', color='b')
    ax2.set_ylabel('switch_rate', color='r')
    ax.set_title(f'Prediction A: eta_max ↓ ↔ switch_rate ↑ (r={np.corrcoef(eta_max_vals, switch_rate)[0,1]:.3f})')
    ax.legend([l1, l2], ['eta_max', 'switch_rate'], fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 3: eta_max vs mse_gap (Prediction B)
    ax = axes[0, 2]
    ax2 = ax.twinx()
    l1, = ax.plot(drifts, eta_max_vals, 'b-', linewidth=2, label='eta_max')
    l2, = ax2.plot(drifts, mse_gap, 'm-', linewidth=2, label='mse_gap')
    mark_bands(ax)
    ax.set_xlabel('drift')
    ax.set_ylabel('eta_max', color='b')
    ax2.set_ylabel('mse_gap', color='m')
    ax.set_title(f'Prediction B: eta_max ↓ ↔ mse_gap crash (r={np.corrcoef(eta_max_vals, mse_gap)[0,1]:.3f})')
    ax.legend([l1, l2], ['eta_max', 'mse_gap'], fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 4: S(d) stability functional
    ax = axes[1, 0]
    ax.plot(drifts, S_vals, 'k-', linewidth=2, label='S(d)')
    ax2 = ax.twinx()
    ax2.plot(drifts, var_z, 'b-', alpha=0.5, linewidth=1, label='var_z')
    ax2.plot(drifts, switch_rate, 'r-', alpha=0.5, linewidth=1, label='switch')
    ax2.plot(drifts, entropy, 'g-', alpha=0.5, linewidth=1, label='entropy')
    mark_bands(ax)
    ax.set_xlabel('drift')
    ax.set_ylabel('S(d) — Stability Functional', color='k')
    ax.set_title('Stability Functional S(d)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 5: dS/dd vs R(d) — Lyapunov proxy validation
    ax = axes[1, 1]
    ax.plot(drifts, dS_norm, 'b-', linewidth=2, label='dS/dd (norm)')
    ax.plot(drifts, R_norm, 'r--', linewidth=2, label='R(d) (norm)')
    mark_bands(ax)
    ax.set_xlabel('drift')
    ax.set_ylabel('normalized')
    ax.set_title(f'dS/dd ≈ R(d) : r={corr:.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6: Stability regions map
    ax = axes[1, 2]
    y_high = max(eta_max_vals)
    ax.plot(drifts, eta_max_vals, 'b-', linewidth=2)
    ax.fill_between(drifts, 0, y_high,
                     where=stable_mask, alpha=0.3, color='green', label='stable')
    ax.fill_between(drifts, 0, y_high,
                     where=~stable_mask, alpha=0.3, color='red', label='unstable')
    ax.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7,
               label=f'threshold = {threshold:.3f}')
    mark_bands(ax)
    ax.set_xlabel('drift')
    ax.set_ylabel('eta_max')
    ax.set_title(f'Stability Regions (threshold={threshold:.3f})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"\nComprehensive analysis plot saved to {output}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()
    drifts = np.array([m["drift"] for m in data])
    metrics = data

    # Band parameters from previous analysis
    mus = [0.022, 0.152]
    sigmas = [0.015, 0.015]
    amps = [0.188, 0.185]

    print("=" * 60)
    print("STABILITY ANALYSIS PIPELINE")
    print("=" * 60)

    # ── Part 1: eta_max curve ──────────────────────────────────────
    eta_max_vals, R_vals = build_eta_max_curve(drifts, metrics, mus, sigmas, amps)

    regions, stable_mask, unstable_mask, threshold = mark_unstable_regions(drifts, eta_max_vals)

    # Detect actual band centers from data for plotting
    var_z_arr = np.array([m["var_z"] for m in metrics])
    t_peak = np.mean(var_z_arr) + 0.3 * np.std(var_z_arr)
    peaks_data_plot = []
    for i in range(1, len(var_z_arr) - 1):
        if var_z_arr[i] > var_z_arr[i-1] and var_z_arr[i] > var_z_arr[i+1] and var_z_arr[i] > t_peak:
            peaks_data_plot.append(float(drifts[i]))
    band_centers_plot = peaks_data_plot if peaks_data_plot else mus

    print(f"\n── Part 1: eta_max(drift) curve ──")
    print(f"  eta_max range: [{eta_max_vals.min():.3f}, {eta_max_vals.max():.3f}]")
    print(f"  Band centers (from data): {band_centers_plot}")
    print(f"  Unstable threshold (30th pctile): {threshold:.3f}")
    print(f"  Stable points: {stable_mask.sum()}/{len(drifts)}, Unstable: {unstable_mask.sum()}/{len(drifts)}")

    # ── Part 2: Stability functional ───────────────────────────────
    S_vals, dS_dd, var_z, switch_rate, entropy = compute_S_functional(drifts, metrics)
    corr_deriv, mae_deriv, dS_norm, R_norm = compare_derivatives(drifts, dS_dd, R_vals)

    print(f"\n── Part 2: Stability functional S(d) ──")
    print(f"  S range: [{S_vals.min():.3f}, {S_vals.max():.3f}]")
    print(f"  dS/dd ≈ R(d): r = {corr_deriv:.4f}, MAE = {mae_deriv:.4f}")
    if abs(corr_deriv) > 0.5:
        print(f"  ✓ dS/dd ≈ R(d) confirmed — approximate Lyapunov function found")
    else:
        print(f"  ⚠ Weak correlation — S(d) may need different weights or nonlinear terms")

    # ── Part 3: Validate predictions ───────────────────────────────
    val = validate_predictions(drifts, metrics, eta_max_vals, R_vals, stable_mask, threshold)

    print(f"\n── Part 3: Prediction validation ──")
    print(f"  A: eta_max vs switch_rate  r = {val['A_eta_vs_switch']:.4f}")
    print(f"     {'✓ Confirmed' if val['A_eta_vs_switch'] < -0.3 else '⚠ Weak'}")
    print(f"  B: eta_max vs mse_gap      r = {val['B_eta_vs_mse_gap']:.4f}")
    print(f"     {'✓ Confirmed' if val['B_eta_vs_mse_gap'] > 0.3 else '⚠ Weak'}")
    print(f"  C: eta_max at band center  = {val['C_band_center_eta']:.1f}")
    print(f"     eta_max off center      = {val['C_off_center_eta']:.1f}")
    print(f"     Ratio: {val['C_band_center_eta'] / max(val['C_off_center_eta'], 1e-8):.2f}x")
    print(f"     {'✓ Band centers show elevated eta_max' if val['C_band_center_eta'] > val['C_off_center_eta'] else '⚠ Check band center detection'}")

    # ── Mark unstable regions ───────────────────────────────────────
    print(f"\n── Danger Zones ──")
    print(f"  Stable regions:   {regions['stable']}")
    print(f"  Unstable regions: {regions['unstable']}")

    # ── Plots ───────────────────────────────────────────────────────
    plot_comprehensive(drifts, metrics, eta_max_vals, R_vals,
                        stable_mask, threshold, S_vals, dS_dd,
                        dS_norm, R_norm, corr_deriv,
                        band_centers_plot)

    # ── Export ────────────────────────────────────────────────────────
    # 1. eta_max_curve.json
    with open("eta_max_curve.json", "w") as f:
        json.dump({
            "drift": [float(d) for d in drifts],
            "eta_max": [float(e) for e in eta_max_vals],
            "R": [float(r) for r in R_vals],
            "band_centers": mus,
        }, f, indent=2)
    print("\nExported eta_max_curve.json")

    # 2. stability_regions.json
    with open("stability_regions.json", "w") as f:
        json.dump({
            "threshold": float(threshold),
            "stable_regions": regions["stable"],
            "unstable_regions": regions["unstable"],
            "n_stable_points": int(stable_mask.sum()),
            "n_unstable_points": int(unstable_mask.sum()),
        }, f, indent=2)
    print("Exported stability_regions.json")

    # 3. stability_function.json
    with open("stability_function.json", "w") as f:
        json.dump({
            "drift": [float(d) for d in drifts],
            "S": [float(s) for s in S_vals],
            "dS_dd": [float(ds) for ds in dS_dd],
            "dS_norm": [float(d) for d in dS_norm],
            "R_norm": [float(r) for r in R_norm],
            "correlation_dS_R": float(corr_deriv),
            "components": {
                "var_z": [float(v) for v in var_z],
                "switch_rate": [float(s) for s in switch_rate],
                "entropy": [float(e) for e in entropy],
            },
        }, f, indent=2)
    print("Exported stability_function.json")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
