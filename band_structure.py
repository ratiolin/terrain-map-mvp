"""
E(drift) — Band structure emergence function.

Dense drift sweep → compute structural metrics → fit Gaussian mixtures.

Key metric: M = Var(z) — variance of routing weights over time,
measuring how strongly the model differentiates between experts.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

from experiment10 import run_triplet


KAPPA = 1.0
SEEDS = (42,)
DRIFT_LIST = list(np.linspace(0.001, 0.2, 30))


def compute_structural_metrics(triplet):
    """Extract structural indicators from experiment results."""
    det = triplet["det"]
    seeds = det.get("seeds", [])
    if not seeds:
        return None

    s = seeds[0]
    z_history = s.get("z_history", [])

    metrics = {}

    if z_history:
        z_arr = np.array([z[:4] if len(z) >= 4 else np.pad(z, (0, 4 - len(z)))
                          for z in z_history])
        K_eff = (z_arr > 0.01).sum(axis=0)
        K_used = int(np.sum(K_eff > 5))

        z_flat = z_arr.mean(axis=0)
        ent = -np.sum(z_flat * np.log(z_flat + 1e-8))
        max_ent = np.log(len(z_flat))
        norm_ent = ent / max_ent if max_ent > 0 else 0

        var_z = float(np.var(z_arr.mean(axis=0)))
        max_z = float(np.max(z_arr.mean(axis=0)))
        switch_rate = det.get("switch_rate_mean", 0)

        metrics["var_z"] = var_z
        metrics["max_z"] = max_z
        metrics["K_used"] = K_used
        metrics["entropy"] = float(norm_ent)
        metrics["switch_rate"] = float(switch_rate)
        metrics["mse_gap"] = det.get("mse_gap_mean", 0)
        metrics["S_adv"] = det.get("S_adv", 0)
        metrics["oracle_mse"] = det.get("oracle_mse_mean", 0)
        metrics["large_mse"] = det.get("test_large_mean", 0)

    return metrics


def run_dense_drift_sweep(drift_list, output_file="band_structure_raw.json"):
    results = []
    for i, drift in enumerate(drift_list):
        print(f"\r[{i+1}/{len(drift_list)}] drift={drift:.4f} ...", end="", flush=True)
        triplet = run_triplet(
            kappa=KAPPA, drift=drift,
            K_budget=4, expert_hidden=2, gating_hidden=8,
            train_steps=800, test_steps=200,
            seeds=SEEDS,
            inertia=0.0,
        )
        m = compute_structural_metrics(triplet)
        if m:
            m["drift"] = float(drift)
            results.append(m)
        if (i + 1) % 5 == 0:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=1)
            print(f"  [saved {len(results)}]")
    print()

    with open(output_file, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {len(results)} points to {output_file}")
    return results


def fit_gaussian_mixture(drifts, M_values, max_peaks=5):
    """Fit sum of Gaussians to M(drift) using peak detection."""
    sigma_smooth = max(2.0, len(drifts) * 0.05)
    window = int(4 * sigma_smooth)
    if window < 2:
        window = 2
    kernel = np.exp(-0.5 * (np.arange(-window, window + 1) / sigma_smooth)**2)
    kernel /= kernel.sum()
    smoothed = np.convolve(M_values, kernel, mode='same')
    peaks = []
    n = len(smoothed)
    for i in range(1, n - 1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            if smoothed[i] > np.mean(smoothed) + 0.3 * np.std(smoothed):
                peaks.append(i)

    if len(peaks) > max_peaks:
        heights = [(smoothed[p], p) for p in peaks]
        heights.sort(reverse=True)
        peaks = [h[1] for h in heights[:max_peaks]]

    peaks.sort()

    fit_params = []
    for p in peaks:
        d_i = drifts[p]
        a_i = M_values[p]
        sigma_i = (drifts[-1] - drifts[0]) * 0.05
        window = max(3, int(0.02 * n))
        lo = max(0, p - window)
        hi = min(n, p + window)
        local_y = M_values[lo:hi]
        local_x = drifts[lo:hi]
        if len(local_y) > 1:
            half_max = a_i / 2
            above = np.where(local_y > half_max)[0]
            if len(above) >= 2:
                sigma_i = (local_x[above[-1]] - local_x[above[0]]) / (2 * np.sqrt(2 * np.log(2)))
        fit_params.append((d_i, a_i, max(sigma_i, 0.005)))

    return fit_params


def plot_band_structure(drifts, metrics_list, fit_params, output="band_structure.png"):
    M = np.array([m["var_z"] for m in metrics_list])
    S = np.array([m["S_adv"] for m in metrics_list])
    mse_gap = np.array([m["mse_gap"] for m in metrics_list])
    switch_rate = np.array([m["switch_rate"] for m in metrics_list])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Var(z) — main structural indicator
    ax = axes[0, 0]
    ax.plot(drifts, M, 'b-', linewidth=1.5, alpha=0.7)
    ax.scatter(drifts, M, c='blue', s=20, alpha=0.5)
    if fit_params:
        d_fine = np.linspace(drifts[0], drifts[-1], 500)
        M_fit = np.zeros_like(d_fine)
        for d_i, a_i, sigma_i in fit_params:
            M_fit += a_i * np.exp(-(d_fine - d_i)**2 / (2 * sigma_i**2))
            ax.axvline(x=d_i, color='red', linestyle='--', alpha=0.4, linewidth=1)
        ax.plot(d_fine, M_fit, 'r-', linewidth=2, label=f'Gaussian mix ({len(fit_params)} peaks)')
    ax.set_xlabel('drift')
    ax.set_ylabel('Var(z) — routing variance')
    ax.set_title('E(drift): Band Structure (Var(z))')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: S_adv
    ax = axes[0, 1]
    ax.plot(drifts, S, 'g-', linewidth=1.5, alpha=0.7)
    ax.scatter(drifts, S, c='green', s=20, alpha=0.5)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='S_adv = 1')
    ax.set_xlabel('drift')
    ax.set_ylabel('S_adv')
    ax.set_title('Stability Advantage')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: mse_gap
    ax = axes[1, 0]
    ax.plot(drifts, mse_gap, 'm-', linewidth=1.5, alpha=0.7)
    ax.scatter(drifts, mse_gap, c='magenta', s=20, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('drift')
    ax.set_ylabel('mse_gap')
    ax.set_title('Performance Gap (multi vs single)')
    ax.grid(True, alpha=0.3)

    # Panel 4: switch_rate
    ax = axes[1, 1]
    ax.plot(drifts, switch_rate, 'c-', linewidth=1.5, alpha=0.7)
    ax.scatter(drifts, switch_rate, c='cyan', s=20, alpha=0.5)
    ax.set_xlabel('drift')
    ax.set_ylabel('switch_rate')
    ax.set_title('Routing Switch Rate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Band structure plot saved to {output}")


if __name__ == "__main__":
    print("=" * 60)
    print("E(drift) — Band Structure Emergence Function")
    print(f"Dense sweep: {len(DRIFT_LIST)} drift values in [{DRIFT_LIST[0]:.4f}, {DRIFT_LIST[-1]:.4f}]")
    print("=" * 60)

    results = run_dense_drift_sweep(DRIFT_LIST)

    drifts = np.array([r["drift"] for r in results])
    M_varz = np.array([r["var_z"] for r in results])

    fit_params = fit_gaussian_mixture(drifts, M_varz)

    print(f"\nDetected {len(fit_params)} emergence windows:")
    for i, (d_i, a_i, sigma_i) in enumerate(fit_params):
        print(f"  Window {i+1}: center={d_i:.4f}  amplitude={a_i:.4f}  width={sigma_i:.4f}")

    plot_band_structure(drifts, results, fit_params)

    export = {
        "n_points": len(results),
        "windows": [
            {"center": float(d), "amplitude": float(a), "sigma": float(s)}
            for d, a, s in fit_params
        ],
        "raw": results,
    }
    with open("band_structure_fit.json", "w") as f:
        json.dump(export, f, indent=1)
    print("Fit saved to band_structure_fit.json")
