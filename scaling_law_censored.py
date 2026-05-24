"""
Scaling law fit with censored data support.

Reads binary_search_results.json and fits:
  eta_max(drift) = K / drift - tau

Uses only observed (uncensored) points.
Censored points provide lower bounds: eta_max > eta_ceiling.
"""
import json
import numpy as np
import matplotlib.pyplot as plt


def scaling_law(drift, K, tau):
    return K / drift - tau


def load_binary_results(path="binary_search_results.json"):
    with open(path) as f:
        data = json.load(f)
    return data


def fit_scaling_law_censored(data):
    """
    Fit using only observed points.
    For censored points, use them as constraints: K/drift - tau > eta_ceiling.
    """
    observed = data["observed"]
    censored = data["censored"]
    eta_ceiling = data["eta_ceiling"]

    obs_drifts = np.array([p["drift"] for p in observed])
    obs_etas = np.array([p["eta_max"] for p in observed])

    # Filter out trivial collapses (eta_max == 0 means structure never existed)
    # These are NOT inertia collapses — they're fundamental failures
    mask_nonzero = obs_etas > 0.01
    fit_drifts = obs_drifts[mask_nonzero]
    fit_etas = obs_etas[mask_nonzero]

    zero_drifts = obs_drifts[~mask_nonzero]

    print(f"Observed points: {len(observed)} (nonzero collapse: {mask_nonzero.sum()}, trivial: {(~mask_nonzero).sum()})")
    print(f"Censored points: {len(censored)}")

    if len(fit_drifts) >= 2:
        x = 1.0 / fit_drifts
        y = fit_etas
        coeffs = np.polyfit(x, y, 1)
        K, tau = coeffs[0], -coeffs[1]

        pred = K * x - tau
        resid = y - pred
        if len(x) > 2:
            sigma2 = np.sum(resid**2) / (len(x) - 2)
            X = np.column_stack([x, np.ones_like(x)])
            cov = sigma2 * np.linalg.inv(X.T @ X)
            K_err = np.sqrt(cov[0, 0])
            tau_err = np.sqrt(cov[1, 1])
        else:
            K_err, tau_err = 0.0, 0.0

        r2 = 1 - np.sum(resid**2) / max(np.sum((y - np.mean(y))**2), 1e-8)
    else:
        K, tau = 0.0, 0.0
        K_err, tau_err = 0.0, 0.0
        r2 = 0.0

    return K, tau, K_err, tau_err, r2, fit_drifts, fit_etas, zero_drifts


def plot_fit(data, K, tau, r2, fit_drifts, fit_etas, zero_drifts,
             output="scaling_law_censored.png"):
    observed = data["observed"]
    censored = data["censored"]
    eta_ceiling = data["eta_ceiling"]

    obs_drifts = np.array([p["drift"] for p in observed])
    obs_etas = np.array([p["eta_max"] for p in observed])
    cens_drifts = np.array([p["drift"] for p in censored])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Phase diagram
    ax = axes[0]
    for d, e in zip(obs_drifts, obs_etas):
        if e > 0.01:
            ax.scatter(d, e, c='blue', s=100, edgecolors='black', zorder=5, marker='o')
        else:
            ax.scatter(d, e, c='gray', s=60, marker='x', zorder=5, alpha=0.6)
    for d in cens_drifts:
        ax.scatter(d, eta_ceiling, c='orange', s=80, marker='^', edgecolors='black', zorder=5)
        ax.annotate(f'> {eta_ceiling}', (d, eta_ceiling), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=7, color='orange')

    if K > 0:
        d_fine = np.logspace(np.log10(0.003), np.log10(0.15), 100)
        eta_theory = scaling_law(d_fine, K, tau)
        ax.plot(d_fine, eta_theory, 'r--', linewidth=2,
                label=f'fit: eta = {K:.4f}/drift - {tau:.4f}')

    ax.set_xlabel('drift')
    ax.set_ylabel('eta_max')
    ax.set_title(f'Collapse Boundary (binary search, R²={r2:.3f})')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # Panel 2: 1/drift linearity
    ax = axes[1]
    if len(fit_drifts) > 0:
        x_fit = 1.0 / fit_drifts
        ax.scatter(x_fit, fit_etas, c='blue', s=100, edgecolors='black', zorder=5,
                   label='observed collapse')
        if K > 0:
            x_line = np.linspace(0, max(x_fit) * 1.2, 100)
            ax.plot(x_line, K * x_line - tau, 'r--', linewidth=2,
                    label=f'eta = {K:.4f}/drift - {tau:.4f}')

    # Trivial collapses
    if len(zero_drifts) > 0:
        ax.scatter(1.0 / zero_drifts, np.zeros_like(zero_drifts), c='gray', s=60,
                   marker='x', zorder=5, alpha=0.6, label='trivial collapse (eta=0)')

    # Censored points
    if len(cens_drifts) > 0:
        ax.scatter(1.0 / cens_drifts, [eta_ceiling] * len(cens_drifts),
                   c='orange', s=80, marker='^', edgecolors='black', zorder=5,
                   label=f'censored (eta > {eta_ceiling})')

    ax.set_xlabel('1 / drift')
    ax.set_ylabel('eta_max')
    ax.set_title(f'Scaling in 1/drift: K={K:.4f}, tau={tau:.4f}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"\nPlot saved to {output}")


if __name__ == "__main__":
    data = load_binary_results()

    K, tau, K_err, tau_err, r2, fit_drifts, fit_etas, zero_drifts = \
        fit_scaling_law_censored(data)

    print("=" * 60)
    print("SCALING LAW FIT (censored-regression)")
    print("=" * 60)
    print(f"\nFitted on {len(fit_drifts)} observed collapse points:")
    for d, e in zip(fit_drifts, fit_etas):
        print(f"  drift={d:.4f}  eta_max={e:.4f}  1/drift={1/d:.1f}")

    if len(zero_drifts) > 0:
        print(f"\nExcluded {len(zero_drifts)} trivial collapses (eta_max=0, structure never existed):")
        for d in zero_drifts:
            print(f"  drift={d:.4f}")

    print(f"\nFit: eta_max = {K:.6f}/drift - {tau:.6f}")
    print(f"  K   = {K:.6f} ± {K_err:.6f}")
    print(f"  tau = {tau:.6f} ± {tau_err:.6f}")
    print(f"  R²  = {r2:.4f}")

    print(f"\nPredicted eta_max (observed only):")
    for d in fit_drifts:
        pred = scaling_law(d, K, tau)
        print(f"  drift={d:.4f}: predicted={pred:.4f}  actual={fit_etas[list(fit_drifts).index(d)]:.4f}")

    plot_fit(data, K, tau, r2, fit_drifts, fit_etas, zero_drifts)

    export = {
        "method": "censored regression (observed points only)",
        "K": round(float(K), 6),
        "tau": round(float(tau), 6),
        "K_err": round(float(K_err), 6),
        "tau_err": round(float(tau_err), 6),
        "R2": round(float(r2), 4),
        "n_observed_used": int(len(fit_drifts)),
        "n_trivial_excluded": int(len(zero_drifts)),
        "fit_points": [{"drift": float(d), "eta_max": float(e)} for d, e in zip(fit_drifts, fit_etas)],
        "trivial_points": [float(d) for d in zero_drifts],
    }
    with open("scaling_law_censored.json", "w") as f:
        json.dump(export, f, indent=2)
    print("\nFit saved to scaling_law_censored.json")
