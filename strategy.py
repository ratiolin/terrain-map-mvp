"""
Stability Strategy Module.

Operational controller that:
  - Reads stability_regions.json → is_stable(drift), danger zones
  - Reads eta_max_curve.json → eta_max(drift) interpolation
  - Reads stability_function.json → dS_dd early warning
  - Provides: control policy, hard limits, behavior mapping

Usage:
  from strategy import StabilityController
  ctrl = StabilityController()
  eta_safe = ctrl.clamp_eta(requested_eta, drift)
  if ctrl.is_danger_zone(drift):
      action = "fallback"
"""
import json
import numpy as np


class StabilityController:
    def __init__(self,
                 regions_file="stability_regions.json",
                 eta_curve_file="eta_max_curve.json",
                 S_file="stability_function.json"):
        # ── Load data ────────────────────────────────────────────────
        with open(regions_file) as f:
            regions_data = json.load(f)

        self.stable_regions = [(r["drift_start"], r["drift_end"])
                                for r in regions_data.get("stable_regions", [])]
        self.unstable_regions = [(r["drift_start"], r["drift_end"])
                                  for r in regions_data.get("unstable_regions", [])]
        self.threshold = regions_data.get("threshold", 0.5)

        with open(eta_curve_file) as f:
            eta_data = json.load(f)
        self._drifts_eta = np.array(eta_data["drift"])
        self._eta_vals = np.array(eta_data["eta_max"])

        with open(S_file) as f:
            S_data = json.load(f)
        self._drifts_S = np.array(S_data["drift"])
        self._dS_dd = np.array(S_data["dS_dd"])
        self._S = np.array(S_data["S"])

    # ── Region queries ───────────────────────────────────────────────
    def is_stable(self, drift):
        """Check if drift falls in any stable region."""
        return any(start <= drift <= end for start, end in self.stable_regions)

    def is_unstable(self, drift):
        """Check if drift falls in any unstable (danger) region."""
        return any(start <= drift <= end for start, end in self.unstable_regions)

    def which_region(self, drift):
        """Return region type: 'stable', 'unstable', 'unknown'."""
        if self.is_stable(drift):
            return "stable"
        elif self.is_unstable(drift):
            return "unstable"
        return "unknown"

    def get_strongest_stable_region(self, drift):
        """Find if drift is in a strong stable region (near band center)."""
        for start, end in self.stable_regions:
            if start <= drift <= end:
                width = end - start
                center = (start + end) / 2
                # "Strong" = within inner 60% of the stable band
                inner_start = center - 0.3 * width
                inner_end = center + 0.3 * width
                if inner_start <= drift <= inner_end:
                    return True
        return False

    # ── eta_max interpolation ────────────────────────────────────────
    def eta_max(self, drift):
        """Interpolate eta_max(drift) from stored curve."""
        return float(np.interp(drift, self._drifts_eta, self._eta_vals))

    def clamp_eta(self, requested_eta, drift):
        """Apply stability limit: eta = min(requested_eta, eta_max(drift))."""
        limit = self.eta_max(drift)
        return min(requested_eta, limit)

    # ── Danger zone protection ────────────────────────────────────────
    def is_danger_zone(self, drift):
        """True if drift is in an unstable region (hard limit)."""
        return self.is_unstable(drift)

    def protect(self, eta, drift):
        """
        Apply hard protection:
          - In unstable region: eta *= 0.5 (aggressive reduction)
          - In stable region:   eta stays as-is
          - Unknown:            slight reduction (eta *= 0.8)
        """
        if self.is_unstable(drift):
            return eta * 0.5
        elif self.is_stable(drift):
            return eta
        else:
            return eta * 0.8

    # ── Early warning via S(d) ────────────────────────────────────────
    def dS_dd_at(self, drift):
        """Derivative of stability functional at given drift."""
        return float(np.interp(drift, self._drifts_S, self._dS_dd))

    def S_at(self, drift):
        """Stability functional value at given drift."""
        return float(np.interp(drift, self._drifts_S, self._S))

    def should_warn(self, drift):
        """Early warning: dS/dd < 0 means stability is deteriorating."""
        return self.dS_dd_at(drift) < 0

    # ── Behavior policy ───────────────────────────────────────────────
    def policy(self, drift, requested_eta=0.5):
        """
        Map drift to behavior strategy.
        Returns dict with: action, eta_used, protected, warning, region.
        """
        region = self.which_region(drift)
        eta_limit = self.eta_max(drift)
        eta_clamped = self.clamp_eta(requested_eta, drift)
        eta_final = self.protect(eta_clamped, drift)
        warning = self.should_warn(drift)
        strong = self.get_strongest_stable_region(drift)

        if self.is_unstable(drift):
            action = "avoid / fallback"
        elif strong:
            action = "increase eta"
        else:
            action = "normal"

        return {
            "drift": float(drift),
            "region": region,
            "action": action,
            "eta_requested": float(requested_eta),
            "eta_limit": float(eta_limit),
            "eta_used": float(eta_final),
            "protected": self.is_unstable(drift),
            "hard_limit_active": self.is_danger_zone(drift),
            "warning": warning,
            "strong_stable": strong,
        }

    # ── Online validation scan ───────────────────────────────────────
    def scan(self, drift_grid=None, requested_eta=0.5):
        """Full scan across drift range, returning policy for each point."""
        if drift_grid is None:
            drift_grid = np.linspace(0.001, 0.2, 100)

        results = []
        for d in drift_grid:
            results.append(self.policy(float(d), requested_eta))

        return results


# ── Report ───────────────────────────────────────────────────────────
def print_report(scan_results):
    """Print a summary table from scan results."""
    print(f"\n{'='*70}")
    print(f"  ONLINE VALIDATION SCAN")
    print(f"{'='*70}")
    header = f"{'drift':>8} {'region':>10} {'action':<20} {'eta_used':>8} {'protected':>10} {'warn':>6} {'strong':>8}"
    print(header)
    print("-" * 70)

    n_protected = 0
    n_warned = 0
    n_strong = 0
    for r in scan_results:
        print(f"{r['drift']:>8.4f} {r['region']:>10} {r['action']:<20} "
              f"{r['eta_used']:>8.3f} {str(r['protected']):>10} "
              f"{str(r['warning']):>6} {str(r['strong_stable']):>8}")
        if r['protected']:
            n_protected += 1
        if r['warning']:
            n_warned += 1
        if r['strong_stable']:
            n_strong += 1

    print("-" * 70)
    print(f"  Protected: {n_protected}/{len(scan_results)}  "
          f"Warned: {n_warned}/{len(scan_results)}  "
          f"Strong-stable: {n_strong}/{len(scan_results)}")


if __name__ == "__main__":
    ctrl = StabilityController()

    print("=" * 60)
    print("STABILITY STRATEGY MODULE — Online Validation")
    print("=" * 60)

    print(f"\nLoaded:")
    print(f"  Stable regions:   {len(ctrl.stable_regions)} bands")
    print(f"  Unstable regions: {len(ctrl.unstable_regions)} bands")
    print(f"  eta_max range:    [{ctrl._eta_vals.min():.3f}, {ctrl._eta_vals.max():.3f}]")
    print(f"  dS/dd range:      [{ctrl._dS_dd.min():.3f}, {ctrl._dS_dd.max():.3f}]")

    # Check a few key drift values
    test_drifts = [0.01, 0.025, 0.06, 0.11, 0.15, 0.18]
    print(f"\n{'─'*60}")
    print(f"  Key drift decision points:")
    for d in test_drifts:
        p = ctrl.policy(d)
        print(f"  drift={d:.4f} → region={p['region']:>10} action={p['action']:<20} "
              f"eta={p['eta_used']:.3f} protected={p['protected']} warn={p['warning']}")

    # Full scan
    scan = ctrl.scan(requested_eta=0.5)
    print_report(scan)

    # Export scan results
    with open("strategy_scan.json", "w") as f:
        json.dump(scan, f, indent=1)
    print("\nScan exported to strategy_scan.json")
