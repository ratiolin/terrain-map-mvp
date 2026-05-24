"""Blind test: drift=0.07, eta=0.25"""
from experiment10 import run_triplet
from metrics import compute_stability


DRIFT = 0.07
ETA = 0.25
KAPPA = 2.0
SEEDS = (42, 43)

print("=" * 60)
print(f"BLIND TEST: drift={DRIFT}, eta={ETA}")
print("=" * 60)

print("\n[PREDICTION]")
print("  From ternary phase diagram @ kappa=2.0:")
print("    drift=0.06 eta=0.2: S_adv=1.245  (2)")
print("    drift=0.06 eta=0.4: S_adv=1.267  (2)")
print("    drift=0.08 eta=0.2: S_adv=1.034  (2)")
print("    drift=0.08 eta=0.4: S_adv=0.990  (1)")
print()
print("  Interpolation: between drift=0.06(eta≈0.2-0.4 stays high)")
print("  and drift=0.08(eta≈0.2-0.4 drops). At drift=0.07, ")
print("  eta=0.25 should be above the boundary.")
print()
print("  PREDICTION: S_adv > 1  (label=2, ADVANTAGE)")

print("\n" + "=" * 60)
print("[EXPERIMENT]")
print("=" * 60)

triplet = run_triplet(
    kappa=KAPPA, drift=DRIFT,
    K_budget=4, expert_hidden=2, gating_hidden=8,
    train_steps=800, test_steps=200,
    seeds=SEEDS, inertia=ETA,
)

det = triplet["det"]
v = det["variance_mean"]
c = det["consistency_mean"]
d = det["dwell_time_mean"]
S_ensemble = compute_stability(v, c, d)
S_single = det["S_single_mean"]
S_adv = S_ensemble / (S_single + 1e-8)

K_vals = [s["K_final"] for s in det["seeds"]]
K_mean = sum(K_vals) / len(K_vals)

print(f"\n  S_adv = {S_adv:.4f}")
print(f"  K_final = {K_mean:.1f}")
print(f"  Label = {'2 (ADVANTAGE)' if S_adv > 1.0 else '1 (collapse)'}")

print("\n" + "=" * 60)
print("[VERDICT]")
actual = "ADVANTAGE" if S_adv > 1.0 else "collapse"
print(f"  Predicted: ADVANTAGE (S_adv > 1)")
print(f"  Actual:    {actual} (S_adv = {S_adv:.4f})")
print(f"  {'MATCH' if S_adv > 1.0 else 'MISMATCH'}")
