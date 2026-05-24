"""Stage 10 Blind Test v2 — dual-boundary interval prediction."""
import numpy as np

from experiment10 import run_triplet
from analysis_stage10 import compute_stability_curve


TAU_LOW = 0.10
TAU_HIGH = 0.17

TAU_RESP_KNOWN = {
    0.005: 2.6,
    0.020: 7.6,
    0.080: 4.1,
}


def classify(R):
    if R < TAU_LOW:
        return "R < τ_low → S_adv < 1 (too static, no structure advantage)"
    elif R < TAU_HIGH:
        return "τ_low < R < τ_high → S_adv > 1 (emergence window)"
    else:
        return "R > τ_high → S_adv < 1 (collapse)"


if __name__ == "__main__":
    print("=" * 60)
    print("BLIND TEST v2: Dual-boundary interval prediction")
    print("=" * 60)

    print(f"\nτ_low={TAU_LOW:.2f}   τ_high={TAU_HIGH:.2f}")
    print(f"Known: R(0.005)=0.013  R(0.01)=0.043  R(0.02)=0.152  R(0.08)=0.330\n")

    # ── Step 2-3: predict for drift=0.008 ──
    d_blind = 0.008
    tau_r_est = float(np.interp(d_blind,
                                list(TAU_RESP_KNOWN.keys()),
                                list(TAU_RESP_KNOWN.values())))
    tau_d = 1.0 / d_blind
    R_est = tau_r_est / tau_d
    prediction = classify(R_est)

    print(f"[PREDICTION]")
    print(f"  drift={d_blind}  τ_resp_est={tau_r_est:.1f}  "
          f"τ_drift={tau_d:.1f}  R={R_est:.4f}")
    print(f"  → {prediction}\n")

    # ── Step 4: run experiment ──
    print("=" * 60)
    print(f"[EXPERIMENT] Running drift={d_blind}")
    print("=" * 60)

    kappas = [1.0, 2.0, 4.0]
    s_adv_all = []
    for kappa in kappas:
        key = f"k{kappa}_d{d_blind}"
        print(f"\n--- kappa={kappa}, drift={d_blind} ---")
        triplet = run_triplet(
            kappa=kappa, drift=d_blind,
            K_budget=4, expert_hidden=2, gating_hidden=8,
            train_steps=1200, test_steps=300,
            seeds=(42, 43, 44),
        )
        _, _, s_adv_blind = compute_stability_curve({key: triplet})
        s_adv_all.append(s_adv_blind[0])
        print(f"  S_adv = {s_adv_blind[0]:.4f}")

    actual = float(np.mean(s_adv_all))
    actual_label = "S_adv > 1" if actual > 1.0 else "S_adv < 1"
    pred_label = "S_adv > 1" if "> 1" in prediction else "S_adv < 1"
    match = actual_label == pred_label

    print(f"\n{'='*60}")
    print(f"[VERDICT]")
    print(f"  Predicted: {pred_label}")
    print(f"  Actual:    {actual_label}  (S_adv = {actual:.4f})")
    print(f"  {'MATCH' if match else 'MISMATCH'}")

    if match:
        print(f"\n  ✅  Dual-boundary interval model holds!")
        print(f"      R < τ_low : collapse at too-slow drift")
        print(f"      τ_low < R < τ_high : emergence advantage")
        print(f"      R > τ_high : collapse at too-fast drift")
    else:
        print(f"\n  ❌  τ_response is not sufficient — need a second variable.")
        print(f"      The model may need state-dependent modulation or"
              f"      an additional timescale beyond τ_response alone.")
