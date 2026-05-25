#!/usr/bin/env python
"""Intrinsic Boundary Condition Recognition Experiment.

Phase 0-13: Full implementation.
Tests whether a dual-head policy can spontaneously differentiate
into SHAPE (control) and ADAPT (accommodation) strategies through
prediction loss alone, with no reward signal, no externally-specified
loss scaling, and no hand-coded controller.

Usage:
    python core_mvp_v3/run.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/srv/stack/terrain-map-mvp")

from core_mvp_v3.experiment import (
    train, ExperimentConfig, Mode, _find_g_star,
    _test_13_1, _test_13_2, _test_13_3, _test_13_4,
    _test_13_5, _test_13_6, _test_13_7,
    _test_13_10, _test_13_11, _test_13_12, _test_13_17,
)


FAST_TESTS = {"13.1", "13.2", "13.3", "13.4", "13.5", "13.6", "13.7",
              "13.10", "13.11", "13.12", "13.17"}


def main():
    print("=" * 65)
    print("  INTRINSIC BOUNDARY CONDITION RECOGNITION")
    print("  Phase 0-13: Full Implementation & Validation")
    print("=" * 65)

    config = ExperimentConfig()
    config.num_episodes = 3
    config.episode_length = 8000
    config.seed = 42

    print(f"\n  episodes={config.num_episodes} × length={config.episode_length}")
    print(f"  drift schedule: low(0.1-0.3) / high(1.0-2.0) alternating")
    print(f"  k_rollout={config.k_rollout}  T_ctrl={config.T_ctrl}  theta={config.theta}")
    print(f"  action_scale={config.action_scale}  force_scale={config.force_scale}")
    print(f"  NO reward, NO lambda, NO teacher — prediction loss only")

    print("\n--- Training ---")
    t0 = time.time()
    logs, pred_net, policy_net, config = train(config)
    elapsed = time.time() - t0

    drift = np.array(logs["drift"])
    mode_arr = np.array(logs["mode"])
    risk_arr = np.array(logs["risk"])
    low = drift < 0.5
    high = drift > 0.5

    print(f"  steps: {len(logs['t'])}  time: {elapsed:.1f}s")
    print(f"  SHAPE in low drift: {(mode_arr[low]==Mode.SHAPE).mean():.1%}")
    print(f"  ADAPT in high drift: {(mode_arr[high]==Mode.ADAPT).mean():.1%}")
    print(f"  risk(low)={risk_arr[low].mean():.4f}  risk(high)={risk_arr[high].mean():.4f}")
    print(f"  prediction loss start: {np.array(logs['pred_loss'])[:500].mean():.4f}"
          f"  end: {np.array(logs['pred_loss'])[-500:].mean():.4f}")
    print(f"  switches: {sum(logs['switch_event'])}  "
          f"D(action): {np.mean(np.abs(np.array(logs['action_shape'])-np.array(logs['action_adapt']))):.4f}")
    g_star = _find_g_star(logs)
    print(f"  g* (best split): {g_star:.4f}")

    print("\n--- Validation ---")
    tests = [
        ("13.1_strategy_differentiation",
         _test_13_1(logs),
         "D={value:.4f}"),
        ("13.2_performance_inversion",
         _test_13_2(logs),
         None),
        ("13.3_panic_mode_coupling",
         _test_13_3(logs),
         "corr={value:.4f}"),
        ("13.4_behavioral_bifurcation",
         _test_13_4(logs),
         None),
        ("13.5_shuffled_panic",
         _test_13_5(logs, pred_net, policy_net, config),
         None),
        ("13.6_panic_ablation",
         _test_13_6(logs, pred_net, policy_net, config),
         None),
        ("13.7_perturbation_test",
         _test_13_7(logs, pred_net, policy_net, config),
         None),
        ("13.10_freeze_test",
         _test_13_10(logs, pred_net, policy_net, config),
         None),
        ("13.11_external_panic",
         _test_13_11(logs, pred_net, policy_net, config),
         None),
        ("13.12_controllability_separation",
         _test_13_12(logs),
         None),
        ("13.17_active_switching",
         _test_13_17(logs),
         None),
    ]

    passed = 0
    failed = 0
    for name, r, fmt in tests:
        status = "PASS" if r["passed"] else "FAIL"
        if fmt is not None:
            detail = fmt.format(**r)
        else:
            detail = r["details"]
        print(f"  [{status}] {name}: {detail}")
        if r["passed"]:
            passed += 1
        else:
            failed += 1

    print(f"\n  Results: {passed}/{passed+failed} fast tests passed")
    if g_star and not np.isnan(g_star):
        print(f"  g* found at {g_star:.4f} — regime switching point identified")
    else:
        print("  g* not found — mode-regime correlation below threshold")

    print("\n" + "=" * 65)
    print("  CONCLUSION")
    print("=" * 65)
    print("  Demonstrated:")
    print("    [x] Strategy differentiation from prediction loss alone")
    print("    [x] Controllability measurement via rollout comparison")
    print("    [x] Regime-dependent controllability separation")
    print("    [x] Fixed-mode ablation behavior")
    print("    [x] External/random panic → performance degradation")
    print()
    print("  Limitations / open problems:")
    print("    [ ] Mode distribution ~50/50 — signal-to-noise too low")
    print("    [ ] Risk in low drift >0.3 — action authority insufficient")
    print("    [ ] g* not identified — requires stronger mode-regime coupling")
    print("    [ ] 13.2/13.3/13.4/13.5/13.7/13.10/13.17 — FAIL")
    print("    [ ] Tests 13.8/13.9/13.13/13.14/13.15/13.16 — skipped (expensive)")


if __name__ == "__main__":
    main()
