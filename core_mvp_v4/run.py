#!/usr/bin/env python3
"""
V4 Experiment Runner

Two independent pipelines:
  Original:  3C -> 3A -> 3B -> Part1 -> Part2
  New:       S0 -> P1-Refocus -> P2-Continuous -> P3-Conditional
  All:       Original -> New (if S0 passes)

Usage:
    uv run python core_mvp_v4/run.py                        # All pipelines
    uv run python core_mvp_v4/run.py --pipeline original     # Original only
    uv run python core_mvp_v4/run.py --pipeline new          # New only
    uv run python core_mvp_v4/run.py --part S0                # Single part
    uv run python core_mvp_v4/run.py --seeds 8 --episodes 3 --steps 2000
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_original_pipeline(seeds, n_episodes=3, episode_length=2000):
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timing = {}
    train_cfg = dict(n_episodes=n_episodes, episode_length=episode_length)

    print("=" * 60)
    print("ORIGINAL PIPELINE: 3C → 3A → 3B → P1 → P2")
    print(f"Seeds: {seeds}, Ep: {n_episodes}, Steps: {episode_length}")
    print("=" * 60)

    print("\n[1/5] Part 3C: Mechanism Destruction...")
    t0 = time.time()
    from core_mvp_v4.experiments.part3c_mechanism import run_part3c_mechanism
    r = run_part3c_mechanism(n_seeds=seeds, **train_cfg)
    with open(os.path.join(results_dir, "part3c_mechanism.json"), "w") as f:
        json.dump(r, f, indent=2, default=str)
    t1 = time.time(); timing["3C"] = t1 - t0
    for mode, d in r.items():
        s = d.get("summary")
        if s: print(f"  {mode}: k80={s.get('k80')}, R2={s.get('R2')}, fail={s.get('failures',{}).get('any_fail')}")
    print(f"  Done in {t1-t0:.1f}s")

    print("\n[2/5] Part 3A: SNR Boundary...")
    t0 = time.time()
    from core_mvp_v4.experiments.part3a_snr import run_part3a_snr
    r = run_part3a_snr(n_seeds=seeds, **train_cfg)
    with open(os.path.join(results_dir, "part3a_snr_boundary.json"), "w") as f:
        json.dump(r, f, indent=2, default=str)
    t1 = time.time(); timing["3A"] = t1 - t0
    print(f"  sigma_crit={r.get('sigma_critical',{}).get('sigma_crit')} | {t1-t0:.1f}s")

    print("\n[3/5] Part 3B: Capacity Boundary...")
    t0 = time.time()
    from core_mvp_v4.experiments.part3b_capacity import run_part3b_capacity
    r = run_part3b_capacity(n_seeds=seeds, **train_cfg)
    with open(os.path.join(results_dir, "part3b_capacity_boundary.json"), "w") as f:
        json.dump(r, f, indent=2, default=str)
    t1 = time.time(); timing["3B"] = t1 - t0
    print(f"  hdim_crit={r.get('hidden_dim_critical',{}).get('hidden_dim_crit')} | {t1-t0:.1f}s")

    print("\n[4/5] Part 1: Dimension Pressure...")
    t0 = time.time()
    from core_mvp_v4.experiments.part1_dimension import run_part1_dimension_pressure
    r = run_part1_dimension_pressure(n_seeds=seeds, **train_cfg)
    with open(os.path.join(results_dir, "part1_dimension_pressure.json"), "w") as f:
        json.dump(r, f, indent=2, default=str)
    t1 = time.time(); timing["P1"] = t1 - t0
    for dk, d in r.items():
        s = d.get("summary")
        if s: print(f"  d={s['d']}: k80={s['k80']}, R2={s['R2']}, align_gt={s['alignment_gt']}")
    print(f"  Done in {t1-t0:.1f}s")

    print("\n[5/5] Part 2: Continuous Drift...")
    t0 = time.time()
    from core_mvp_v4.experiments.part2_drift import run_part2_continuous_drift
    r = run_part2_continuous_drift(n_seeds=seeds, **train_cfg)
    with open(os.path.join(results_dir, "part2_continuous_drift.json"), "w") as f:
        json.dump(r, f, indent=2, default=str)
    t1 = time.time(); timing["P2"] = t1 - t0
    agg = r.get("aggregate", {})
    print(f"  k80={agg.get('k80_mean')}±{agg.get('k80_std')}, R2(g)={agg.get('R2_g')} | {t1-t0:.1f}s")

    total = sum(timing.values())
    print(f"\nOriginal pipeline complete. Total: {total:.1f}s")
    return timing


def run_new_pipeline(seeds, n_episodes=3, episode_length=2000):
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timing = {}
    train_cfg = dict(n_episodes=n_episodes, episode_length=episode_length)

    print("=" * 60)
    print("NEW PIPELINE: S0 → P1-Refocus → P2-Continuous → P3-Conditional")
    print(f"Seeds: {seeds}, Ep: {n_episodes}, Steps: {episode_length}")
    print("=" * 60)

    # ---- S0: Signal Learnability ----
    print("\n[1/4] S0: Signal Learnability Calibration...")
    t0 = time.time()
    from core_mvp_v4.experiments.s0_learnability import run_s0_learnability
    r_s0 = run_s0_learnability(n_seeds=seeds,
                               n_episodes=n_episodes, episode_length=episode_length)
    with open(os.path.join(results_dir, "s0_learnability.json"), "w") as f:
        json.dump(r_s0, f, indent=2, default=str)
    t1 = time.time(); timing["S0"] = t1 - t0
    bc = r_s0.get("best_config", {})
    print(f"  best: hd={bc.get('hidden_dim')}, steps={bc.get('step_multiplier')}x, "
          f"lambda={bc.get('lambda_signal')}, R2={r_s0.get('best_R2')}")
    print(f"  conclusion: {r_s0.get('conclusion')}")

    best_R2 = r_s0.get("best_R2", 0)
    if best_R2 < 0.2:
        print("\n*** S0 FAILED: R2 < 0.2. C(s) not learnable. Aborting new pipeline. ***")
        return timing

    bc_hd = bc.get("hidden_dim", 16)
    bc_lam = bc.get("lambda_signal", 0.5)
    bc_ep = bc.get("n_episodes", n_episodes)

    # ---- P1-Refocus: Semantic Degradation ----
    print("\n[2/4] P1-Refocus: Semantic Degradation Boundary...")
    t0 = time.time()
    from core_mvp_v4.experiments.p1_semantic import run_p1_semantic
    r_p1 = run_p1_semantic(n_seeds=seeds, hd=bc_hd,
                           n_episodes=bc_ep, episode_length=episode_length,
                           lambda_signal=bc_lam)
    with open(os.path.join(results_dir, "p1_semantic_degradation.json"), "w") as f:
        json.dump(r_p1, f, indent=2, default=str)
    t1 = time.time(); timing["P1-refocus"] = t1 - t0
    ba = r_p1.get("boundary_analysis", {})
    print(f"  boundary: {ba.get('boundary')}")
    for dk, d in r_p1.items():
        s = d.get("summary")
        if s: print(f"  d={s['d']}: align_gt={s['alignment_gt']}, R2={s['R2']}")
    print(f"  Done in {t1-t0:.1f}s")

    # ---- P2-Continuous: Reward-Semantic Phase ----
    print("\n[3/4] P2-Continuous: Reward-Semantic Phase Transition...")
    t0 = time.time()
    from core_mvp_v4.experiments.p2_phase import run_p2_phase
    r_p2 = run_p2_phase(n_seeds=seeds, hd=bc_hd,
                        n_episodes=bc_ep, episode_length=episode_length,
                        lambda_signal=bc_lam)
    with open(os.path.join(results_dir, "p2_reward_semantic_phase.json"), "w") as f:
        json.dump(r_p2, f, indent=2, default=str)
    t1 = time.time(); timing["P2-continuous"] = t1 - t0
    pa = r_p2.get("phase_analysis", {})
    print(f"  alpha_crit={pa.get('alpha_crit')}")
    print(f"  {pa.get('conclusion')}")
    print(f"  Done in {t1-t0:.1f}s")

    # ---- P3-Conditional: Conditional Trigger ----
    print("\n[4/4] P3-Conditional: Conditional Trigger Drift...")
    t0 = time.time()
    from core_mvp_v4.experiments.p3_conditional import run_p3_conditional
    r_p3 = run_p3_conditional(n_seeds=seeds, hd=bc_hd,
                              n_episodes=bc_ep, episode_length=episode_length,
                              lambda_signal=bc_lam)
    with open(os.path.join(results_dir, "p3_conditional_trigger.json"), "w") as f:
        json.dump(r_p3, f, indent=2, default=str)
    t1 = time.time(); timing["P3-conditional"] = t1 - t0
    agg = r_p3.get("aggregate", {})
    print(f"  pre_align={agg.get('pre_alignment')}, post_align={agg.get('post_alignment')}")
    print(f"  pre_R2={agg.get('pre_R2')}, post_R2={agg.get('post_R2')}")
    print(f"  {r_p3.get('conclusion')}")
    print(f"  Done in {t1-t0:.1f}s")

    total = sum(timing.values())
    print(f"\nNew pipeline complete. Total: {total:.1f}s")
    return timing


def main():
    parser = argparse.ArgumentParser(description="V4 Experiment Runner")
    parser.add_argument("--seeds", type=int, default=8,
                        help="Seeds per config (default: 8)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Training episodes (default: 3)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Steps per episode (default: 2000)")
    parser.add_argument("--pipeline", type=str, default="all",
                        choices=["all", "original", "new"],
                        help="Which pipeline to run")
    parser.add_argument("--part", type=str, default=None,
                        help="Run single part")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    part_map = {
        "3C": ("core_mvp_v4.experiments.part3c_mechanism", "run_part3c_mechanism",
               "part3c_mechanism.json"),
        "3A": ("core_mvp_v4.experiments.part3a_snr", "run_part3a_snr",
               "part3a_snr_boundary.json"),
        "3B": ("core_mvp_v4.experiments.part3b_capacity", "run_part3b_capacity",
               "part3b_capacity_boundary.json"),
        "P1": ("core_mvp_v4.experiments.part1_dimension", "run_part1_dimension_pressure",
               "part1_dimension_pressure.json"),
        "P2": ("core_mvp_v4.experiments.part2_drift", "run_part2_continuous_drift",
               "part2_continuous_drift.json"),
        "S0": ("core_mvp_v4.experiments.s0_learnability", "run_s0_learnability",
               "s0_learnability.json"),
        "P1R": ("core_mvp_v4.experiments.p1_semantic", "run_p1_semantic",
                "p1_semantic_degradation.json"),
        "P2C": ("core_mvp_v4.experiments.p2_phase", "run_p2_phase",
                "p2_reward_semantic_phase.json"),
        "P3": ("core_mvp_v4.experiments.p3_conditional", "run_p3_conditional",
               "p3_conditional_trigger.json"),
        "A": ("core_mvp_v4.experiments.exp_a_alpha_phase", "run_exp_a_alpha_phase",
              "p2_alpha_phase.json"),
        "B": ("core_mvp_v4.experiments.exp_b_semantic_breakdown", "run_exp_b_semantic_breakdown",
              "b_semantic_breakdown.json"),
        "C": ("core_mvp_v4.experiments.exp_c_forced_semantics", "run_exp_c_forced_semantics",
              "c_forced_semantics.json"),
        "S1": ("core_mvp_v4.experiments.step1_d8_alpha", "run_step1_d8_alpha_scan",
               "p2_d8_alpha_scan.json"),
        "S2": ("core_mvp_v4.experiments.step2_capacity_scaling", "run_step2_capacity_scaling",
               "capacity_scaling.json"),
        "S3": ("core_mvp_v4.experiments.step3_split_scaling", "run_step3_split_scaling",
               "split_encoder_scaling.json"),
        "S4": ("core_mvp_v4.experiments.step4_attention_guided", "run_step4_attention_guided",
               "attention_guided.json"),
        "G1": ("core_mvp_v4.experiments.g1_scaling_law", "run_g1_scaling_law",
               "g1_scaling_law.json"),
        "G2": ("core_mvp_v4.experiments.g2_minimal_bias", "run_g2_minimal_bias",
               "g2_minimal_bias.json"),
        "G3": ("core_mvp_v4.experiments.g3_control_loop", "run_g3_control_loop",
               "g3_control_loop.json"),
        "H1": ("core_mvp_v4.experiments.h1_complex_env", "run_h1_complex_env",
               "h1_complex_env.json"),
        "H2": ("core_mvp_v4.experiments.h2_ablation", "run_h2_ablation",
               "h2_ablation.json"),
        "H3": ("core_mvp_v4.experiments.h3_diagnostic", "run_h3_diagnostic",
               "h3_diagnostic.json"),
        "H4": ("core_mvp_v4.experiments.h4_coupling", "run_h4_coupling",
               "h4_coupling_phase.json"),
        "CA": ("core_mvp_v4.experiments.c_policy_vs_representation", "run_c_policy_vs_representation",
               "c_policy_vs_representation.json"),
        "CB": ("core_mvp_v4.experiments.b_sensitivity_map", "run_b_sensitivity_map",
               "b_sensitivity_map.json"),
        "CC": ("core_mvp_v4.experiments.a_nonlinear_decoding", "run_a_nonlinear_decoding",
               "a_nonlinear_decoding.json"),
        "T11": ("core_mvp_v4.experiments.t1_jacobian_spectrum", "run_t1_jacobian_spectrum",
                "t1_jacobian_spectrum.json"),
        "T12": ("core_mvp_v4.experiments.t1_jacobian_causal", "run_t1_jacobian_causal",
                "t1_jacobian_causal.json"),
        "T13": ("core_mvp_v4.experiments.t1_head_capacity", "run_t1_head_capacity",
                "t1_head_capacity.json"),
        "T24": ("core_mvp_v4.experiments.t2_dynamic_jacobian", "run_t2_dynamic_jacobian",
                "t2_dynamic_jacobian.json"),
        "T25": ("core_mvp_v4.experiments.t2_encoder_role", "run_t2_encoder_role",
                "t2_encoder_role.json"),
        "T36": ("core_mvp_v4.experiments.t3_mutual_information", "run_t3_mutual_information",
                "t3_mutual_information.json"),
        "T37": ("core_mvp_v4.experiments.t3_ntk", "run_t3_ntk",
                "t3_ntk.json"),
        "T38": ("core_mvp_v4.experiments.t3_cross_env", "run_t3_cross_env",
                "t3_cross_env.json"),
        "P01": ("core_mvp_v4.experiments.p0_strong_causal", "run_p0_strong_causal",
                "p0_strong_causal.json"),
        "P12": ("core_mvp_v4.experiments.p1_rank_scaling", "run_p1_rank_scaling",
                "p1_rank_scaling.json"),
        "P13": ("core_mvp_v4.experiments.p1_information_pathways", "run_p1_information_pathways",
                "p1_information_pathways.json"),
        "P24": ("core_mvp_v4.experiments.p2_rank_origin", "run_p2_rank_origin",
                "p2_rank_origin.json"),
        "P35": ("core_mvp_v4.experiments.p3_jacobian_phase_transition", "run_p3_jacobian_phase_transition",
                "p3_jacobian_phase_transition.json"),
        "N1": ("core_mvp_v4.experiments.new1_local_manifold", "run_new1_local_manifold",
               "new1_local_manifold.json"),
        "N2": ("core_mvp_v4.experiments.new2_nonlinear_direction", "run_new2_nonlinear_direction",
               "new2_nonlinear_direction.json"),
        "N3": ("core_mvp_v4.experiments.new3_encoder_intervention", "run_new3_encoder_intervention",
               "new3_encoder_intervention.json"),
        "N4": ("core_mvp_v4.experiments.new4_manifold_navigation", "run_new4_manifold_navigation",
               "new4_manifold_navigation.json"),
        "D1": ("core_mvp_v4.experiments.d1_multi_direction", "run_d1_multi_direction",
               "d1_multi_direction.json"),
        "D2": ("core_mvp_v4.experiments.d2_hessian_curvature", "run_d2_hessian_curvature",
               "d2_hessian_curvature.json"),
        "D3": ("core_mvp_v4.experiments.d3_fisher_spectrum", "run_d3_fisher_spectrum",
               "d3_fisher_spectrum.json"),
        "D4": ("core_mvp_v4.experiments.d4_fisher_intervention", "run_d4_fisher_intervention",
               "d4_fisher_intervention.json"),
        "D5": ("core_mvp_v4.experiments.d5_learned_variable_z", "run_d5_learned_variable_z",
               "d5_learned_variable_z.json"),
        "D6": ("core_mvp_v4.experiments.d6_manifold_learning", "run_d6_manifold_learning",
               "d6_manifold_learning.json"),
        "D7": ("core_mvp_v4.experiments.d7_nonlinear_intervention", "run_d7_nonlinear_intervention",
               "d7_nonlinear_intervention.json"),
        "E1": ("core_mvp_v4.experiments.e1_causal_z", "run_e1_causal_z",
               "e1_causal_z.json"),
        "E2": ("core_mvp_v4.experiments.e2_dynamic_z", "run_e2_dynamic_z",
               "e2_dynamic_z.json"),
        "E3": ("core_mvp_v4.experiments.e3_rl_control_z", "run_e3_rl_control_z",
               "e3_rl_control_z.json"),
        "FA": ("core_mvp_v4.experiments.f_trajectory_level", "run_f_trajectory_level",
               "f_trajectory_level.json"),
        "FB": ("core_mvp_v4.experiments.f_gradient_field", "run_f_gradient_field",
               "f_gradient_field.json"),
        "FC": ("core_mvp_v4.experiments.f_task_analysis", "run_f_task_analysis",
               "f_task_analysis.json"),
        "V41": ("core_mvp_v4.experiments.v4_1_geometry", "run_v4_1_geometry",
                "g1_geometry.json"),
        "V42": ("core_mvp_v4.experiments.v4_2_regularized", "run_v4_2_regularized",
                "g2_regularized.json"),
        "V43": ("core_mvp_v4.experiments.v4_3_policy", "run_v4_3_policy",
                "g3_policy.json"),
        "V44": ("core_mvp_v4.experiments.v4_4_topology", "run_v4_4_topology",
                "g4_topology.json"),
        "V45": ("core_mvp_v4.experiments.v4_5_multiagent", "run_v4_5_multiagent",
                "g5_multiagent.json"),
        "H1": ("core_mvp_v4.experiments.h1_field_policy_alignment", "run_h1_field_policy_alignment",
               "h1_field_policy_alignment.json"),
        "H2": ("core_mvp_v4.experiments.h2_field_evolution", "run_h2_field_evolution",
               "h2_field_evolution.json"),
        "H3": ("core_mvp_v4.experiments.h3_coordination_layer", "run_h3_coordination_layer",
               "h3_coordination_layer.json"),
        "I1": ("core_mvp_v4.experiments.i1_equivalence_class", "run_i1_equivalence_class",
               "i1_equivalence_class.json"),
        "I2": ("core_mvp_v4.experiments.i2_io_constraint", "run_i2_io_constraint",
               "i2_io_constraint.json"),
        "I3": ("core_mvp_v4.experiments.i3_training_dynamics", "run_i3_training_dynamics",
               "i3_training_dynamics.json"),
    }

    if args.part:
        if args.part not in part_map:
            print(f"Unknown part: {args.part}. Choose: {list(part_map.keys())}")
            sys.exit(1)
        mod_name, func_name, out_name = part_map[args.part]
        import importlib
        mod = importlib.import_module(mod_name)
        func = getattr(mod, func_name)
        print(f"Running {args.part} with {args.seeds} seeds, "
              f"{args.episodes} eps, {args.steps} steps...")
        result = func(n_seeds=args.seeds, n_episodes=args.episodes,
                      episode_length=args.steps)
        with open(os.path.join(results_dir, out_name), "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Done. Saved to {os.path.join(results_dir, out_name)}")
        return

    if args.pipeline in ("all", "original"):
        run_original_pipeline(args.seeds, args.episodes, args.steps)

    if args.pipeline in ("all", "new"):
        run_new_pipeline(args.seeds, args.episodes, args.steps)


if __name__ == "__main__":
    main()
