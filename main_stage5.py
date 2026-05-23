from env import Env
from env_double_well import DoubleWellEnv, TripleWellEnv
from agent import Agent
from controller import GatingGrowthController, Controller as BaseCtrl
from loop_multi import experiment_soft, run_soft
from loop import run
from metrics import Metrics
from analyze import classify, is_stable, rollout, all_separated

import random
import numpy as np
import torch


def reset_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def test_cartpole_soft():
    print("=" * 50)
    print("TEST 1: CartPole — Gating (K=1)")
    print("=" * 50)

    env = Env()
    agent = Agent(obs_dim=4, act_dim=2)
    ctrl = GatingGrowthController(env_type="cartpole")

    hist, _ = experiment_soft(env, agent, ctrl, steps=10000)

    print("Classification:", classify(hist))
    print("Model count:", ctrl.n_models())
    print("Gating output dim:", ctrl.gating.output.out_features)
    assert ctrl.n_models() == 1, "CartPole should maintain single model"
    assert ctrl.gating.output.out_features == 1, "Gating dim should match K"
    print("PASS: CartPole gating aligned with single model")
    print()
    reset_seed()


def test_double_well_soft():
    print("=" * 50)
    print("TEST 2: DoubleWell — Soft routing + gating")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03, max_models=8)

    hist, _ = experiment_soft(env, agent, ctrl, steps=10000)

    n = ctrl.n_models()
    print("Classification:", classify(hist))
    print("Model count:", n)
    print("Gating dim:", ctrl.gating.output.out_features)
    print("Usage:", ctrl.usage)

    assert n > 1, "DoubleWell should grow beyond 1 model"
    assert ctrl.gating.output.out_features == n, "Gating dim must match K"

    # Check gating isn't collapsed (single model dominating)
    if n > 1:
        total_u = sum(ctrl.usage)
        max_frac = max(ctrl.usage) / max(1, total_u)
        collapsed = max_frac > 0.95
        print(f"Gating max model fraction: {max_frac:.3f}  collapsed: {collapsed}")

    print("PASS: DoubleWell soft routing + gating aligned")
    print()
    reset_seed()


def test_triple_well_soft():
    print("=" * 50)
    print("TEST 3: TripleWell — Multi-basin gating")
    print("=" * 50)

    env = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="triplewell",
                                  merge_thresh=0.2, prune_thresh=0.03, max_models=10)

    hist, _ = experiment_soft(env, agent, ctrl, steps=20000)

    n = ctrl.n_models()
    print("Classification:", classify(hist))
    print("Model count:", n)
    print("Gating dim:", ctrl.gating.output.out_features)
    print("All separated:", all_separated(ctrl.models, min_dist=0.3))
    print("Usage:", ctrl.usage)

    assert n >= 2, "TripleWell should have >= 2 models"
    assert ctrl.gating.output.out_features == n, "Gating dim must match K"

    # Collapse check
    if n > 1:
        total_usage = sum(ctrl.usage)
        max_frac = max(ctrl.usage) / max(1, total_usage) if total_usage > 0 else 0
        collapsed = max_frac > 0.95
        print(f"Gating max model fraction: {max_frac:.3f}  collapsed: {collapsed}")

    print("PASS: TripleWell gating + structure aligned")
    print()
    reset_seed()


def test_collapse_detection():
    print("=" * 50)
    print("TEST 4: Gating Collapse Detection")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03, max_models=8)

    hist, _ = experiment_soft(env, agent, ctrl, steps=10000)

    n = ctrl.n_models()
    gating_check = ctrl.gating.output.out_features == n

    # Check recent usage distribution
    recent_usage = ctrl.usage
    total = max(1, sum(recent_usage))
    fractions = [u / total for u in recent_usage]
    max_f = max(fractions) if fractions else 1.0
    collapsed = max_f > 0.95 if n > 1 else False

    print(f"K={n} max_frac={max_f:.3f} gating_aligned={gating_check}")
    if n > 1:
        print(f"  Model usage fractions: {[f'{f:.3f}' for f in fractions]}")
        if collapsed:
            print("  WARNING: gating collapsed — single model dominates")
    print(f"PASS: K={n}, gating aligned={gating_check}, collapse={collapsed}")
    print()
    reset_seed()


def test_soft_vs_hard():
    print("=" * 50)
    print("TEST 5: Soft vs Hard Routing Performance")
    print("=" * 50)

    noise = 0.02

    # Hard routing (from stage 4)
    from controller import GrowthController
    from loop_multi import experiment_growth

    env_hard = DoubleWellEnv(noise=noise, reset_pos=0.5)
    agent_hard = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl_hard = GrowthController(check_interval=200, env_type="doublewell",
                                 merge_thresh=0.2, prune_thresh=0.03, max_models=8)
    hist_hard, _ = experiment_growth(env_hard, agent_hard, ctrl_hard, steps=10000)
    err_hard = np.mean(hist_hard[-2000:])
    k_hard = ctrl_hard.n_models()

    # Soft routing
    env_soft = DoubleWellEnv(noise=noise, reset_pos=0.5)
    agent_soft = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl_soft = GatingGrowthController(check_interval=200, env_type="doublewell",
                                       merge_thresh=0.2, prune_thresh=0.03, max_models=8)
    hist_soft, _ = experiment_soft(env_soft, agent_soft, ctrl_soft, steps=10000)
    err_soft = np.mean(hist_soft[-2000:])
    k_soft = ctrl_soft.n_models()

    print(f"Hard routing: error={err_hard:.6f}  K={k_hard}")
    print(f"Soft routing: error={err_soft:.6f}  K={k_soft}")
    print(f"Soft better: {err_soft < err_hard}")

    assert ctrl_soft.gating.output.out_features == k_soft, "Gating must match K"

    print("PASS: Soft/hard routing comparison complete")
    print()
    reset_seed()


if __name__ == "__main__":
    reset_seed()

    test_cartpole_soft()
    test_double_well_soft()
    test_triple_well_soft()
    test_collapse_detection()
    test_soft_vs_hard()

    print("=" * 50)
    print("STAGE 5 COMPLETE")
