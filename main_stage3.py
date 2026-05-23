from env import Env
from env_double_well import DoubleWellEnv
from agent import Agent
from controller import BifurcationController, Controller as BaseCtrl
from loop_multi import experiment_multi, run_multi
from loop import run
from metrics import Metrics
from analyze import classify, is_stable, rollout, rollout_mean_state, verify_stage3

import random
import numpy as np
import torch


def test_cartpole():
    print("=" * 50)
    print("TEST 1: CartPole — Structural Conservativeness")
    print("=" * 50)

    env = Env()
    agent = Agent(obs_dim=4, act_dim=2)
    controller = BifurcationController(check_interval=200, env_type="cartpole")

    history, metrics = experiment_multi(env, agent, controller, steps=10000)

    print("Classification:", classify(history))
    print("Bifurcated:", controller.bifurcated)
    print("Multi-model active:", controller.multi_model_active)

    assert not controller.bifurcated, "CartPole should NOT trigger bifurcation"
    print("PASS: CartPole conserves single-model structure")
    print()


def test_double_well():
    print("=" * 50)
    print("TEST 2: DoubleWell — Bifurcation & Stability Mapping")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    controller = BifurcationController(check_interval=200, env_type="doublewell")

    history, metrics = experiment_multi(env, agent, controller, steps=10000)

    status_full = classify(history)
    print("DoubleWell classification:", status_full)
    print("Bifurcated:", controller.bifurcated)
    print()

    if not controller.bifurcated:
        print("WARNING: bifurcation not triggered")
        return False

    # --- Verification: P1 and P2 stability & divergence ---
    verified = verify_stage3(controller.P1, controller.P2, DoubleWellEnv(noise=0.02), T=2000, var_th=2e-3)
    print("verify_stage3:", verified)

    h1 = rollout(controller.P1, DoubleWellEnv(noise=0.02), 2000)
    h2 = rollout(controller.P2, DoubleWellEnv(noise=0.02), 2000)
    s1 = is_stable(h1, var_th=2e-3)
    s2 = is_stable(h2, var_th=2e-3)
    print(f"  P1 stable: {s1} (var={np.var(h1[-1000:]):.6f})")
    print(f"  P2 stable: {s2} (var={np.var(h2[-1000:]):.6f})")

    m1 = rollout_mean_state(controller.P1, DoubleWellEnv(noise=0.02), 500)
    m2 = rollout_mean_state(controller.P2, DoubleWellEnv(noise=0.02), 500)
    print(f"  P1 mean state: {m1:.4f} | P2 mean state: {m2:.4f}")
    print(f"  Diverged (opposite sign): {m1 * m2 < 0}")
    print(f"  Separated (|diff|>0.2): {abs(m1 - m2) > 0.2}")
    print()

    # --- Performance comparison: multi-model vs single-model ---
    print("--- Multi vs Single Error Comparison ---")
    env_s = DoubleWellEnv(noise=0.02)
    agent_s = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    metrics_s = Metrics()
    hist_s = run(env_s, agent_s, metrics_s, BaseCtrl(), steps=5000)
    err_single = np.mean(hist_s[-1000:])
    var_single = np.var(hist_s[-1000:])

    env_m = DoubleWellEnv(noise=0.02)
    # Re-run multi model (fresh) to compare fairly
    agent_m = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl_m = BifurcationController(check_interval=150)
    hist_m, _ = experiment_multi(env_m, agent_m, ctrl_m, steps=5000)
    err_multi = np.mean(hist_m[-1000:])
    var_multi = np.var(hist_m[-1000:])

    print(f"  Single: mean_err={err_single:.6f}  var={var_single:.6f}")
    print(f"  Multi:  mean_err={err_multi:.6f}  var={var_multi:.6f}")
    print(f"  Multi improves (lower error): {err_multi < err_single}")
    print()

    return verified


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    test_cartpole()
    test_double_well()
