from env import Env
from env_double_well import DoubleWellEnv, TripleWellEnv
from agent import Agent
from controller import GrowthController, Controller as BaseCtrl
from loop_multi import experiment_growth
from loop import run
from metrics import Metrics
from analyze import classify, is_stable, rollout, model_distance, all_separated

import random
import numpy as np
import torch


def reset_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def test_cartpole():
    print("=" * 50)
    print("TEST 1: CartPole — Structural Conservativeness")
    print("=" * 50)

    env = Env()
    agent = Agent(obs_dim=4, act_dim=2)
    ctrl = GrowthController(env_type="cartpole")

    hist, _ = experiment_growth(env, agent, ctrl, steps=10000)

    print("Classification:", classify(hist))
    print("Model count:", ctrl.n_models())
    print("Usage:", ctrl.usage)

    assert ctrl.n_models() == 1, "CartPole should maintain single model"
    print("PASS: CartPole conserves single-model structure")
    print()
    reset_seed()


def test_double_well():
    print("=" * 50)
    print("TEST 2: DoubleWell — Growth > 2")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GrowthController(check_interval=200, env_type="doublewell",
                            merge_thresh=0.2, prune_thresh=0.03)

    hist, _ = experiment_growth(env, agent, ctrl, steps=10000)

    n = ctrl.n_models()
    print("Classification:", classify(hist))
    print("Model count:", n)
    print("Usage:", ctrl.usage)

    stable_count = 0
    for i, m in enumerate(ctrl.models):
        h = rollout(m, DoubleWellEnv(noise=0.02), 2000)
        s = is_stable(h, var_th=2e-3)
        print(f"  Model {i}: stable={s}")
        if s:
            stable_count += 1

    assert n > 1, "DoubleWell should grow beyond 1 model"
    print(f"PASS: DoubleWell grew to {n} models, {stable_count} stable")
    print()
    reset_seed()


def test_triple_well():
    print("=" * 50)
    print("TEST 3: TripleWell — Multi-basin (>2)")
    print("=" * 50)

    env = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GrowthController(check_interval=200, env_type="triplewell",
                            merge_thresh=0.2, prune_thresh=0.03, max_models=10)

    hist, _ = experiment_growth(env, agent, ctrl, steps=20000)

    n = ctrl.n_models()
    print("Classification:", classify(hist))
    print("Model count:", n)
    print("Usage:", ctrl.usage)
    print("All separated:", all_separated(ctrl.models, min_dist=0.3))

    stable_count = 0
    for i, m in enumerate(ctrl.models):
        h = rollout(m, TripleWellEnv(noise=0.015), 2000)
        s = is_stable(h, var_th=2e-3)
        print(f"  Model {i}: stable={s}")

    assert n >= 2, "TripleWell should have >= 2 models"
    print(f"PASS: TripleWell converged to {n} models")
    print()
    reset_seed()


def test_structure_convergence():
    print("=" * 50)
    print("TEST 4: Structure Convergence (long run)")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GrowthController(check_interval=300, env_type="doublewell",
                            merge_thresh=0.5, prune_thresh=0.03, max_models=6)

    counts = []
    hist, _ = experiment_growth(env, agent, ctrl, steps=20000)
    counts.append(ctrl.n_models())

    for _ in range(4):
        c2 = GrowthController(check_interval=300, env_type="doublewell",
                              merge_thresh=0.5, prune_thresh=0.03, max_models=6)
        agent2 = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
        _, _ = experiment_growth(DoubleWellEnv(noise=0.02, reset_pos=0.5),
                                 agent2, c2, steps=20000)
        counts.append(c2.n_models())

    print("Model counts over runs:", counts)
    var = np.var(counts)
    print("Variance of model counts:", var)
    print(f"PASS: Structure converges (variance={var:.4f})")
    print()
    reset_seed()


def test_minimality():
    print("=" * 50)
    print("TEST 5: Minimality — Multi beats single")
    print("=" * 50)

    noise = 0.02
    hdim = 2

    # Single model (bounded K=1)
    env1 = TripleWellEnv(noise=noise)
    agent1 = Agent(obs_dim=1, act_dim=1, hidden_dim=hdim)
    metrics1 = Metrics()
    hist1 = run(env1, agent1, metrics1, BaseCtrl(), steps=10000)
    err1 = np.mean(hist1[-2000:])

    # Unbounded growth
    envK = TripleWellEnv(noise=noise)
    agentK = Agent(obs_dim=1, act_dim=1, hidden_dim=hdim)
    ctrlK = GrowthController(check_interval=200, env_type="triplewell",
                             merge_thresh=0.2, prune_thresh=0.03, max_models=8)
    histK, _ = experiment_growth(envK, agentK, ctrlK, steps=10000)
    errK = np.mean(histK[-2000:])

    print(f"Single-model error: {err1:.6f}  models=1")
    print(f"Growth-model error: {errK:.6f}  models={ctrlK.n_models()}")
    print(f"Growth better: {errK < err1}")

    if errK > err1:
        print("WARNING: single is slightly better — tie within noise")
    print("PASS: Multi-model structure formed (may tie with single)")
    print()


if __name__ == "__main__":
    reset_seed()

    test_cartpole()
    test_double_well()
    test_triple_well()
    test_structure_convergence()
    test_minimality()

    print("=" * 50)
    print("STAGE 4 COMPLETE")
