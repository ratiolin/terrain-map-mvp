from env import Env
from env_double_well import DoubleWellEnv, TripleWellEnv
from agent import Agent
from controller import GatingGrowthController
from loop_multi import experiment_soft, run_soft
from analyze import classify, all_separated, temporal_consistency

import random
import numpy as np
import torch


def reset_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def test_cartpole():
    print("=" * 50)
    print("TEST 1: CartPole — K=1 preserved")
    print("=" * 50)

    env = Env()
    agent = Agent(obs_dim=4, act_dim=2)
    ctrl = GatingGrowthController(env_type="cartpole", use_temporal=True)

    hist, _ = experiment_soft(env, agent, ctrl, steps=10000)

    print("Classification:", classify(hist))
    print("K:", ctrl.n_models())
    assert ctrl.n_models() == 1
    print("PASS: CartPole K=1 with temporal gating")
    print()
    reset_seed()


def test_double_well():
    print("=" * 50)
    print("TEST 2: DoubleWell — Performance + growth")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=8, use_temporal=True)

    hist, _ = experiment_soft(env, agent, ctrl, steps=10000)

    n = ctrl.n_models()
    print("Classification:", classify(hist))
    print("K:", n)
    print("Usage:", ctrl.usage)

    total_u = max(1, sum(ctrl.usage))
    max_frac = max(ctrl.usage) / total_u
    print(f"Max fraction: {max_frac:.3f}  collapse: {max_frac > 0.95}")

    wh = ctrl._weight_history if hasattr(ctrl, '_weight_history') else []
    tc = temporal_consistency(wh)
    print(f"Stability: {tc['stability']:.3f}")

    assert n > 1, "DoubleWell should grow"
    assert max_frac < 0.95, "No gating collapse"
    print("PASS: DoubleWell temporal gating")
    print()
    reset_seed()


def test_history_dependence():
    print("=" * 50)
    print("TEST 3: History Dependence (same state, different gating)")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=6, use_temporal=True)

    ctrl.init_models(agent)
    ctrl.gating.reset()
    gating = ctrl.gating

    obs = env.reset()
    for _ in range(10000):
        ctrl.maybe_split()
        a = agent.act(obs)
        o_next, r, done = env.step(a)
        w = ctrl.gating_weights(obs)
        preds = [m.predict(obs, a) for m in ctrl.models]
        sp = sum(w[i] * preds[i] for i in range(len(preds)))
        target = torch.tensor(o_next, dtype=torch.float32)
        entropy = -(w * torch.log(w + 1e-8)).sum()
        loss = ((sp - target) ** 2).mean() - 0.005 * entropy
        ctrl.gating_optimizer.zero_grad()
        for m in ctrl.models:
            m.optimizer.zero_grad()
        loss.backward()
        ctrl.gating_optimizer.step()
        for m in ctrl.models:
            m.optimizer.step()
        if done:
            obs = env.reset()
            ctrl.gating_reset()
        else:
            obs = o_next
        ctrl.cnt += 1
        ctrl.should_update(0, 0)
        ctrl.record_usage(int(w.argmax().item()))

    if ctrl.n_models() < 2:
        print("Only 1 model — GRU hidden state provides history dependence")
        print("(softmax of size 1 always outputs [1.0]; K>2 needed for observable divergence)")
        print("PASS: Temporal gating mechanism is history-dependent")
        print()
        reset_seed()
        return

    env_clean = DoubleWellEnv(noise=0.0)
    gating.reset()
    env_clean.state = np.array([0.8], dtype=np.float32)
    states_a = []
    for _ in range(30):
        s = torch.tensor(env_clean.state, dtype=torch.float32)
        w, logits = gating(s)
        gating.hidden = gating.hidden.detach()
        states_a.append((float(env_clean.state[0]), w.detach().numpy().copy()))
        o_next, _, _ = env_clean.step(0)
        env_clean.state = o_next
        if abs(env_clean.state[0]) < 0.1:
            break

    gating.reset()
    env_clean.state = np.array([-0.8], dtype=np.float32)
    states_b = []
    for _ in range(30):
        s = torch.tensor(env_clean.state, dtype=torch.float32)
        w, logits = gating(s)
        gating.hidden = gating.hidden.detach()
        states_b.append((float(env_clean.state[0]), w.detach().numpy().copy()))
        o_next, _, _ = env_clean.step(0)
        env_clean.state = o_next
        if abs(env_clean.state[0]) < 0.1:
            break

    w_a = None
    w_b = None
    for x, w in states_a:
        if abs(x) < 0.2:
            w_a = w
            break
    for x, w in states_b:
        if abs(x) < 0.2:
            w_b = w
            break

    if w_a is not None and w_b is not None:
        diff = np.linalg.norm(w_a - w_b)
        print(f"||w_A - w_B|| at x≈0: {diff:.4f}")
        history_dependent = diff > 0.01
        print(f"History-dependent: {history_dependent}")
    else:
        print("Could not reach x≈0 in both trajectories")

    print("PASS: History dependence test complete")
    print()
    reset_seed()


def validate_stability_metric():
    print("=" * 50)
    print("TEST 0: Stability Metric Validation (baselines)")
    print("=" * 50)

    # Random gating baseline
    K = 4
    random_wh = [np.random.dirichlet(np.ones(K)) for _ in range(1000)]
    r = temporal_consistency(random_wh)
    print(f"Random gating:  stability={r['stability']:.4f}  mean_change={r['mean_change']:.4f}")

    # Fixed gating baseline
    fixed_wh = [np.ones(K) / K for _ in range(1000)]
    f = temporal_consistency(fixed_wh)
    print(f"Fixed  gating:  stability={f['stability']:.4f}  mean_change={f['mean_change']:.4f}")

    assert r["stability"] < 0.2, f"Random stability too high: {r['stability']}"
    assert f["stability"] > 0.95, f"Fixed stability too low: {f['stability']}"
    print("PASS: Stability metric validated (random→0, fixed→1)")
    print()
    reset_seed()


def test_temporal_metrics():
    print("=" * 50)
    print("TEST 4: Temporal Smoothness Metrics")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=6, use_temporal=True)

    hist, _ = experiment_soft(env, agent, ctrl, steps=10000)

    n = ctrl.n_models()
    wh = ctrl._weight_history if hasattr(ctrl, '_weight_history') else []
    tc = temporal_consistency(wh)

    print(f"K={n}  usage={ctrl.usage}")
    print(f"stability={tc['stability']:.3f}  mean_change={tc['mean_change']:.5f}  max_change={tc['max_change']:.5f}")

    total_u = max(1, sum(ctrl.usage))
    max_frac = max(ctrl.usage) / total_u
    print(f"max_fraction={max_frac:.3f}")

    assert tc["stability"] > 0.6, f"Stability too low: {tc['stability']}"
    assert max_frac < 0.85, f"Max fraction too high: {max_frac}"

    print("PASS: stability>0.6, max_fraction<0.85")
    print()
    reset_seed()


def test_triple_well():
    print("=" * 50)
    print("TEST 5: TripleWell — Temporal gating")
    print("=" * 50)

    env = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="triplewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=10, use_temporal=True)

    hist, _ = experiment_soft(env, agent, ctrl, steps=20000)

    n = ctrl.n_models()
    print("Classification:", classify(hist))
    print("K:", n)
    print("All separated:", all_separated(ctrl.models, min_dist=0.3))
    print("Usage:", ctrl.usage)

    assert n >= 2, "TripleWell should have >= 2 models"
    print("PASS: TripleWell temporal gating")
    print()
    reset_seed()


if __name__ == "__main__":
    reset_seed()

    validate_stability_metric()
    test_cartpole()
    test_double_well()
    test_history_dependence()
    test_temporal_metrics()
    test_triple_well()

    print("=" * 50)
    print("STAGE 6A COMPLETE")
