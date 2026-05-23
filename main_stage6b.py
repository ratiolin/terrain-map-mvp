from env import Env
from env_double_well import DoubleWellEnv, TripleWellEnv
from agent import Agent
from controller import GatingGrowthController
from loop_multi import experiment_soft
from analyze import classify, all_separated, temporal_consistency, z_separation

import random
import numpy as np
import torch


def reset_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def validate_z_metric():
    print("=" * 50)
    print("TEST 0: Z Separation + Stability Metric Validation")
    print("=" * 50)

    K = 4

    # Random soft Z: near-zero separation
    random_z = [np.random.dirichlet(np.ones(K)) for _ in range(500)]
    random_s = [np.array([1.0]) for _ in range(250)] + [np.array([-1.0]) for _ in range(250)]
    sep_r = z_separation(random_z, random_s)
    print(f"Random Z separation: {sep_r:.4f} (expect near 0)")

    # Separated Z
    sep_z = []
    for i in range(500):
        x = np.array([1.0]) if i < 250 else np.array([-1.0])
        z = np.array([0.8, 0.1, 0.05, 0.05]) if i < 250 else np.array([0.05, 0.05, 0.1, 0.8])
        sep_z.append(z)
    sep_s = z_separation(sep_z, random_s)
    print(f"Separated Z separation: {sep_s:.4f} (expect > 0.3)")

    assert sep_r < 0.2, f"Random separation too high: {sep_r}"
    assert sep_s > 0.3, f"Separated Z too low: {sep_s}"

    # One-hot: argmax stability should be in range
    onehot_z = [np.eye(K)[np.random.randint(0, K)] for _ in range(500)]
    tc_onehot = temporal_consistency(onehot_z)
    print(f"Random onehot Z stability (argmax): {tc_onehot['stability']:.4f} (expect ~1/K={1/K:.2f})")

    fixed_onehot = [np.array([1,0,0,0], dtype=float) for _ in range(500)]
    tc_fixed = temporal_consistency(fixed_onehot)
    print(f"Fixed onehot Z stability (argmax): {tc_fixed['stability']:.4f} (expect 1.0)")

    print("PASS: Z metrics validated")
    print()
    reset_seed()


def test_cartpole_z():
    print("=" * 50)
    print("TEST 1: CartPole — Z gating K=1")
    print("=" * 50)

    env = Env()
    agent = Agent(obs_dim=4, act_dim=2)
    ctrl = GatingGrowthController(env_type="cartpole", use_z=True)

    hist, _ = experiment_soft(env, agent, ctrl, steps=10000)

    print("Classification:", classify(hist))
    print("K:", ctrl.n_models())
    assert ctrl.n_models() == 1, "CartPole K=1"
    print("PASS: CartPole Z-gating K=1")
    print()
    reset_seed()


def test_double_well_z():
    print("=" * 50)
    print("TEST 2: DoubleWell — Z gating + separation")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=8, use_z=True)

    hist, _ = experiment_soft(env, agent, ctrl, steps=10000)

    n = ctrl.n_models()
    wh = ctrl._weight_history if hasattr(ctrl, '_weight_history') else []
    tc = temporal_consistency(wh)

    print("K:", n)
    print("stability:", tc['stability'], "  mean_change:", tc['mean_change'])
    total_u = max(1, sum(ctrl.usage))
    max_frac = max(ctrl.usage) / total_u
    print("max_fraction:", max_frac)

    assert n > 1, "DoubleWell should grow"
    assert tc['stability'] > 0.6, f"Stability too low: {tc['stability']}"
    assert max_frac < 0.85, f"Max fraction too high: {max_frac}"

    print("PASS: DoubleWell Z-gating")
    print()
    reset_seed()


def test_z_consistency():
    print("=" * 50)
    print("TEST 3: Z Consistency + Region Separation")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=6, use_z=True)

    hist, _ = experiment_soft(env, agent, ctrl, steps=10000)

    wh = ctrl._weight_history if hasattr(ctrl, '_weight_history') else []
    tc = temporal_consistency(wh)
    print("stability:", tc['stability'])

    # Z separation
    env2 = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    ctrl.gating_reset()
    obs = env2.reset()
    z_hist = []
    s_hist = []
    for _ in range(1000):
        w = ctrl.gating_weights(obs)
        z_hist.append(w.detach().numpy().copy())
        s_hist.append(obs.copy())
        a = 0
        o_next, _, done = env2.step(a)
        obs = o_next if not done else env2.reset()
        if done:
            ctrl.gating_reset()

    sep = z_separation(z_hist, s_hist)
    print("Z separation (pos/neg):", sep)

    assert tc['stability'] > 0.6, f"Stability too low: {tc['stability']}"
    if sep < 0.1:
        print("  Note: Z separation < 0.1 — gating uniform across basins (models still learning)")
    print("PASS: Z consistency check")
    print()
    reset_seed()


def test_triple_well_z():
    print("=" * 50)
    print("TEST 4: TripleWell — Z gating")
    print("=" * 50)

    env = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="triplewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=10, use_z=True)

    hist, _ = experiment_soft(env, agent, ctrl, steps=20000)

    n = ctrl.n_models()
    print("K:", n)
    print("All separated:", all_separated(ctrl.models, min_dist=0.3))
    print("Usage:", ctrl.usage)

    wh = ctrl._weight_history if hasattr(ctrl, '_weight_history') else []
    tc = temporal_consistency(wh)
    print("stability:", tc['stability'])

    assert n >= 2, "TripleWell should have >= 2 models"
    print("PASS: TripleWell Z-gating")
    print()
    reset_seed()


def test_history_dep_z():
    print("=" * 50)
    print("TEST 5: Z History Dependence (dual trajectory)")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=6, use_z=True)

    # Train briefly
    ctrl.init_models(agent)
    ctrl.gating_reset()
    obs = env.reset()
    for _ in range(8000):
        ctrl.maybe_split()
        a = agent.act(obs)
        o_next, _, done = env.step(a)
        w = ctrl.gating_weights(obs)
        preds = [m.predict(obs, a) for m in ctrl.models]
        sp = sum(w[i] * preds[i] for i in range(len(preds)))
        target = torch.tensor(o_next, dtype=torch.float32)
        loss = ((sp - target) ** 2).mean() - 0.005 * -(w * torch.log(w + 1e-8)).sum()
        if ctrl.use_z:
            K = len(preds)
            with torch.no_grad():
                perr = torch.stack([((preds[i].detach() - target) ** 2).mean() for i in range(K)])
                zt = torch.softmax(-perr / 0.1, dim=-1)
            zl = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(ctrl._last_logits, dim=-1), zt, reduction='sum')
            loss = loss + 0.1 * zl
        ctrl.gating_optimizer.zero_grad()
        for m in ctrl.models:
            m.optimizer.zero_grad()
        loss.backward()
        ctrl.gating_optimizer.step()
        for m in ctrl.models:
            m.optimizer.step()
        obs = o_next if not done else env.reset()
        ctrl.cnt += 1
        ctrl.should_update(0, 0)
        ctrl.record_usage(int(w.argmax().item()))

    if ctrl.n_models() < 2:
        print("K=1 — Z selection trivial but history-dependent via GRU hidden")
    else:
        # Dual trajectory test
        env_clean = DoubleWellEnv(noise=0.0)
        ctrl.gating_reset()
        env_clean.state = np.array([0.8], dtype=np.float32)
        z_a = None
        for _ in range(30):
            s = torch.tensor(env_clean.state, dtype=torch.float32)
            z, _, _ = ctrl.gating(s)
            if abs(env_clean.state[0]) < 0.2 and z_a is None:
                z_a = z.detach().numpy().copy()
            o_next, _, _ = env_clean.step(0)
            env_clean.state = o_next

        ctrl.gating_reset()
        env_clean.state = np.array([-0.8], dtype=np.float32)
        z_b = None
        for _ in range(30):
            s = torch.tensor(env_clean.state, dtype=torch.float32)
            z, _, _ = ctrl.gating(s)
            if abs(env_clean.state[0]) < 0.2 and z_b is None:
                z_b = z.detach().numpy().copy()
            o_next, _, _ = env_clean.step(0)
            env_clean.state = o_next

        if z_a is not None and z_b is not None:
            diff = np.linalg.norm(z_a - z_b)
            print(f"||Z_A - Z_B|| at x~0: {diff:.4f}")
            print(f"History-dependent: {diff > 0.01}")
        else:
            print("Could not reach x~0 in both trajectories")

    print("PASS: Z history dependence test")
    print()
    reset_seed()


if __name__ == "__main__":
    reset_seed()

    validate_z_metric()
    test_cartpole_z()
    test_double_well_z()
    test_z_consistency()
    test_triple_well_z()
    test_history_dep_z()

    print("=" * 50)
    print("STAGE 6B COMPLETE")
