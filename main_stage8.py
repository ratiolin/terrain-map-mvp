import random
import numpy as np
import torch
import torch.optim as optim
import copy

from env import Env
from env_double_well import DoubleWellEnv
from agent import Agent
from controller import GatingGrowthController
from loop_policy import run_policy
from analyze import classify


def reset_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def init_fixed_k(controller, agent, K):
    controller.env_type = "cartpole"
    controller.init_models(agent)
    for _ in range(K - 1):
        child = copy.deepcopy(controller.models[0])
        child.optimizer = optim.Adam(child.parameters(), lr=1e-3)
        with torch.no_grad():
            for p in child.predictor.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        controller.models.append(child)
        gating = controller.gating
        gating.expand()
        controller.gating_optimizer = optim.Adam(gating.parameters(), lr=1e-3)
    controller.usage = [0] * K
    controller.errors = [[] for _ in range(K)]
    controller.birth_step = [0] * K
    controller.freeze_structure = True
    controller.use_z = True


def check_entropy_evolution(entropy_history, step=200):
    n = len(entropy_history)
    if n < 3 * step:
        return False
    early = np.mean(entropy_history[:step])
    mid = np.mean(entropy_history[n // 2 - step // 2:n // 2 + step // 2])
    late = np.mean(entropy_history[-step:])
    return abs(early - mid) > 1e-6 or abs(mid - late) > 1e-6 or abs(early - late) > 1e-6


def test_doublewell_z_policy():
    print("=" * 60)
    print("TEST 1: DoubleWell — Z-policy (self-growing)")
    print("=" * 60)

    env = DoubleWellEnv(noise=0.03)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=8, use_z=True)

    hist, wh, rewards, entropies, Phi = run_policy(env, agent, ctrl, steps=15000)

    K = ctrl.n_models()

    print(f"K_final = {K}")
    print()

    # ---- 6. 打印 Phi ----
    print("Phi:", Phi)
    print()

    # ---- 7. 验证 ----
    print("--- entropy analysis ---")
    n_seg = min(1000, len(entropies))
    seg = len(entropies) // 3
    e0 = np.mean(entropies[:seg]) if seg > 0 else 0
    e1 = np.mean(entropies[seg:2 * seg]) if 2 * seg <= len(entropies) else 0
    e2 = np.mean(entropies[-seg:]) if seg > 0 else 0
    print(f"  entropy early:  {e0:.4f}")
    print(f"  entropy mid:    {e1:.4f}")
    print(f"  entropy late:   {e2:.4f}")

    ent_var = np.var(entropies[-n_seg:]) if len(entropies) >= n_seg else np.var(entropies)
    print(f"  entropy tail var: {ent_var:.6f}")

    entropy_changes = check_entropy_evolution(entropies)
    print(f"  entropy changes over training: {entropy_changes}")

    reward_var = Phi["reward_variance"]
    print(f"\n  reward_variance: {reward_var:.6f}")

    print(f"\n  pred_error: {Phi['pred_error']:.6f}")

    assert entropy_changes, "policy_entropy should change over training"
    assert reward_var > 1e-8, f"reward_variance too small: {reward_var}"
    print("\nPASS: DoubleWell Z-policy strategy loop verified")
    print()
    reset_seed()


def test_doublewell_fixed_k():
    print("=" * 60)
    print("TEST 2: DoubleWell — Z-policy K=3 fixed")
    print("=" * 60)

    env = DoubleWellEnv(noise=0.03)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=8, use_z=True)
    init_fixed_k(ctrl, agent, K=3)

    hist, wh, rewards, entropies, Phi = run_policy(env, agent, ctrl, steps=10000)

    K = ctrl.n_models()
    print(f"K = {K} (fixed)")
    print()

    print("Phi:", Phi)
    print()

    n_seg = min(500, len(entropies))
    seg = len(entropies) // 3
    e0 = np.mean(entropies[:seg]) if seg > 0 else 0
    e1 = np.mean(entropies[seg:2 * seg]) if 2 * seg <= len(entropies) else 0
    e2 = np.mean(entropies[-seg:]) if seg > 0 else 0
    print(f"entropy early:  {e0:.4f}")
    print(f"entropy mid:    {e1:.4f}")
    print(f"entropy late:   {e2:.4f}")

    entropy_changes = check_entropy_evolution(entropies)
    print(f"entropy changes over training: {entropy_changes}")

    max_entropy = np.log(K)
    print(f"early entropy / max entropy ({max_entropy:.2f}): {e0 / max_entropy:.3f}")
    print(f"late entropy / max entropy ({max_entropy:.2f}): {e2 / max_entropy:.3f}")

    reward_var = Phi["reward_variance"]
    print(f"reward_variance: {reward_var:.6f}")

    assert entropy_changes, "policy_entropy should change over training"
    assert reward_var > 1e-8, f"reward_variance too small: {reward_var}"
    assert K == 3, f"K should stay 3, got {K}"
    print("\nPASS: DoubleWell K=3 fixed Z-policy verified")
    print()
    reset_seed()


def test_cartpole_z_policy():
    print("=" * 60)
    print("TEST 3: CartPole — Z-policy K=2")
    print("=" * 60)

    env = Env()
    agent = Agent(obs_dim=4, act_dim=2)
    ctrl = GatingGrowthController(env_type="cartpole", use_z=True)
    init_fixed_k(ctrl, agent, K=2)

    hist, wh, rewards, entropies, Phi = run_policy(env, agent, ctrl, steps=5000, act_dim=2)

    K = ctrl.n_models()
    print(f"K = {K}")
    print()

    print("Phi:", Phi)
    print()

    print("Classification:", classify(hist))

    seg = len(entropies) // 3
    e0 = np.mean(entropies[:seg]) if seg > 0 else 0
    e1 = np.mean(entropies[seg:2 * seg]) if 2 * seg <= len(entropies) else 0
    e2 = np.mean(entropies[-seg:]) if seg > 0 else 0
    print(f"entropy early:  {e0:.4f}")
    print(f"entropy mid:    {e1:.4f}")
    print(f"entropy late:   {e2:.4f}")

    entropy_changes = check_entropy_evolution(entropies, step=100)
    print(f"entropy changes over training: {entropy_changes}")

    reward_var = Phi["reward_variance"]
    print(f"reward_variance: {reward_var:.6f}")

    assert K == 2, f"K should be 2, got {K}"
    assert entropy_changes, "policy_entropy should change over training"
    if reward_var < 1e-8:
        print("  Note: CartPole-v1 has constant per-step reward=1.0, variance naturally 0")
    print("\nPASS: CartPole Z-policy strategy loop verified")
    print()
    reset_seed()


if __name__ == "__main__":
    reset_seed()

    test_doublewell_z_policy()
    test_doublewell_fixed_k()
    test_cartpole_z_policy()

    print("=" * 60)
    print("STAGE 8 COMPLETE — Strategy Closed Loop Verified")
    print("=" * 60)
