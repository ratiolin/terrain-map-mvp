import json
import numpy as np
import torch

from core_mvp_v2.control_env_torch import TorchControlWell, ControlPolicy


def reset_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_policy_bptt(policy, env, total_steps=500, bptt_len=5):
    costs = []
    env.reset()
    state = env.state.detach()
    grads = []

    step = 0
    while step < total_steps:
        rollout_cost = torch.tensor(0.0)
        rollout_states = []

        for k in range(bptt_len):
            if step + k >= total_steps:
                break
            action = policy(state.unsqueeze(0))
            cost = env.step(action)
            rollout_cost = rollout_cost + cost
            rollout_states.append(env.state)
            costs.append(float(cost.item()))
            state = env.state

        policy.optimizer.zero_grad()
        rollout_cost.backward()
        g = sum(p.abs().mean().item() for p in policy.parameters() if p.grad is not None)
        grads.append(g)
        policy.optimizer.step()
        env.detach_state()
        state = env.state.detach()
        step += bptt_len

    return costs, grads


def run_policy(policy, env, steps=500, train=False):
    costs = []
    env.reset()
    state = env.state.detach()

    for _ in range(steps):
        with torch.no_grad():
            action = policy(state.unsqueeze(0))
        cost = env.step(action)
        costs.append(float(cost.item()))

        if train:
            policy.optimizer.zero_grad()
            cost.backward()
            policy.optimizer.step()
            env.detach_state()

        state = env.state.detach()

    return costs


def run_frozen(env, steps=500):
    costs = []
    env.reset()
    state = env.state.detach()
    policy = ControlPolicy()

    for _ in range(steps):
        with torch.no_grad():
            action = policy(state.unsqueeze(0))
        cost = env.step(action)
        costs.append(float(cost.item()))
        state = env.state.detach()

    return costs


def run_random(env, steps=500):
    costs = []
    env.reset()
    state = env.state.detach()

    for _ in range(steps):
        with torch.no_grad():
            action = torch.randn(1) * 0.5
            action = torch.clamp(action, -1.0, 1.0)
        cost = env.step(action)
        costs.append(float(cost.item()))
        state = env.state.detach()

    return costs


def run_zero(env, steps=500):
    costs = []
    env.reset()
    state = env.state.detach()

    for _ in range(steps):
        action = torch.tensor(0.0)
        cost = env.step(action)
        costs.append(float(cost.item()))
        state = env.state.detach()

    return costs


def gradient_check():
    env = TorchControlWell(drift_rate=0.02, alpha=0.5)
    policy = ControlPolicy()
    env.reset()
    state = env.state.detach()

    action = policy(state.unsqueeze(0))
    cost = env.step(action)

    policy.optimizer.zero_grad()
    cost.backward()

    grads = []
    for name, p in policy.named_parameters():
        if p.grad is not None:
            grads.append(float(p.grad.abs().mean().item()))
    grad_mean = np.mean(grads) if grads else 0.0

    print(f"\n=== GRADIENT CHECK ===")
    print(f"  cost={cost.item():.4f}")
    print(f"  grad_abs_mean={grad_mean:.8f}")
    print(f"  gradient_path: {'PASS ✓' if grad_mean > 0 else 'FAIL ✗'}")

    return grad_mean > 0


def run_control_experiment(g_min=0.0, g_max=3.0, n_points=12, eta=0.5, alpha=0.2):
    g_vals = np.linspace(g_min, g_max, n_points)
    all_results = {}

    print(f"\n=== CONTROL PHASE DIAGRAM: α={alpha}, g∈[{g_min},{g_max}] ===")
    print(f"{'g':>8} {'cost_learn':>12} {'cost_frozen':>12} {'cost_random':>12} {'cost_zero':>12}")

    for g in g_vals:
        drift = g / eta if g > 1e-8 else 1e-4
        row = []

        for mode, fn in [("learned", train_policy), ("frozen", run_frozen),
                          ("random", run_random), ("zero", run_zero)]:
            reset_seed(0)
            env = TorchControlWell(drift_rate=drift, alpha=alpha)
            if mode in ("learned", "frozen"):
                policy = ControlPolicy()
                if mode == "learned":
                    costs, grads = fn(policy, env, steps=500)
                else:
                    costs = fn(env, steps=500)
            else:
                costs = fn(env, steps=500)
            mean_cost = float(np.mean(costs[-200:]))
            row.append(mean_cost)

        print(f"{g:>8.2f} {row[0]:>12.4f} {row[1]:>12.4f} {row[2]:>12.4f} {row[3]:>12.4f}")

    return g_vals, all_results


def run_target_switch(alpha=0.2):
    print(f"\n=== TARGET SWITCHING ===")
    for period in [50, 100]:
        env = TorchControlWell(drift_rate=0.02, alpha=alpha, target_state=1.0)
        policy = ControlPolicy()
        env.reset()
        state = env.state.detach()
        switch_costs = []

        for step in range(800):
            if step > 0 and step % period == 0:
                env.target_state = -env.target_state
                switch_start = step

            action = policy(state.unsqueeze(0))
            cost = env.step(action)
            if step >= 100 and step < 200:
                switch_costs.append(float(cost.item()))

            policy.optimizer.zero_grad()
            cost.backward()
            policy.optimizer.step()
            env.detach_state()
            state = env.state.detach()

        print(f"  period={period}: adaptation_cost={np.mean(switch_costs):.4f}")


if __name__ == "__main__":
    if not gradient_check():
        print("GRADIENT CHECK FAILED — aborting")
        exit(1)

    run_control_experiment(n_points=10, alpha=0.2)
    run_target_switch(alpha=0.2)
