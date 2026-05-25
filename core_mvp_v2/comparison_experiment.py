import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core_mvp_v2.control_env_torch import TorchControlWell


class Predictor(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, state, action):
        state = state.reshape(-1, 1) if state.dim() > 1 else state.view(-1, 1)
        action = action.reshape(-1, 1) if action.dim() > 1 else action.view(-1, 1)
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


class ControlPolicy(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, state):
        if state.dim() == 0:
            state = state.unsqueeze(0)
        return torch.tanh(self.net(state).squeeze(-1))


def evaluate_policy(env, policy, n_episodes=50, steps_per=50, perturb=False):
    metrics = {'costs': [], 'in_target': [], 'recovery': [], 'failure': 0}
    eps = 0.1
    for _ in range(n_episodes):
        env.reset()
        env.state = torch.tensor([np.random.uniform(-2, 2)])
        state = env.state.detach()
        costs_ep = []
        in_target_ep = []

        if perturb:
            env.state = env.state + torch.tensor([1.5])
            recovering = True
        else:
            recovering = False

        recovery_steps = 0
        for t in range(steps_per):
            with torch.no_grad():
                action = policy(state.unsqueeze(0))
            cost = env.step(action)
            costs_ep.append(float(cost.item()))
            in_target_ep.append(1 if abs(float(env.state)) < eps else 0)
            state = env.state.detach()
            if recovering:
                recovery_steps += 1
                if abs(float(env.state)) < eps:
                    recovering = False
                    metrics['recovery'].append(recovery_steps)

        metrics['costs'].append(np.mean(costs_ep))
        metrics['in_target'].append(np.mean(in_target_ep))
        if recover_steps(steps_per, in_target_ep):
            metrics['failure'] += 1

    n = n_episodes
    return {
        'mean_cost': float(np.mean(metrics['costs'])),
        'time_in_target': float(np.mean(metrics['in_target'])),
        'recovery_time': float(np.mean(metrics['recovery'])) if metrics['recovery'] else float(steps_per),
        'failure_rate': float(metrics['failure'] / n),
    }


def recover_steps(max_steps, in_target_list):
    window = 30
    for t in range(len(in_target_list) - window):
        if np.mean(in_target_list[t:t+window]) > 0.5:
            return False
    return True


def train_condition_B(policy, env, n_episodes=200, steps_per=50, rollout_N=10):
    env_costs = []
    for ep in range(n_episodes):
        env.reset()
        env.state = torch.tensor([np.random.uniform(-2, 2)])
        state = env.state.detach()
        for t in range(steps_per):
            rollout_cost = torch.tensor(0.0)
            for k in range(rollout_N):
                action = policy(state.unsqueeze(0))
                cost = env.step(action)
                rollout_cost = rollout_cost + cost
                state = env.state
            loss = rollout_cost / rollout_N
            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()
            env.detach_state()
            state = env.state.detach()
        env_costs.append(float(loss.item()))
    return env_costs


def train_condition_A(policy, predictor, env, n_episodes=200, steps_per=50, rollout_N=10):
    env_costs = []
    for ep in range(n_episodes):
        env.reset()
        env.state = torch.tensor([np.random.uniform(-2, 2)])
        state = env.state.detach()
        for t in range(steps_per):
            for k in range(rollout_N):
                action = policy(state.unsqueeze(0))
                cost = env.step(action)
                pred_next = predictor(state.unsqueeze(0), action)
                actual_next = env.state.detach()
                pred_loss = ((pred_next - actual_next) ** 2).mean()

                predictor.optimizer.zero_grad()
                policy.optimizer.zero_grad()
                pred_loss.backward()
                predictor.optimizer.step()
                policy.optimizer.step()

                env.detach_state()
                state = env.state.detach()
            env_costs.append(float(cost.item()))
    return env_costs


def run_comparison_experiment(n_seeds=5):
    print("=" * 60)
    print("CONDITION A (prediction-first) vs CONDITION B (stability-first)")
    print("=" * 60)

    all_results = {'A': [], 'B': [], 'random': []}

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        env_B = TorchControlWell(kappa=1.0, drift_rate=0.0, alpha=2.0,
                                  noise_std=0.05, target_state=0.0)
        policy_B = ControlPolicy()
        train_condition_B(policy_B, env_B)
        policy_B.eval()

        eval_env = TorchControlWell(kappa=1.0, drift_rate=0.0, alpha=2.0,
                                     noise_std=0.05, target_state=0.0)
        m_B = evaluate_policy(eval_env, policy_B)

        torch.manual_seed(seed)
        np.random.seed(seed)
        env_A = TorchControlWell(kappa=1.0, drift_rate=0.0, alpha=2.0,
                                  noise_std=0.05, target_state=0.0)
        policy_A = ControlPolicy()
        predictor_A = Predictor()
        train_condition_A(policy_A, predictor_A, env_A)
        policy_A.eval()

        eval_env2 = TorchControlWell(kappa=1.0, drift_rate=0.0, alpha=2.0,
                                      noise_std=0.05, target_state=0.0)
        m_A = evaluate_policy(eval_env2, policy_A)

        torch.manual_seed(seed)
        np.random.seed(seed)
        eval_env3 = TorchControlWell(kappa=1.0, drift_rate=0.0, alpha=2.0,
                                      noise_std=0.05, target_state=0.0)
        policy_R = ControlPolicy()
        m_R = evaluate_policy(eval_env3, policy_R)

        all_results['A'].append(m_A)
        all_results['B'].append(m_B)
        all_results['random'].append(m_R)

        print(f"seed={seed}: A_cost={m_A['mean_cost']:.4f} B_cost={m_B['mean_cost']:.4f} R_cost={m_R['mean_cost']:.4f}")

    # Aggregate
    for cond in ['A', 'B', 'random']:
        arr = all_results[cond]
        for k in ['mean_cost', 'time_in_target', 'recovery_time', 'failure_rate']:
            vals = [a[k] for a in arr]
            print(f"  {cond} {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Judgment
    A_cost = np.mean([a['mean_cost'] for a in all_results['A']])
    B_cost = np.mean([a['mean_cost'] for a in all_results['B']])
    R_cost = np.mean([a['mean_cost'] for a in all_results['random']])

    print(f"\n=== JUDGMENT ===")
    if B_cost < A_cost and B_cost < R_cost:
        print("Case 1: THEORY SUPPORTED ✅ — stability-first beats prediction-first")
    elif abs(A_cost - B_cost) < 0.1:
        print("Case 2: OBJECTIVES EQUIVALENT ⚠️")
    elif A_cost < B_cost:
        print("Case 3: THEORY REFUTED ❌ — prediction-first beats stability-first")
    else:
        print(f"Unclassified: A={A_cost:.3f} B={B_cost:.3f} R={R_cost:.3f}")

    return all_results


if __name__ == "__main__":
    run_comparison_experiment()
