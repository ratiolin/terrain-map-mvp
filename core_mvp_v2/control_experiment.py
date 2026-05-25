import random
import json
import numpy as np
import torch
import torch.optim as optim

from core_mvp_v2.control_env import ControlDoubleWell


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ActionPolicy(torch.nn.Module):
    def __init__(self, obs_dim=1, hidden_dim=8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, obs):
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.net(s)

    def act(self, obs):
        logit = self.forward(obs)
        mean = torch.tanh(logit)
        action = mean + torch.randn(1) * 0.3
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = torch.distributions.Normal(mean, 0.3).log_prob(action)
        return float(action.item()), log_prob


def compute_metrics(cost_history, in_target_zone, target=1.0):
    costs = np.array(cost_history)
    in_zone = np.array(in_target_zone)
    return {
        "mean_cost": float(np.mean(costs)),
        "time_in_target": float(np.mean(in_zone)),
        "failure_rate": 1.0 - float(np.mean(in_zone[-100:])) if len(in_zone) >= 100 else 0.0,
        "recovery_time": compute_recovery_time(costs),
    }


def compute_recovery_time(costs, threshold=0.25, window=10):
    if len(costs) < window:
        return float(len(costs))
    for t in range(len(costs) - window):
        if np.mean(costs[t:t + window]) < threshold:
            return float(t + window)
    return float(len(costs))


def run_control_episode(env, policy, steps=500, mode="learned"):
    obs = env.reset()
    costs = []
    in_zone = []

    for _ in range(steps):
        if mode in ("learned", "frozen"):
            action, log_prob = policy.act(obs)
        elif mode == "random":
            action = float(np.random.randn()) * 0.5
            action = np.clip(action, -1.0, 1.0)
            log_prob = None
        else:
            action = 0.0
            log_prob = None

        o_next, reward, _ = env.step(action)
        costs.append(float(env.cost_history[-1]))
        in_zone.append(env.in_target_zone[-1])

        if mode == "learned" and log_prob is not None:
            policy_loss = -float(env.cost_history[-1]) * log_prob
            policy.optimizer.zero_grad()
            policy_loss.backward()
            policy.optimizer.step()

        obs = o_next

    return compute_metrics(env.cost_history, env.in_target_zone)


def run_control_scan(g_min=0.0, g_max=3.0, n_points=16, eta=0.5, seed=0):
    g_vals = np.linspace(g_min, g_max, n_points)
    modes = ["learned", "frozen", "random", "zero"]
    all_results = {m: [] for m in modes}

    print(f"\n=== CONTROL PHASE DIAGRAM: g ∈ [{g_min}, {g_max}] ===")
    header = f"{'g':>8} " + " ".join(f"{'cost('+m[:4]+')':>12}" for m in modes)
    print(header)
    print("-" * (8 + 12 * 4))

    for g in g_vals:
        drift = g / eta if g > 1e-8 else 1e-4
        row = [g]
        for mode in modes:
            reset_seed(seed)
            env = ControlDoubleWell(drift_rate=drift, alpha=0.1, target_state=1.0)
            policy = ActionPolicy(obs_dim=1, hidden_dim=8)
            if mode == "frozen":
                for p in policy.parameters():
                    p.requires_grad = False
            metrics = run_control_episode(env, policy, steps=500, mode=mode)
            metrics["g"] = float(g)
            all_results[mode].append(metrics)
            row.append(metrics["mean_cost"])
        print(f"{g:>8.2f} " + " ".join(f"{row[i]:>12.4f}" for i in range(1, 5)))

    return g_vals, all_results


def run_target_switch_test(seed=0):
    print(f"\n=== TARGET SWITCHING TEST ===")
    results = []
    for scenario in ["fast_switch", "slow_switch"]:
        reset_seed(seed)
        period = 50 if scenario == "fast_switch" else 200
        env = ControlDoubleWell(drift_rate=0.02, alpha=0.1, target_state=1.0,
                               flip_period=period)
        policy = ActionPolicy(obs_dim=1, hidden_dim=8)
        obs = env.reset()
        switch_costs = []
        for step in range(1000):
            if step % period == 0:
                env.set_target(-env.target_state)
                switch_start = step
            action, log_prob = policy.act(obs)
            o_next, reward, _ = env.step(action)
            if log_prob is not None:
                ploss = -float(env.cost_history[-1]) * log_prob
                policy.optimizer.zero_grad()
                ploss.backward()
                policy.optimizer.step()
            obs = o_next
            if step >= switch_start and step < switch_start + 20:
                switch_costs.append(float(env.cost_history[-1]))
        results.append({
            "scenario": scenario,
            "period": period,
            "switch_adaptation_cost": float(np.mean(switch_costs)) if switch_costs else 0.0,
        })

    for r in results:
        print(f"  {r['scenario']}: period={r['period']} adaptation_cost={r['switch_adaptation_cost']:.4f}")
    return results


if __name__ == "__main__":
    g_vals, all_results = run_control_scan(n_points=12)

    # Judgment
    learned_costs = [m["mean_cost"] for m in all_results["learned"]]
    random_costs = [m["mean_cost"] for m in all_results["random"]]
    frozen_costs = [m["mean_cost"] for m in all_results["frozen"]]

    if np.mean(learned_costs) < np.mean(random_costs):
        verdict = "TRUE CLOSED-LOOP CONTROL ✅ — learned beats random"
    else:
        verdict = "NOT A CONTROL SYSTEM ❌ — random matches or beats learned"

    print(f"\n=== VERDICT ===")
    print(f"  mean cost: learned={np.mean(learned_costs):.4f}  random={np.mean(random_costs):.4f}  frozen={np.mean(frozen_costs):.4f}")
    print(f"  {verdict}")

    # Target switch test
    switch_results = run_target_switch_test()

    # Save
    output = {
        "g_axis": g_vals.tolist(),
        "results": {m: [{k: v for k, v in r.items()} for r in all_results[m]] for m in all_results},
        "verdict": verdict,
        "target_switch": switch_results,
    }

    with open("results_final/control_phase_diagram.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Saved results_final/control_phase_diagram.json")
