"""
Gradient alignment experiment: 2D sweep over (g, lambda).
Computes cosine similarity between ∇pred_loss and ∇ctrl_loss
w.r.t. policy parameters at each training step.
Finds lambda*(g) where alignment crosses zero.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from core_mvp_v2.controller import Controller
from core_mvp_v2.metrics import (
    state_coverage,
    transition_entropy,
    visitation_uniformity,
    compute_dwell_time,
    compute_prediction_variance,
    compute_stability,
    classify_structure_type,
)
from core_mvp_v2.run_mvp import (
    reset_seed,
    routing_entropy,
    expert_utilization,
)
from core_mvp_v2.agent import Expert

# ─── fixed parameters ──────────────────────────────────────────────────────
ALPHA = 2.0
NOISE = 0.05
TARGET = 0.0
ROLLOUT_N = 10
EPISODES = 200
SEEDS = 3
IN_ZONE_RADIUS = 0.1
STATE_CLIP = 3.0
FORCE_SCALE = 0.1
KAPPA = 4.0

# Scan ranges
G_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
LAMBDA_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ═════════════════════════════════════════════════════════════════════════════
# Environment
# ═════════════════════════════════════════════════════════════════════════════


class TorchControlledWell(nn.Module):
    """Differentiable double-well + direct control + target tracking."""

    def __init__(self, kappa=KAPPA, drift_rate=0.0, noise_std=NOISE,
                 alpha=ALPHA, target=TARGET, state_clip=STATE_CLIP,
                 force_scale=FORCE_SCALE):
        super().__init__()
        self.kappa = kappa
        self.drift_rate = drift_rate
        self.noise_std = noise_std
        self.alpha = alpha
        self.target = target
        self.state_clip = state_clip
        self.force_scale = force_scale
        self.state = torch.tensor(0.0)
        self.t = 0
        self.drift = 0.0

    def reset(self):
        self.state = torch.empty(1).uniform_(-2, 2)
        self.t = 0
        self.drift = 0.0
        return self.state.detach()

    def _grad_potential(self, x):
        b = 1.0 + self.drift
        return 4.0 * x ** 3 - 2.0 * b * x + self.kappa * torch.sign(x)

    def step(self, action):
        self.drift = self.drift_rate * self.t
        x = torch.clamp(self.state, -self.state_clip, self.state_clip)
        force = -self.force_scale * self._grad_potential(x)
        control = action * self.alpha
        eps = torch.randn(1)
        noise = self.noise_std * eps
        x_next = x + force + control + noise
        x_next = torch.clamp(x_next, -self.state_clip, self.state_clip)
        cost = (x_next - self.target) ** 2
        self.state = x_next
        self.t += 1
        return cost

    def detach_state(self):
        self.state = self.state.detach()


# ═════════════════════════════════════════════════════════════════════════════
# Policy
# ═════════════════════════════════════════════════════════════════════════════


class TorchPolicy(nn.Module):
    """Deterministic continuous-action policy for BPTT training."""

    def __init__(self, obs_dim=1, hidden_dim=16, lr=3e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs):
        if obs.dim() == 0:
            obs = obs.unsqueeze(0)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)


# ═════════════════════════════════════════════════════════════════════════════
# Training with gradient alignment
# ═════════════════════════════════════════════════════════════════════════════


def train_one_episode_with_alignment(env, ctrl, policy, obs_dim, steps,
                                     lambda_ctrl):
    """
    Train policy + experts for *steps* with mixed loss.
    At each step, compute gradient alignment between pred_loss and ctrl_loss.
    Returns traces + alignment stats.
    """
    env.reset()
    state = env.state.detach()

    cost_hist = []
    state_hist = []
    action_hist = []
    mse_hist = []
    alignment_hist = []
    pred_loss_hist = []
    ctrl_loss_hist = []

    for _ in range(steps):
        ctrl.maybe_split()

        # ── forward: policy -> env ──
        action = policy(state.unsqueeze(0)).squeeze(-1)
        cost_tensor = env.step(action)
        next_state = env.state

        # detached copies for Controller
        state_np = state.detach().numpy()
        o_next_np = next_state.detach().numpy()
        action_np = float(action.detach().item())

        # ── Controller prediction (detached) ──
        target_t = torch.tensor(o_next_np, dtype=torch.float32)
        z, _ = ctrl.route(state_np)
        weights = z.detach()
        K_cur = weights.size(-1)

        preds = []
        for m in ctrl.models:
            p = m(state_np, action_np)
            preds.append(p)

        soft_pred_val = sum(float(weights[0, i]) * float(preds[i].detach().item())
                            for i in range(K_cur))
        pred_loss_val = float((soft_pred_val - float(o_next_np[0])) ** 2)

        ctrl.record_usage(int(weights.argmax().item()))
        for i in range(K_cur):
            ctrl.track_error(i, abs(float(preds[i].detach().item()) - float(o_next_np[0])))

        # ── train experts + gating (supervised, detached) ──
        ctrl.gating_optimizer.zero_grad()
        for m in ctrl.models:
            m.optimizer.zero_grad()
        gating_loss = torch.tensor(0.0)
        for i in range(K_cur):
            p_i = ctrl.models[i](state_np, action_np)
            gating_loss = gating_loss + weights[0, i].detach() * ((p_i - target_t) ** 2).mean()
        gating_loss.backward()
        ctrl.gating_optimizer.step()
        for m in ctrl.models:
            m.optimizer.step()

        # ── gradient alignment on policy params ──
        policy_params = list(policy.parameters())

        # Differentiable pred_loss: next_state depends on action through env
        pred_loss_tensor = (next_state - torch.tensor(soft_pred_val, dtype=torch.float32)) ** 2
        pred_loss_tensor = pred_loss_tensor.mean()

        # Differentiable ctrl_loss: already a tensor from env.step
        ctrl_loss_tensor = cost_tensor

        # grad_pred
        policy.optimizer.zero_grad()
        pred_loss_tensor.backward(retain_graph=True)
        grad_pred_parts = []
        for p in policy_params:
            if p.grad is not None:
                grad_pred_parts.append(p.grad.flatten().clone())
        if grad_pred_parts:
            grad_pred = torch.cat(grad_pred_parts)
        else:
            grad_pred = torch.zeros(1)

        # grad_ctrl
        policy.optimizer.zero_grad()
        ctrl_loss_tensor.backward(retain_graph=True)
        grad_ctrl_parts = []
        for p in policy_params:
            if p.grad is not None:
                grad_ctrl_parts.append(p.grad.flatten().clone())
        if grad_ctrl_parts:
            grad_ctrl = torch.cat(grad_ctrl_parts)
        else:
            grad_ctrl = torch.zeros(1)

        # cosine alignment
        dot = torch.dot(grad_pred, grad_ctrl)
        norm_prod = grad_pred.norm() * grad_ctrl.norm() + 1e-8
        alignment = float((dot / norm_prod).item())

        # ── normal training: combined loss ──
        policy.optimizer.zero_grad()
        combined_loss = ctrl_loss_tensor + lambda_ctrl * pred_loss_tensor
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy.optimizer.step()

        env.detach_state()

        # ── structural maintenance ──
        drift_val = env.drift if env.t > 1 else 0.0
        advantage_val = 0.0
        if K_cur >= 1:
            best_err = min(abs(float(preds[i].detach().item()) - float(o_next_np[0]))
                          for i in range(K_cur))
            advantage_val = float(abs(soft_pred_val - float(o_next_np[0]))) - best_err
        ctrl.step_record(drift_val, advantage_val)
        ctrl.maybe_merge()
        ctrl.maybe_prune()

        # ── track ──
        ctrl_loss_val = float((float(next_state.detach().item()) - env.target) ** 2)
        cost_hist.append(ctrl_loss_val)
        state_hist.append(float(next_state.detach().item()))
        action_hist.append(action_np)
        mse_hist.append(pred_loss_val)
        alignment_hist.append(alignment)
        pred_loss_hist.append(pred_loss_val)
        ctrl_loss_hist.append(ctrl_loss_val)

        state = next_state.detach()

    alignment_mean = float(np.mean(alignment_hist))
    alignment_std = float(np.std(alignment_hist))

    return (cost_hist, state_hist, action_hist, mse_hist,
            alignment_hist, pred_loss_hist, ctrl_loss_hist,
            alignment_mean, alignment_std)


def run_test_phase(ctrl, policy, env_factory, steps=200):
    """Evaluate trained system on a fresh environment."""
    env = env_factory()
    env.reset()
    state = env.state.detach()

    costs = []
    states = []
    actions = []
    mses = []
    ents = []

    for _ in range(steps):
        with torch.no_grad():
            action = policy(state.unsqueeze(0)).squeeze(-1)
        cost = env.step(action)
        next_state = env.state.detach()

        state_np = state.detach().numpy()
        action_np = float(action.detach().item())
        next_np = next_state.detach().numpy()

        with torch.no_grad():
            z, _ = ctrl.route(state_np)
        weights = z.detach()
        K_cur = weights.size(-1)

        preds = []
        with torch.no_grad():
            for m in ctrl.models:
                p = m(state_np, action_np)
                preds.append(p)
        soft_pred = sum(float(weights[0, i]) * float(preds[i].item()) for i in range(K_cur))
        mse_val = float((soft_pred - float(next_np[0])) ** 2)

        costs.append(float((float(next_state.item()) - TARGET) ** 2))
        states.append(float(next_state.item()))
        actions.append(action_np)
        mses.append(mse_val)
        ents.append(routing_entropy(weights))

        state = next_state

    return costs, states, actions, mses, ents


def train_single_expert_on_trajectory(states, actions, next_states, obs_dim=1,
                                      hidden_dim=4, epochs=10, lr=1e-3):
    model = Expert(obs_dim=obs_dim, hidden_dim=hidden_dim, lr=lr)
    n = len(states)
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            batch_s = np.array(states, dtype=np.float32)[idx]
            batch_a = np.array(actions, dtype=np.float32)[idx]
            batch_t = np.array(next_states, dtype=np.float32)[idx]
            model.optimizer.zero_grad()
            total_loss = torch.tensor(0.0)
            for s, a, t_ns in zip(batch_s, batch_a, batch_t):
                pred = model(np.array([float(s)], dtype=np.float32), float(a))
                total_loss = total_loss + ((pred - torch.tensor([t_ns], dtype=torch.float32)) ** 2).mean()
            total_loss.backward()
            model.optimizer.step()
    return model


def compute_S_adv_light(ctrl, test_states, test_actions, test_mses):
    var_m = compute_prediction_variance(test_mses)
    dwell_m = compute_dwell_time(test_states)
    S_multi = compute_stability(var_m, 1.0, dwell_m)

    train_n = len(test_states) // 2
    train_s = test_states[:train_n]
    train_a = test_actions[:train_n]
    train_ns = test_states[1:train_n + 1] + [test_states[-1]]
    test_s = test_states[train_n:]
    test_a = test_actions[train_n:]
    test_ns = test_states[train_n + 1:] + [test_states[-1]]

    single = train_single_expert_on_trajectory(train_s, train_a, train_ns)
    single_mses = []
    for s, a, t_ns in zip(test_s, test_a, test_ns):
        with torch.no_grad():
            pred = single(np.array([float(s)], dtype=np.float32), float(a))
            single_mses.append(float(((pred - torch.tensor([t_ns], dtype=torch.float32)) ** 2).mean().item()))

    var_s = compute_prediction_variance(single_mses)
    dwell_s = compute_dwell_time(test_s)
    S_single = compute_stability(var_s, 1.0, dwell_s)
    S_adv = S_multi / (S_single + 1e-8)
    return S_adv, S_multi, S_single


# ═════════════════════════════════════════════════════════════════════════════
# Single (g, lambda) experiment
# ═════════════════════════════════════════════════════════════════════════════


def experiment_single(g_val: float, lambda_val: float, seed: int = 0,
                      silent: bool = False) -> Dict:
    drift_rate = g_val
    reset_seed(seed)
    env = TorchControlledWell(drift_rate=drift_rate)

    ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=1)
    ctrl.init()
    policy = TorchPolicy(obs_dim=1, hidden_dim=16, lr=3e-3)

    total_cost = []
    total_state = []
    total_action = []
    total_mse = []
    total_alignment = []
    total_pred_loss = []
    total_ctrl_loss = []

    for _ in range(EPISODES):
        (c_h, s_h, a_h, m_h,
         al_h, pl_h, cl_h,
         al_mean, al_std) = train_one_episode_with_alignment(
            env, ctrl, policy, 1, ROLLOUT_N, lambda_val)
        total_cost.extend(c_h)
        total_state.extend(s_h)
        total_action.extend(a_h)
        total_mse.extend(m_h)
        total_alignment.extend(al_h)
        total_pred_loss.extend(pl_h)
        total_ctrl_loss.extend(cl_h)

    # ── test phase ──
    reset_seed(seed + 20000)
    env_factory = lambda: TorchControlledWell(drift_rate=drift_rate)
    t_costs, t_states, t_actions, t_mses, t_ents = run_test_phase(
        ctrl, policy, env_factory, steps=min(200, EPISODES * ROLLOUT_N))

    # ── metrics ──
    tail_len = max(1, len(total_cost) // 3)
    mean_cost = float(np.mean(total_cost[-tail_len:]))
    in_zone_rate = float(np.mean([1.0 if abs(s) < IN_ZONE_RADIUS else 0.0
                                   for s in total_state[-tail_len:]]))

    slope = float(np.polyfit(t_states, t_actions, 1)[0]) if len(t_states) > 1 else 0.0
    S_adv, S_multi, S_single = compute_S_adv_light(ctrl, t_states, t_actions, t_mses)

    alignment_mean = float(np.mean(total_alignment))
    alignment_std = float(np.std(total_alignment))

    cov = state_coverage(t_states)
    trans_ent = transition_entropy(t_states)

    if not silent:
        print(f"  g={g_val:.2f}  lambda={lambda_val:.3f}  "
              f"align={alignment_mean:+.4f}+/-{alignment_std:.4f}  "
              f"cost={mean_cost:.4f}  S_adv={S_adv:.4f}  in_zone={in_zone_rate:.3f}")

    return {
        "g": g_val,
        "lambda": lambda_val,
        "alignment_mean": alignment_mean,
        "alignment_std": alignment_std,
        "mean_cost": mean_cost,
        "S_adv": S_adv,
        "policy_slope": slope,
        "in_zone_rate": in_zone_rate,
        "coverage": cov,
        "transition_entropy": trans_ent,
        "n_experts": ctrl.n_models(),
        "alignment_series": total_alignment,
        "pred_loss_series": total_pred_loss,
        "ctrl_loss_series": total_ctrl_loss,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2D sweep
# ═════════════════════════════════════════════════════════════════════════════


def run_2d_sweep(g_values=None, lambda_values=None, seeds=None):
    if g_values is None:
        g_values = G_VALUES
    if lambda_values is None:
        lambda_values = LAMBDA_VALUES
    if seeds is None:
        seeds = list(range(SEEDS))

    ng = len(g_values)
    nl = len(lambda_values)
    total = ng * nl

    print(f"\n{'='*70}")
    print(f"2D GRADIENT ALIGNMENT SWEEP")
    print(f"  g in [{min(g_values)}, {max(g_values)}]  n={ng}")
    print(f"  lambda in [{min(lambda_values)}, {max(lambda_values)}]  n={nl}")
    print(f"  seeds={len(seeds)}  total runs={total * len(seeds)}")
    print(f"{'='*70}")

    results = {}
    alignment_map = np.zeros((ng, nl))
    count = 0

    for gi, g in enumerate(g_values):
        for li, lam in enumerate(lambda_values):
            count += 1
            print(f"\n[{count}/{total}] g={g:.2f} lambda={lam:.3f}")

            run_results = []
            for s in seeds:
                r = experiment_single(g, lam, seed=s, silent=True)
                run_results.append(r)

            # Average over seeds
            avg = {}
            for k in ["alignment_mean", "alignment_std", "mean_cost",
                      "S_adv", "policy_slope", "in_zone_rate",
                      "coverage", "transition_entropy", "n_experts"]:
                avg[k] = float(np.mean([rr[k] for rr in run_results]))
            avg["g"] = g
            avg["lambda"] = lam
            avg["per_seed_alignment"] = [rr["alignment_mean"] for rr in run_results]

            results[(g, lam)] = avg
            alignment_map[gi, li] = avg["alignment_mean"]

            print(f"  avg: align={avg['alignment_mean']:+.4f}+/-{avg['alignment_std']:.4f}  "
                  f"cost={avg['mean_cost']:.4f}  S_adv={avg['S_adv']:.3f}  "
                  f"in_zone={avg['in_zone_rate']:.3f}")

    return results, alignment_map, g_values, lambda_values


# ═════════════════════════════════════════════════════════════════════════════
# Analysis: find lambda*(g) boundary
# ═════════════════════════════════════════════════════════════════════════════


def find_lambda_star(alignment_map, g_values, lambda_values):
    """Find lambda where alignment crosses zero for each g."""
    ng = len(g_values)
    nl = len(lambda_values)
    sign_map = np.sign(alignment_map)
    lambda_star = np.full(ng, np.nan)

    for gi in range(ng):
        row = alignment_map[gi, :]
        sign_row = sign_map[gi, :]

        for li in range(nl - 1):
            if sign_row[li] * sign_row[li + 1] < 0:
                a1 = row[li]
                a2 = row[li + 1]
                lam1 = lambda_values[li]
                lam2 = lambda_values[li + 1]
                lam_star = lam1 + (0 - a1) * (lam2 - lam1) / (a2 - a1)
                lambda_star[gi] = lam_star
                break
            elif abs(row[li]) < 0.02:
                lambda_star[gi] = lambda_values[li]
                break

    return lambda_star


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════


def plot_alignment_heatmap(alignment_map, g_values, lambda_values,
                           lambda_star=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(alignment_map.T, aspect='auto', origin='lower',
                   extent=[g_values[0], g_values[-1],
                           lambda_values[0], lambda_values[-1]],
                   cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Gradient Alignment (cosine)')

    if lambda_star is not None:
        valid = ~np.isnan(lambda_star)
        if valid.any():
            ax.plot(np.array(g_values)[valid], lambda_star[valid],
                    'k.-', linewidth=2, markersize=8, label='lambda*(g)')

    ax.set_xlabel('g (drift)')
    ax.set_ylabel('lambda (control weight)')
    ax.set_title('Gradient Alignment: cos(grad_pred, grad_ctrl)')
    ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.5, label='lambda=0.05')
    if lambda_star is not None and ~np.isnan(lambda_star).all():
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Alignment heatmap saved to {save_path}")
    plt.close()


def plot_alignment_vs_lambda(results, g_values, lambda_values, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    selected_g = [g for g in g_values if g in [0.0, 0.2, 0.5, 1.0, 2.0, 3.0]][:6]

    for i, g in enumerate(selected_g):
        ax = axes[i // 3, i % 3]
        lam_vals = []
        align_vals = []
        align_stds = []
        for lam in lambda_values:
            if (g, lam) in results:
                r = results[(g, lam)]
                lam_vals.append(lam)
                align_vals.append(r["alignment_mean"])
                align_stds.append(r["alignment_std"])

        ax.errorbar(lam_vals, align_vals, yerr=align_stds, fmt='o-', markersize=4)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('lambda')
        ax.set_ylabel('alignment')
        ax.set_title(f'g={g}')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Gradient Alignment vs lambda at Fixed g', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Alignment vs lambda plot saved to {save_path}")
    plt.close()


def plot_lambda_star_curve(g_values, lambda_star, save_path=None):
    valid = ~np.isnan(lambda_star)
    if not valid.any():
        print("  No sign-flip boundary found. Skipping lambda* curve plot.")
        return

    g_arr = np.array(g_values)[valid]
    ls_arr = lambda_star[valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(g_arr, ls_arr, 'ko-', markersize=6)
    ax.set_xlabel('g (drift)')
    ax.set_ylabel('lambda* (crossing point)')
    ax.set_title('lambda*(g): Critical Control Weight vs Drift')
    ax.grid(True, alpha=0.3)

    if len(g_arr) > 1:
        for i in range(len(g_arr)):
            ax.annotate(f'{ls_arr[i]:.3f}', (g_arr[i], ls_arr[i]),
                        textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  lambda*(g) plot saved to {save_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seeds = list(range(SEEDS))

    # ── Run 2D sweep ──
    results, alignment_map, g_vals, lam_vals = run_2d_sweep(
        g_values=G_VALUES, lambda_values=LAMBDA_VALUES, seeds=seeds)

    # ── Find lambda*(g) boundary ──
    lambda_star = find_lambda_star(alignment_map, g_vals, lam_vals)

    print(f"\n{'='*60}")
    print("SIGN-FLIP BOUNDARY lambda*(g)")
    print(f"{'='*60}")
    print(f"{'g':>8} {'lambda*':>10}")
    print("-" * 20)
    for i, g in enumerate(g_vals):
        ls = lambda_star[i]
        if np.isnan(ls):
            print(f"{g:>8.2f} {'no crossing':>10}")
        else:
            print(f"{g:>8.2f} {ls:>10.4f}")

    # ── Plot ──
    plot_alignment_heatmap(alignment_map, g_vals, lam_vals, lambda_star,
                           save_path=os.path.join(OUTPUT_DIR, "alignment_heatmap.png"))
    plot_alignment_vs_lambda(results, g_vals, lam_vals,
                             save_path=os.path.join(OUTPUT_DIR, "alignment_vs_lambda.png"))
    plot_lambda_star_curve(g_vals, lambda_star,
                           save_path=os.path.join(OUTPUT_DIR, "lambda_star_curve.png"))

    # ── Save results ──
    output = {
        "experiment": "gradient_alignment_2d_sweep",
        "parameters": {
            "alpha": ALPHA, "noise": NOISE, "target": TARGET,
            "rollout_N": ROLLOUT_N, "episodes": EPISODES, "seeds": SEEDS,
        },
        "g_values": list(g_vals),
        "lambda_values": list(lam_vals),
        "lambda_star": {str(g): (float(ls) if not np.isnan(ls) else None)
                        for g, ls in zip(g_vals, lambda_star)},
        "alignment_map": alignment_map.tolist(),
        "results": {
            f"{r['g']},{r['lambda']}": {
                k: v for k, v in r.items()
                if k not in ["alignment_series", "pred_loss_series", "ctrl_loss_series"]
            }
            for r in results.values()
        },
    }

    output_path = os.path.join(OUTPUT_DIR, "gradient_alignment_2d.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {output_path}")

    return output


if __name__ == "__main__":
    main()
