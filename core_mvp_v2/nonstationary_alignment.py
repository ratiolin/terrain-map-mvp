"""
Non-stationary gradient alignment experiment.
Introduces: time-varying drift, oscillating target,
delayed observations, fragile features.
Checks whether gradient alignment ever goes negative.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict

from core_mvp_v2.controller import Controller
from core_mvp_v2.metrics import (
    state_coverage, transition_entropy,
    compute_dwell_time, compute_prediction_variance, compute_stability,
)
from core_mvp_v2.run_mvp import reset_seed, routing_entropy, expert_utilization
from core_mvp_v2.agent import Expert

# ─── parameters ────────────────────────────────────────────────────────────
ALPHA = 2.0
NOISE = 0.05
STATE_CLIP = 3.0
FORCE_SCALE = 0.1
KAPPA = 4.0
ROLLOUT_N = 10
EPISODES = 200
SEEDS = 3
IN_ZONE_RADIUS = 0.1

# Non-stationary params
DRIFT_OMEGA = 0.05          # drift oscillation freq
TARGET_AMPLITUDE = 1.0      # target oscillation amplitude
TARGET_OMEGA = 0.03         # target oscillation freq
DELAY_K = 2                 # observation delay steps
FRAGILE_P = 0.7             # probability feature is real signal

# Scan ranges (reduced for speed)
G_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0]
LAMBDA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ═════════════════════════════════════════════════════════════════════════════
# Non-stationary Environment
# ═════════════════════════════════════════════════════════════════════════════


class NonStationaryWell(nn.Module):
    """Double-well with time-varying drift, oscillating target,
    delayed observation, and fragile prediction features."""

    def __init__(self, g=0.0, noise_std=NOISE, alpha=ALPHA,
                 state_clip=STATE_CLIP, force_scale=FORCE_SCALE, kappa=KAPPA,
                 drift_omega=DRIFT_OMEGA, target_amp=TARGET_AMPLITUDE,
                 target_omega=TARGET_OMEGA, delay_k=DELAY_K, fragile_p=FRAGILE_P):
        super().__init__()
        self.g = g
        self.noise_std = noise_std
        self.alpha = alpha
        self.state_clip = state_clip
        self.force_scale = force_scale
        self.kappa = kappa
        self.drift_omega = drift_omega
        self.target_amp = target_amp
        self.target_omega = target_omega
        self.delay_k = delay_k
        self.fragile_p = fragile_p

        self.state = torch.tensor(0.0)
        self.t = 0
        self.state_history = []  # buffer for delayed obs
        self.true_feature = torch.tensor([0.0])

    def reset(self):
        self.state = torch.empty(1).uniform_(-2, 2)
        self.t = 0
        self.state_history = [self.state.clone().detach() for _ in range(self.delay_k + 1)]
        self.true_feature = torch.tensor([0.0])
        return self.observe()

    def _grad_potential(self, x):
        drift_t = self.g * np.sin(self.drift_omega * self.t)
        b = 1.0 + drift_t
        return 4.0 * x ** 3 - 2.0 * b * x + self.kappa * torch.sign(x)

    def _current_target(self):
        return self.target_amp * np.sin(self.target_omega * self.t)

    def observe(self):
        """Return observation with delay and fragile feature."""
        # Delayed state
        delay_idx = max(0, self.t - self.delay_k)
        delay_idx = min(delay_idx, len(self.state_history) - 1)
        delayed_state = self.state_history[delay_idx].detach().clone()

        if self.t > 0:
            obs = torch.cat([delayed_state, self.true_feature])
        else:
            obs = torch.cat([delayed_state, torch.tensor([0.0])])
        return obs

    def step(self, action):
        # Physics
        drift_t = self.g * np.sin(self.drift_omega * self.t)
        x = torch.clamp(self.state, -self.state_clip, self.state_clip)
        b = 1.0 + drift_t
        grad_pot = 4.0 * x ** 3 - 2.0 * b * x + self.kappa * torch.sign(x)
        force = -self.force_scale * grad_pot
        control = action * self.alpha
        eps = torch.randn(1)
        noise = self.noise_std * eps
        x_next = x + force + control + noise
        x_next = torch.clamp(x_next, -self.state_clip, self.state_clip)

        target = self._current_target()
        cost = (x_next - target) ** 2

        self.state = x_next
        self.state_history.append(x_next.clone().detach())
        if len(self.state_history) > self.delay_k + 5:
            self.state_history.pop(0)
        self.t += 1

        # Fragile feature: sometimes noise
        if np.random.random() < self.fragile_p:
            self.true_feature = torch.tensor([float(torch.tanh(x_next).item())])
        else:
            self.true_feature = torch.randn(1) * 0.3

        return cost, target

    def detach_state(self):
        self.state = self.state.detach()


# ═════════════════════════════════════════════════════════════════════════════
# Policy (takes 2D obs: delayed_state + feature)
# ═════════════════════════════════════════════════════════════════════════════


class TorchPolicy2D(nn.Module):
    def __init__(self, obs_dim=2, hidden_dim=16, lr=3e-3):
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
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)


# ═════════════════════════════════════════════════════════════════════════════
# Training with alignment (adapted for 2D obs)
# ═════════════════════════════════════════════════════════════════════════════


def train_episode_nonstationary(env, ctrl, policy, steps, lambda_ctrl):
    env.reset()
    obs = env.observe()
    cost_hist, state_hist, action_hist, mse_hist = [], [], [], []
    alignment_hist, pred_loss_hist, ctrl_loss_hist = [], [], []

    for _ in range(steps):
        ctrl.maybe_split()

        # ── policy forward ──
        action = policy(obs.unsqueeze(0)).squeeze(-1)
        cost_tensor, target = env.step(action)
        next_state = env.state

        # detached copies for Controller
        obs_np = obs.detach().numpy()
        next_np = next_state.detach().numpy()
        action_np = float(action.detach().item())

        # ── Controller prediction ──
        target_t = torch.tensor(next_np, dtype=torch.float32)
        ctrl_obs = np.array([float(obs_np[0])], dtype=np.float32)  # 1D for Controller
        z, _ = ctrl.route(ctrl_obs)
        weights = z.detach()
        K_cur = weights.size(-1)

        preds = []
        for m in ctrl.models:
            p = m(ctrl_obs, action_np)
            preds.append(p)

        soft_pred_val = sum(float(weights[0, i]) * float(preds[i].detach().item())
                            for i in range(K_cur))
        pred_loss_val = float((soft_pred_val - float(next_np[0])) ** 2)

        ctrl.record_usage(int(weights.argmax().item()))
        for i in range(K_cur):
            ctrl.track_error(i, abs(float(preds[i].detach().item()) - float(next_np[0])))

        # ── train experts + gating ──
        ctrl.gating_optimizer.zero_grad()
        for m in ctrl.models:
            m.optimizer.zero_grad()
        gating_loss = torch.tensor(0.0)
        for i in range(K_cur):
            p_i = ctrl.models[i](ctrl_obs, action_np)
            gating_loss = gating_loss + weights[0, i].detach() * ((p_i - target_t) ** 2).mean()
        gating_loss.backward()
        ctrl.gating_optimizer.step()
        for m in ctrl.models:
            m.optimizer.step()

        # ── gradient alignment on policy params ──
        policy_params = list(policy.parameters())

        pred_loss_tensor = (next_state - torch.tensor(soft_pred_val)) ** 2
        pred_loss_tensor = pred_loss_tensor.mean()
        ctrl_loss_tensor = cost_tensor

        policy.optimizer.zero_grad()
        pred_loss_tensor.backward(retain_graph=True)
        grad_pred = torch.cat([p.grad.flatten().clone() for p in policy_params if p.grad is not None])

        policy.optimizer.zero_grad()
        ctrl_loss_tensor.backward(retain_graph=True)
        grad_ctrl = torch.cat([p.grad.flatten().clone() for p in policy_params if p.grad is not None])

        if grad_pred.numel() > 0 and grad_ctrl.numel() > 0:
            alignment = float((torch.dot(grad_pred, grad_ctrl) /
                              (grad_pred.norm() * grad_ctrl.norm() + 1e-8)).item())
        else:
            alignment = 0.0

        # ── training ──
        policy.optimizer.zero_grad()
        combined_loss = ctrl_loss_tensor + lambda_ctrl * pred_loss_tensor
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy.optimizer.step()

        env.detach_state()

        # ── structural maintenance ──
        drift_val = env.g * np.sin(env.drift_omega * env.t) if env.t > 1 else 0.0
        advantage_val = 0.0
        if K_cur >= 1:
            best_err = min(abs(float(preds[i].detach().item()) - float(next_np[0])) for i in range(K_cur))
            advantage_val = float(abs(soft_pred_val - float(next_np[0]))) - best_err
        ctrl.step_record(drift_val, advantage_val)
        ctrl.maybe_merge()
        ctrl.maybe_prune()

        # ── track ──
        ctrl_loss_val = float((float(next_state.detach().item()) - target) ** 2)
        cost_hist.append(ctrl_loss_val)
        state_hist.append(float(next_state.detach().item()))
        action_hist.append(action_np)
        mse_hist.append(pred_loss_val)
        alignment_hist.append(alignment)
        pred_loss_hist.append(pred_loss_val)
        ctrl_loss_hist.append(ctrl_loss_val)

        obs = env.observe()

    return (cost_hist, state_hist, action_hist, mse_hist,
            alignment_hist, pred_loss_hist, ctrl_loss_hist)


# ═════════════════════════════════════════════════════════════════════════════
# Single (g, lambda) experiment
# ═════════════════════════════════════════════════════════════════════════════


def experiment_single(g_val, lambda_val, seed=0, silent=False):
    reset_seed(seed)
    env = NonStationaryWell(g=g_val)
    obs_dim = 2

    ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=1)
    ctrl.init()
    policy = TorchPolicy2D(obs_dim=obs_dim, hidden_dim=16, lr=3e-3)

    total_cost, total_state, total_action = [], [], []
    total_alignment, total_mse = [], []

    for _ in range(EPISODES):
        (c_h, s_h, a_h, m_h,
         al_h, pl_h, cl_h) = train_episode_nonstationary(
            env, ctrl, policy, ROLLOUT_N, lambda_val)
        total_cost.extend(c_h)
        total_state.extend(s_h)
        total_action.extend(a_h)
        total_mse.extend(m_h)
        total_alignment.extend(al_h)

    tail_len = max(1, len(total_cost) // 3)
    mean_cost = float(np.mean(total_cost[-tail_len:]))
    in_zone_rate = float(np.mean([1.0 if abs(s) < IN_ZONE_RADIUS else 0.0
                                   for s in total_state[-tail_len:]]))
    if len(total_state) > 1 and len(total_action) > 1:
        slope = float(np.polyfit(total_state[-tail_len:], total_action[-tail_len:], 1)[0])
    else:
        slope = 0.0

    alignment_mean = float(np.mean(total_alignment))
    alignment_std = float(np.std(total_alignment))

    # Fraction of steps with negative alignment
    neg_frac = float(np.mean([1.0 if a < 0 else 0.0 for a in total_alignment]))

    if not silent:
        print(f"  g={g_val:.1f}  lambda={lambda_val:.3f}  "
              f"align={alignment_mean:+.4f}+/-{alignment_std:.3f}  "
              f"neg_frac={neg_frac:.3f}  cost={mean_cost:.3f}  in_zone={in_zone_rate:.3f}")

    return {
        "g": g_val, "lambda": lambda_val,
        "alignment_mean": alignment_mean,
        "alignment_std": alignment_std,
        "neg_fraction": neg_frac,
        "mean_cost": mean_cost,
        "in_zone_rate": in_zone_rate,
        "policy_slope": slope,
        "alignment_series": total_alignment,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Sweeps
# ═════════════════════════════════════════════════════════════════════════════


def run_lambda_sweep(g_fixed=1.0, lambda_values=None, seeds=None):
    if lambda_values is None:
        lambda_values = LAMBDA_VALUES
    if seeds is None:
        seeds = list(range(SEEDS))

    print(f"\n{'='*60}")
    print(f"LAMBDA SWEEP at g={g_fixed}")
    print(f"  Non-stationary: delay_k={DELAY_K}, fragile_p={FRAGILE_P}")
    print(f"{'='*60}")

    results = []
    for lam in lambda_values:
        run_results = []
        for s in seeds:
            r = experiment_single(g_fixed, lam, seed=s, silent=True)
            run_results.append(r)

        avg = {k: float(np.mean([rr[k] for rr in run_results]))
               for k in ["alignment_mean", "alignment_std", "neg_fraction",
                         "mean_cost", "in_zone_rate", "policy_slope"]}
        avg["g"] = g_fixed
        avg["lambda"] = lam
        results.append(avg)
        print(f"  lam={lam:.3f}  align={avg['alignment_mean']:+.4f}  "
              f"neg={avg['neg_fraction']:.3f}  cost={avg['mean_cost']:.3f}  "
              f"in_zone={avg['in_zone_rate']:.3f}  slope={avg['policy_slope']:+.4f}")

    return results


def run_2d_sweep(g_values=None, lambda_values=None, seeds=None):
    if g_values is None:
        g_values = G_VALUES
    if lambda_values is None:
        lambda_values = LAMBDA_VALUES
    if seeds is None:
        seeds = list(range(SEEDS))

    ng, nl = len(g_values), len(lambda_values)
    total = ng * nl

    print(f"\n{'='*70}")
    print(f"2D NON-STATIONARY ALIGNMENT SWEEP")
    print(f"  g in {g_values}  n={ng}")
    print(f"  lambda in {lambda_values}  n={nl}")
    print(f"  seeds={len(seeds)}  delay_k={DELAY_K}  fragile_p={FRAGILE_P}")
    print(f"  total runs={total * len(seeds)}")
    print(f"{'='*70}")

    results = {}
    alignment_map = np.zeros((ng, nl))
    negfrac_map = np.zeros((ng, nl))
    cost_map = np.zeros((ng, nl))
    count = 0

    for gi, g in enumerate(g_values):
        for li, lam in enumerate(lambda_values):
            count += 1
            run_results = []
            for s in seeds:
                r = experiment_single(g, lam, seed=s, silent=True)
                run_results.append(r)

            avg = {k: float(np.mean([rr[k] for rr in run_results]))
                   for k in ["alignment_mean", "alignment_std", "neg_fraction",
                             "mean_cost", "in_zone_rate", "policy_slope"]}
            avg["g"] = g
            avg["lambda"] = lam
            results[(g, lam)] = avg
            alignment_map[gi, li] = avg["alignment_mean"]
            negfrac_map[gi, li] = avg["neg_fraction"]
            cost_map[gi, li] = avg["mean_cost"]

            print(f"  [{count}/{total}] g={g:.1f} lam={lam:.3f}  "
                  f"align={avg['alignment_mean']:+.4f}  "
                  f"neg={avg['neg_fraction']:.3f}  "
                  f"cost={avg['mean_cost']:.3f}  "
                  f"in_zone={avg['in_zone_rate']:.3f}")

    return results, alignment_map, negfrac_map, cost_map, g_values, lambda_values


# ═════════════════════════════════════════════════════════════════════════════
# Analysis
# ═════════════════════════════════════════════════════════════════════════════


def analyze_sign_flip(alignment_map, g_values, lambda_values):
    """Check for sign flip (negative alignment) and near-zero alignment."""
    ng, nl = len(g_values), len(lambda_values)

    has_negative = alignment_map.min() < 0
    negative_points = []
    near_zero_points = []

    for gi in range(ng):
        for li in range(nl):
            val = alignment_map[gi, li]
            if val < 0:
                negative_points.append((g_values[gi], lambda_values[li], val))
            elif abs(val) < 0.1:
                near_zero_points.append((g_values[gi], lambda_values[li], val))

    # Check for sign-flip boundary
    sign_map = np.sign(alignment_map)
    lambda_star = np.full(ng, np.nan)

    for gi in range(ng):
        row = alignment_map[gi, :]
        sign_row = sign_map[gi, :]
        for li in range(nl - 1):
            if sign_row[li] * sign_row[li + 1] < 0:
                a1, a2 = row[li], row[li + 1]
                lam1, lam2 = lambda_values[li], lambda_values[li + 1]
                lambda_star[gi] = lam1 + (0 - a1) * (lam2 - lam1) / (a2 - a1 + 1e-8)
                break

    return {
        "has_negative": bool(has_negative),
        "negative_points": [(float(g), float(l), float(v)) for g, l, v in negative_points],
        "near_zero_points": [(float(g), float(l), float(v)) for g, l, v in near_zero_points[:10]],
        "n_negative": len(negative_points),
        "n_near_zero": len(near_zero_points),
        "lambda_star": {str(g): (float(ls) if not np.isnan(ls) else None)
                        for g, ls in zip(g_values, lambda_star)},
        "min_alignment": float(alignment_map.min()),
        "max_alignment": float(alignment_map.max()),
    }


def plot_nonstationary_results(alignment_map, negfrac_map, g_values, lambda_values,
                               save_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Non-Stationary Gradient Alignment (delayed+oscillating+fragile)", fontsize=12)

    im0 = axes[0].imshow(alignment_map.T, aspect="auto", origin="lower",
                          extent=[g_values[0], g_values[-1],
                                  lambda_values[0], lambda_values[-1]],
                          cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Alignment (cosine)")
    axes[0].set_xlabel("g"); axes[0].set_ylabel("lambda")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(negfrac_map.T * 100, aspect="auto", origin="lower",
                          extent=[g_values[0], g_values[-1],
                                  lambda_values[0], lambda_values[-1]],
                          cmap="Reds", vmin=0, vmax=50)
    axes[1].set_title("Negative Alignment %")
    axes[1].set_xlabel("g"); axes[1].set_ylabel("lambda")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    sp = os.path.join(save_dir, "nonstationary_alignment.png")
    plt.savefig(sp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {sp}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seeds = list(range(SEEDS))

    # ── Step 6: lambda sweep at fixed g ──
    print("\n>>> STEP 6: Lambda sweep at g=1.0")
    lam_results = run_lambda_sweep(g_fixed=1.0, seeds=seeds)

    # ── Step 7: (g, lambda) grid ──
    print("\n>>> STEP 7: 2D grid sweep")
    results, align_map, negfrac_map, cost_map, g_vals, lam_vals = run_2d_sweep(seeds=seeds)

    # ── Step 8-9: gradient alignment analysis ──
    print("\n>>> STEP 8-9: Alignment analysis")
    analysis = analyze_sign_flip(align_map, g_vals, lam_vals)

    print(f"\n{'='*60}")
    print("ALIGNMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"  Has negative alignment:  {analysis['has_negative']}")
    print(f"  N negative points:       {analysis['n_negative']}")
    print(f"  N near-zero (<0.1):      {analysis['n_near_zero']}")
    print(f"  Min alignment:           {analysis['min_alignment']:+.4f}")
    print(f"  Max alignment:           {analysis['max_alignment']:+.4f}")
    print()

    if analysis["negative_points"]:
        print("  NEGATIVE ALIGNMENT POINTS:")
        for g, l, v in analysis["negative_points"][:15]:
            print(f"    g={g:.1f}  lambda={l:.3f}  alignment={v:+.4f}")
    elif analysis["near_zero_points"]:
        print("  NEAR-ZERO ALIGNMENT POINTS:")
        for g, l, v in analysis["near_zero_points"][:10]:
            print(f"    g={g:.1f}  lambda={l:.3f}  alignment={v:+.4f}")
    else:
        print("  All alignment values are strongly positive.")

    print(f"\n  lambda*(g) sign-flip boundary:")
    any_flip = False
    for g, ls in analysis["lambda_star"].items():
        if ls is not None:
            any_flip = True
            print(f"    g={g}  lambda*={ls:.4f}")
        else:
            print(f"    g={g}  no crossing")
    if not any_flip:
        print("  (no sign-flip boundary detected)")

    # ── Step 10-12: verdict ──
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    if analysis["has_negative"] and analysis["n_negative"] > 3:
        print("  SIGN FLIP EXISTS")
        print("  Non-stationary environment induces gradient opposition.")
        print("  lambda*(g) exists -> switching criterion is viable.")
        verdict = "sign_flip_found"
    elif analysis["n_near_zero"] > 5:
        print("  WEAK COUPLING (alignment ~ 0)")
        print("  Objectives decouple under non-stationarity.")
        print("  Trade-off via decoupling, not opposition.")
        verdict = "weak_coupling"
    else:
        print("  ALWAYS POSITIVE")
        print("  Even under non-stationary stress, gradient alignment stays positive.")
        print("  Two objectives remain compatible — the conflict is distribution-level, not gradient-level.")
        verdict = "always_positive"

    # ── Plot ──
    plot_nonstationary_results(align_map, negfrac_map, g_vals, lam_vals)

    # ── Save ──
    output = {
        "experiment": "nonstationary_alignment",
        "parameters": {
            "alpha": ALPHA, "noise": NOISE,
            "delay_k": DELAY_K, "fragile_p": FRAGILE_P,
            "drift_omega": DRIFT_OMEGA, "target_amp": TARGET_AMPLITUDE,
            "target_omega": TARGET_OMEGA, "episodes": EPISODES,
            "rollout_N": ROLLOUT_N, "seeds": SEEDS,
        },
        "g_values": list(g_vals),
        "lambda_values": list(lam_vals),
        "alignment_map": align_map.tolist(),
        "negfrac_map": negfrac_map.tolist(),
        "analysis": analysis,
        "verdict": verdict,
        "lambda_sweep_g1": lam_results,
        "grid_results": {
            f"{r['g']},{r['lambda']}": {
                k: v for k, v in r.items()
                if k not in ["alignment_series"]
            }
            for r in results.values()
        },
    }

    output_path = os.path.join(OUTPUT_DIR, "nonstationary_alignment.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
