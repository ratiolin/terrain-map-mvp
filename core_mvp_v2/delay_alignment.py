"""
3D sweep: (g, lambda, delay_k) at prediction horizon Delta.
Oscillating drift: g_t = g * sin(omega * t) with omega=1.0.
pred_loss uses state_{t+Delta} — prediction looks ahead.
ctrl_loss uses immediate step cost.
"""

import json, os, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

from core_mvp_v2.controller import Controller
from core_mvp_v2.run_mvp import reset_seed, routing_entropy, expert_utilization

# ─── parameters ────────────────────────────────────────────────────────────
ALPHA = 2.0; NOISE = 0.05; TARGET = 0.0
STATE_CLIP = 3.0; FORCE_SCALE = 0.1; KAPPA = 4.0
ROLLOUT_N = 10; EPISODES = 150; SEEDS = 3; IN_ZONE_RADIUS = 0.1
DRIFT_OMEGA = 1.0; PRED_DELTA = 2  # prediction horizon

G_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0]
LAMBDA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
DELAY_VALUES = [0, 1, 2, 5]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ═════════════════════════════════════════════════════════════════════════════
# Environment
# ═════════════════════════════════════════════════════════════════════════════


class OscillatingDriftWell(nn.Module):
    def __init__(self, g=0.0, noise_std=NOISE, alpha=ALPHA, target=TARGET,
                 state_clip=STATE_CLIP, force_scale=FORCE_SCALE, kappa=KAPPA,
                 omega=DRIFT_OMEGA, delay_k=0):
        super().__init__()
        self.g = g; self.noise_std = noise_std; self.alpha = alpha
        self.target = target; self.state_clip = state_clip
        self.force_scale = force_scale; self.kappa = kappa
        self.omega = omega; self.delay_k = delay_k
        self.state = torch.tensor(0.0); self.t = 0
        self.state_buffer = []

    def reset(self):
        self.state = torch.empty(1).uniform_(-2, 2)
        self.t = 0
        self.state_buffer = [self.state.clone().detach()]
        return self.observe()

    def _grad_potential(self, x):
        drift_t = self.g * np.sin(self.omega * self.t)
        return 4.0 * x**3 - 2.0 * (1.0 + drift_t) * x + self.kappa * torch.sign(x)

    def observe(self):
        idx = max(0, self.t - self.delay_k)
        idx = min(idx, len(self.state_buffer) - 1)
        return self.state_buffer[idx].detach().clone()

    def step(self, action):
        x = torch.clamp(self.state, -self.state_clip, self.state_clip)
        force = -self.force_scale * self._grad_potential(x)
        control = action * self.alpha
        eps = torch.randn(1)
        x_next = x + force + control + self.noise_std * eps
        x_next = torch.clamp(x_next, -self.state_clip, self.state_clip)
        cost = (x_next - self.target) ** 2
        self.state = x_next
        self.state_buffer.append(x_next.clone().detach())
        if len(self.state_buffer) > self.delay_k + PRED_DELTA + 2:
            self.state_buffer.pop(0)
        self.t += 1
        return cost

    def detach_state(self):
        self.state = self.state.detach()


# ═════════════════════════════════════════════════════════════════════════════
# Policy
# ═════════════════════════════════════════════════════════════════════════════


class TorchPolicy(nn.Module):
    def __init__(self, obs_dim=1, hidden_dim=16, lr=3e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Tanh())
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs):
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        return self.net(obs)


# ═════════════════════════════════════════════════════════════════════════════
# Training with prediction horizon Delta
# ═════════════════════════════════════════════════════════════════════════════


def train_episode_with_horizon(env, ctrl, policy, steps, lambda_ctrl):
    """Train with prediction horizon PRED_DELTA.
    pred_loss targets state_{t+Delta} (future prediction).
    ctrl_loss uses (state_{t+1} - target)^2 (immediate control)."""
    env.reset()

    # Buffer to hold upcoming steps for Delta-ahead prediction
    step_buffer = []  # list of (obs, action, cost_tensor, next_state, pred_target_state)
    obs = env.observe()

    cost_hist, state_hist, action_hist, mse_hist, align_hist = [], [], [], [], []

    for _ in range(steps):
        ctrl.maybe_split()

        action = policy(obs.unsqueeze(0)).squeeze(-1)
        cost_tensor = env.step(action)  # immediate cost
        next_state = env.state
        t_now = env.t

        obs_np = obs.detach().numpy()
        next_np = next_state.detach().numpy()
        action_np = float(action.detach().item())

        # Record step: (obs, action, cost_tensor, next_state, t)
        step_buffer.append({
            "obs": obs_np,
            "action": action_np,
            "cost_tensor": cost_tensor,
            "next_state": next_state,
            "t": t_now,
        })

        # When buffer has enough history, compute pred_loss targeting state_{t-Delta}
        if len(step_buffer) > PRED_DELTA:
            old_step = step_buffer.pop(0)
            # old_step was from t-Delta ago
            # pred_loss = MSE on forcasting state at current time using old_step's obs+action
            ctrl_obs = np.array([float(old_step["obs"][0])], dtype=np.float32)
            pred_target_val = float(obs_np[0])  # the state NOW (t) was the future at t-Delta

            # Controller prediction at time t-Delta
            z, _ = ctrl.route(ctrl_obs)
            weights = z.detach()
            K_cur = weights.size(-1)

            preds = []
            for m in ctrl.models:
                p = m(ctrl_obs, old_step["action"])
                preds.append(p)
            soft_pred_val = sum(float(weights[0, i]) * float(preds[i].detach().item())
                                for i in range(K_cur))
            pred_loss_val = float((soft_pred_val - pred_target_val) ** 2)

            ctrl.record_usage(int(weights.argmax().item()))
            for i in range(K_cur):
                ctrl.track_error(i, abs(float(preds[i].detach().item()) - pred_target_val))

            # Train experts + gating on the delayed (t-Delta) data
            ctrl.gating_optimizer.zero_grad()
            for m in ctrl.models:
                m.optimizer.zero_grad()
            target_t = torch.tensor([pred_target_val], dtype=torch.float32)
            gating_loss = torch.tensor(0.0)
            for i in range(K_cur):
                p_i = ctrl.models[i](ctrl_obs, old_step["action"])
                gating_loss = gating_loss + weights[0, i].detach() * ((p_i - target_t) ** 2).mean()
            gating_loss.backward()
            ctrl.gating_optimizer.step()
            for m in ctrl.models:
                m.optimizer.step()

            # ── gradient alignment ──
            policy_params = list(policy.parameters())

            # Differentiable pred_loss: old action → old env step → old next_state → ... → current state
            # For simplicity, use the current next_state as pred target (with grad)
            pred_loss_tensor = (next_state - torch.tensor(soft_pred_val, dtype=torch.float32)) ** 2
            pred_loss_tensor = pred_loss_tensor.mean()
            ctrl_loss_tensor = cost_tensor

            policy.optimizer.zero_grad()
            pred_loss_tensor.backward(retain_graph=True)
            gpred_parts = [p.grad.flatten().clone() for p in policy_params if p.grad is not None]
            gpred = torch.cat(gpred_parts) if gpred_parts else torch.zeros(1)

            policy.optimizer.zero_grad()
            ctrl_loss_tensor.backward(retain_graph=True)
            gctrl_parts = [p.grad.flatten().clone() for p in policy_params if p.grad is not None]
            gctrl = torch.cat(gctrl_parts) if gctrl_parts else torch.zeros(1)

            if gpred.numel() > 0 and gctrl.numel() > 0:
                align = float((torch.dot(gpred, gctrl) /
                              (gpred.norm() * gctrl.norm() + 1e-8)).item())
            else:
                align = 0.0

            # Train policy
            policy.optimizer.zero_grad()
            combined = ctrl_loss_tensor + lambda_ctrl * pred_loss_tensor
            combined.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy.optimizer.step()

            # Structural maintenance
            drift_val = env.g * np.sin(env.omega * env.t) if env.t > 1 else 0.0
            best_err = min(abs(float(preds[i].detach().item()) - pred_target_val) for i in range(K_cur))
            advantage_val = float(abs(soft_pred_val - pred_target_val)) - best_err
            ctrl.step_record(drift_val, advantage_val)
            ctrl.maybe_merge(); ctrl.maybe_prune()

            # Track
            mse_hist.append(pred_loss_val)
            align_hist.append(align)
        else:
            # Before buffer fills: just accumulate, train experts on current step
            ctrl_obs = np.array([float(obs_np[0])], dtype=np.float32)
            target_t = torch.tensor(next_np, dtype=torch.float32)
            z, _ = ctrl.route(ctrl_obs)
            weights = z.detach()
            K_cur = weights.size(-1)

            preds = [m(ctrl_obs, action_np) for m in ctrl.models]

            ctrl.gating_optimizer.zero_grad()
            for m in ctrl.models: m.optimizer.zero_grad()
            gating_loss = torch.tensor(0.0)
            for i in range(K_cur):
                p_i = ctrl.models[i](ctrl_obs, action_np)
                gating_loss = gating_loss + weights[0,i].detach() * ((p_i - target_t) ** 2).mean()
            gating_loss.backward()
            ctrl.gating_optimizer.step()
            for m in ctrl.models: m.optimizer.step()

            # Just train policy on control loss
            policy.optimizer.zero_grad()
            cost_tensor.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy.optimizer.step()

            soft_pred_val = float(np.mean([float(p.detach().item()) for p in preds]))
            pred_loss_val = float((soft_pred_val - float(next_np[0])) ** 2)
            mse_hist.append(pred_loss_val)
            align_hist.append(0.0)

        env.detach_state()

        # Track detached metrics
        cost_hist.append(float((float(next_state.detach().item()) - TARGET) ** 2))
        state_hist.append(float(next_state.detach().item()))
        action_hist.append(action_np)

        obs = env.observe()

    return cost_hist, state_hist, action_hist, mse_hist, align_hist


# ═════════════════════════════════════════════════════════════════════════════
# Single experiment
# ═════════════════════════════════════════════════════════════════════════════


def experiment_single(g, lam, delay_k, seed=0, silent=False):
    reset_seed(seed)
    env = OscillatingDriftWell(g=g, delay_k=delay_k)
    ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=1); ctrl.init()
    policy = TorchPolicy(obs_dim=1, hidden_dim=16, lr=3e-3)

    total_cost, total_state, total_action, total_align = [], [], [], []
    for _ in range(EPISODES):
        c_h, s_h, a_h, m_h, al_h = train_episode_with_horizon(env, ctrl, policy, ROLLOUT_N, lam)
        total_cost.extend(c_h); total_state.extend(s_h)
        total_action.extend(a_h); total_align.extend(al_h)

    tail = max(1, len(total_cost) // 3)
    mean_cost = float(np.mean(total_cost[-tail:]))
    in_zone = float(np.mean([1.0 if abs(s) < IN_ZONE_RADIUS else 0.0 for s in total_state[-tail:]]))
    slope = float(np.polyfit(total_state[-tail:], total_action[-tail:], 1)[0]) if len(total_state) > 1 else 0.0
    align_mean = float(np.mean(total_align))
    align_std = float(np.std(total_align))
    neg_frac = float(np.mean([1.0 if a < 0 else 0.0 for a in total_align]))

    if not silent:
        print(f"  g={g:.1f} lam={lam:.3f} k={delay_k}  "
              f"align={align_mean:+.4f}+/-{align_std:.3f}  neg={neg_frac:.3f}  "
              f"cost={mean_cost:.3f}  in_zone={in_zone:.3f}")

    return {"g": g, "lambda": lam, "delay_k": delay_k,
            "pred_delta": PRED_DELTA,
            "alignment_mean": align_mean, "alignment_std": align_std,
            "neg_fraction": neg_frac, "mean_cost": mean_cost,
            "in_zone_rate": in_zone, "policy_slope": slope}


# ═════════════════════════════════════════════════════════════════════════════
# 3D sweep
# ═════════════════════════════════════════════════════════════════════════════


def run_3d_sweep(g_values=None, lambda_values=None, delay_values=None, seeds=None):
    if g_values is None: g_values = G_VALUES
    if lambda_values is None: lambda_values = LAMBDA_VALUES
    if delay_values is None: delay_values = DELAY_VALUES
    if seeds is None: seeds = list(range(SEEDS))

    ng, nl, nk = len(g_values), len(lambda_values), len(delay_values)
    total = ng * nl * nk
    print(f"\n{'='*70}")
    print(f"3D SWEEP: g×λ×k  (Δ={PRED_DELTA}, ω={DRIFT_OMEGA})")
    print(f"  g in {g_values}")
    print(f"  λ in {lambda_values}")
    print(f"  k in {delay_values}")
    print(f"  total={total} × {len(seeds)} seeds")
    print(f"{'='*70}")

    all_results = {}
    count = 0
    for k_val in delay_values:
        print(f"\n─── delay k={k_val} ───")
        for g in g_values:
            for lam in lambda_values:
                count += 1
                runs = [experiment_single(g, lam, k_val, seed=s, silent=True) for s in seeds]
                avg = {}
                for key in ["alignment_mean", "alignment_std", "neg_fraction",
                            "mean_cost", "in_zone_rate", "policy_slope"]:
                    avg[key] = float(np.mean([r[key] for r in runs]))
                avg["g"] = g; avg["lambda"] = lam; avg["delay_k"] = k_val
                all_results[(g, lam, k_val)] = avg
                print(f"  [{count}/{total}] g={g:.1f} lam={lam:.3f} k={k_val}  "
                      f"align={avg['alignment_mean']:+.4f}  neg={avg['neg_fraction']:.3f}  "
                      f"cost={avg['mean_cost']:.3f}  in_zone={avg['in_zone_rate']:.3f}")
    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Analysis & Plotting
# ═════════════════════════════════════════════════════════════════════════════


def analyze_all(all_results, g_values, lambda_values, delay_values):
    ng, nl = len(g_values), len(lambda_values)
    analysis = {}
    for k_val in delay_values:
        am = np.zeros((ng, nl))
        for gi, g in enumerate(g_values):
            for li, lam in enumerate(lambda_values):
                am[gi, li] = all_results[(g, lam, k_val)]["alignment_mean"]
        sign_map = np.sign(am)
        ls_arr = np.full(ng, np.nan)
        for gi in range(ng):
            for li in range(nl - 1):
                if sign_map[gi, li] * sign_map[gi, li + 1] < 0:
                    a1, a2 = am[gi, li], am[gi, li + 1]
                    l1, l2 = lambda_values[li], lambda_values[li + 1]
                    ls_arr[gi] = l1 + (0 - a1) * (l2 - l1) / (a2 - a1 + 1e-8)
                    break
        any_flip = bool(not np.all(np.isnan(ls_arr)))
        analysis[k_val] = {
            "min_alignment": float(am.min()),
            "max_alignment": float(am.max()),
            "has_negative": bool(am.min() < 0),
            "n_negative_cells": int(np.sum(am < 0)),
            "has_sign_flip": any_flip,
            "lambda_star": {str(g_values[gi]): (float(ls_arr[gi]) if not np.isnan(ls_arr[gi]) else None)
                            for gi in range(ng)},
        }
    return analysis


def plot_all(all_results, g_values, lambda_values, delay_values, save_dir=OUTPUT_DIR):
    ng, nl, nk = len(g_values), len(lambda_values), len(delay_values)

    # Alignment heatmaps per delay
    fig, axes = plt.subplots(1, nk, figsize=(5*nk, 4))
    if nk == 1: axes = [axes]
    fig.suptitle(f"Alignment vs (g,λ) by delay k  (Δ={PRED_DELTA}, ω={DRIFT_OMEGA})", fontsize=12)

    for ki, k_val in enumerate(delay_values):
        am = np.zeros((ng, nl))
        for gi, g in enumerate(g_values):
            for li, lam in enumerate(lambda_values):
                am[gi, li] = all_results[(g, lam, k_val)]["alignment_mean"]
        v = max(abs(am.min()), abs(am.max()), 0.1)
        im = axes[ki].imshow(am.T, aspect="auto", origin="lower",
                              extent=[g_values[0], g_values[-1],
                                      lambda_values[0], lambda_values[-1]],
                              cmap="RdBu_r", vmin=-v, vmax=v)
        axes[ki].set_title(f"k={k_val}  min={am.min():+.3f}")
        axes[ki].set_xlabel("g"); axes[ki].set_ylabel("λ")
        plt.colorbar(im, ax=axes[ki], fraction=0.046)
    plt.tight_layout()
    sp = os.path.join(save_dir, "delay_alignment_heatmaps.png")
    plt.savefig(sp, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot saved: {sp}")

    # Delay effect summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Effect of delay k (Δ={PRED_DELTA})", fontsize=12)
    k_arr = np.array(delay_values)
    for gi, g in enumerate(g_values):
        neg = [np.mean([all_results[(g, lam, k)]["neg_fraction"] for lam in lambda_values])
               for k in delay_values]
        align = [np.mean([all_results[(g, lam, k)]["alignment_mean"] for lam in lambda_values])
                 for k in delay_values]
        ax1.plot(k_arr, neg, 'o-', label=f"g={g}")
        ax2.plot(k_arr, align, 'o-', label=f"g={g}")
    ax1.set_xlabel("delay k"); ax1.set_ylabel("mean neg_fraction")
    ax1.set_title("Negative fraction vs delay"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("delay k"); ax2.set_ylabel("mean alignment")
    ax2.set_title("Alignment vs delay"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    sp2 = os.path.join(save_dir, "delay_effect_summary.png")
    plt.savefig(sp2, dpi=150, bbox_inches="tight"); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seeds = list(range(SEEDS))
    all_results = run_3d_sweep(seeds=seeds)
    analysis = analyze_all(all_results, G_VALUES, LAMBDA_VALUES, DELAY_VALUES)

    print(f"\n{'='*60}")
    print("PER-DELAY ANALYSIS")
    print(f"{'='*60}")
    for k in DELAY_VALUES:
        a = analysis[k]
        flips = [f"g={g}={ls:.3f}" for g, ls in a["lambda_star"].items() if ls is not None]
        print(f"  k={k}: min_align={a['min_alignment']:+.4f}  "
              f"neg_cells={a['n_negative_cells']}  sign_flip={a['has_sign_flip']}")
        if flips: print(f"    lambda*: {', '.join(flips)}")

    any_negative = any(analysis[k]["has_negative"] for k in DELAY_VALUES)
    any_flip = any(analysis[k]["has_sign_flip"] for k in DELAY_VALUES)
    all_min = min(analysis[k]["min_alignment"] for k in DELAY_VALUES)

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    if any_flip:
        print(f"  SIGN FLIP EXISTS → lambda*(g,k) curves found")
        verdict = "sign_flip_found"
    elif any_negative:
        print(f"  NEGATIVE CELLS (min={all_min:+.4f}) but no crossing curve")
        verdict = "negative_cells_no_flip"
    elif all_min < 0.05:
        print(f"  NEAR-ZERO (min={all_min:+.4f}) — decoupling regime")
        verdict = "near_zero"
    else:
        print(f"  ALWAYS POSITIVE (min={all_min:+.4f})")
        verdict = "always_positive"

    plot_all(all_results, G_VALUES, LAMBDA_VALUES, DELAY_VALUES)

    output = {
        "experiment": "delay_alignment_with_horizon",
        "parameters": {"alpha": ALPHA, "noise": NOISE, "target": TARGET,
                       "drift_omega": DRIFT_OMEGA, "pred_delta": PRED_DELTA,
                       "episodes": EPISODES, "rollout_N": ROLLOUT_N, "seeds": SEEDS},
        "g_values": G_VALUES, "lambda_values": LAMBDA_VALUES,
        "delay_values": DELAY_VALUES, "pred_delta": PRED_DELTA,
        "analysis": analysis, "verdict": verdict,
        "results": {f"{r['g']},{r['lambda']},{r['delay_k']}":
                    {kr: r[kr] for kr in ["alignment_mean", "alignment_std",
                                           "neg_fraction", "mean_cost",
                                           "in_zone_rate", "policy_slope"]}
                    for r in all_results.values()},
    }
    path = os.path.join(OUTPUT_DIR, "delay_alignment_3d.json")
    with open(path, "w") as f: json.dump(output, f, indent=2)
    print(f"\n  Saved to {path}")


if __name__ == "__main__":
    main()
