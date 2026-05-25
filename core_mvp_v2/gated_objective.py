"""
Gated objective selection experiment.
At each step, a learned gate chooses: prediction or control.
Uses straight-through estimator for binary gating.
Scans (g, lambda, delay_k) to find strategy-switching boundaries.
"""

import json, os, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core_mvp_v2.controller import Controller
from core_mvp_v2.run_mvp import reset_seed, routing_entropy, expert_utilization

# ─── parameters ────────────────────────────────────────────────────────────
ALPHA = 2.0; NOISE = 0.05; TARGET = 0.0
STATE_CLIP = 3.0; FORCE_SCALE = 0.1; KAPPA = 4.0
ROLLOUT_N = 10; EPISODES = 200; SEEDS = 3; IN_ZONE_RADIUS = 0.1
DRIFT_OMEGA = 1.0; PRED_DELTA = 2; SWITCH_BETA = 0.05

G_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0]
LAMBDA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
DELAY_VALUES = [0, 1, 2, 5]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ═════════════════════════════════════════════════════════════════════════════
# Environment (same as delay_alignment)
# ═════════════════════════════════════════════════════════════════════════════


class OscillatingDriftWell(nn.Module):
    def __init__(self, g=0.0, delay_k=0):
        super().__init__()
        self.g = g; self.noise_std = NOISE; self.alpha = ALPHA
        self.target = TARGET; self.state_clip = STATE_CLIP
        self.force_scale = FORCE_SCALE; self.kappa = KAPPA
        self.omega = DRIFT_OMEGA; self.delay_k = delay_k
        self.state = torch.tensor(0.0); self.t = 0; self.state_buffer = []

    def reset(self):
        self.state = torch.empty(1).uniform_(-2, 2); self.t = 0
        self.state_buffer = [self.state.clone().detach()]
        return self.observe()

    def _grad_potential(self, x):
        drift_t = self.g * np.sin(self.omega * self.t)
        return 4.0*x**3 - 2.0*(1.0+drift_t)*x + self.kappa*torch.sign(x)

    def observe(self):
        idx = max(0, self.t - self.delay_k)
        idx = min(idx, len(self.state_buffer) - 1)
        return self.state_buffer[idx].detach().clone()

    def step(self, action):
        x = torch.clamp(self.state, -self.state_clip, self.state_clip)
        force = -self.force_scale * self._grad_potential(x)
        control = action * self.alpha
        x_next = x + force + control + self.noise_std * torch.randn(1)
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
# Policy + Gate
# ═════════════════════════════════════════════════════════════════════════════


class GatedPolicy(nn.Module):
    """Policy with binary objective gate (straight-through estimator)."""

    def __init__(self, obs_dim=1, hidden_dim=16, lr=3e-3):
        super().__init__()
        # Action network
        self.act_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Tanh())
        # Gate network (separate small head)
        self.gate_head = nn.Sequential(
            nn.Linear(obs_dim, 8), nn.ReLU(),
            nn.Linear(8, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward_action(self, obs):
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        return self.act_net(obs)

    def forward_gate(self, obs):
        """Returns hard gate (0/1) with ST gradient and probability."""
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        logit = self.gate_head(obs)
        prob = torch.sigmoid(logit)
        gate_hard = (prob > 0.5).float()
        gate = gate_hard.detach() - prob.detach() + prob  # straight-through
        return gate, prob


# ═════════════════════════════════════════════════════════════════════════════
# Training with gated objectives
# ═════════════════════════════════════════════════════════════════════════════


def train_episode_gated(env, ctrl, policy, steps, lambda_ctrl):
    env.reset()
    obs = env.observe()

    cost_hist, state_hist, action_hist, mse_hist, align_hist = [], [], [], [], []
    gate_hist = []
    step_buffer = []
    prev_gate = torch.tensor(0.0)  # for switch cost

    for step_i in range(steps):
        ctrl.maybe_split()

        # ── gate selection ──
        gate, prob = policy.forward_gate(obs)
        gate_val = float(gate.item())
        prob_val = float(prob.item())

        # ── action ──
        action = policy.forward_action(obs).squeeze(-1)
        cost_tensor = env.step(action)
        next_state = env.state
        t_now = env.t

        obs_np = obs.detach().numpy()
        next_np = next_state.detach().numpy()
        action_np = float(action.detach().item())

        # Record step in buffer for Delta-ahead prediction
        step_buffer.append({
            "obs": obs_np, "action": action_np,
            "cost_tensor": cost_tensor, "next_state": next_state, "t": t_now,
        })

        # ── Process delayed prediction ──
        if len(step_buffer) > PRED_DELTA:
            old_step = step_buffer.pop(0)
            ctrl_obs = np.array([float(old_step["obs"][0])], dtype=np.float32)
            pred_target_val = float(obs_np[0])

            z, _ = ctrl.route(ctrl_obs)
            weights = z.detach(); K_cur = weights.size(-1)
            preds = []
            for m in ctrl.models:
                p = m(ctrl_obs, old_step["action"])
                preds.append(p)
            soft_pred_val = sum(float(weights[0,i]) * float(preds[i].detach().item()) for i in range(K_cur))
            pred_loss_val = float((soft_pred_val - pred_target_val) ** 2)

            ctrl.record_usage(int(weights.argmax().item()))
            for i in range(K_cur):
                ctrl.track_error(i, abs(float(preds[i].detach().item()) - pred_target_val))

            # Train experts + gating (always train on prediction)
            ctrl.gating_optimizer.zero_grad()
            for m in ctrl.models: m.optimizer.zero_grad()
            target_t = torch.tensor([pred_target_val], dtype=torch.float32)
            gating_loss = torch.tensor(0.0)
            for i in range(K_cur):
                p_i = ctrl.models[i](ctrl_obs, old_step["action"])
                gating_loss = gating_loss + weights[0,i].detach() * ((p_i - target_t) ** 2).mean()
            gating_loss.backward()
            ctrl.gating_optimizer.step()
            for m in ctrl.models: m.optimizer.step()

            # ── Gated loss for policy ──
            pred_loss_tensor = (next_state - torch.tensor(soft_pred_val, dtype=torch.float32)) ** 2
            pred_loss_tensor = pred_loss_tensor.mean()
            ctrl_loss_tensor = cost_tensor

            # Switch cost
            switch_cost = (gate - prev_gate).abs()

            # Gated loss: gate selects which objective
            gated_loss = (1.0 - gate) * pred_loss_tensor + lambda_ctrl * gate * ctrl_loss_tensor
            gated_loss = gated_loss + SWITCH_BETA * switch_cost

            # ── Gradient alignment (on gated loss components) ──
            policy_params = list(policy.parameters())

            policy.optimizer.zero_grad()
            pred_loss_tensor.backward(retain_graph=True)
            gpred = torch.cat([p.grad.flatten().clone() for p in policy_params if p.grad is not None])

            policy.optimizer.zero_grad()
            ctrl_loss_tensor.backward(retain_graph=True)
            gctrl = torch.cat([p.grad.flatten().clone() for p in policy_params if p.grad is not None])

            if gpred.numel() > 0 and gctrl.numel() > 0:
                align = float((torch.dot(gpred, gctrl) /
                              (gpred.norm() * gctrl.norm() + 1e-8)).item())
            else:
                align = 0.0

            # ── Train policy + gate ──
            policy.optimizer.zero_grad()
            gated_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy.optimizer.step()

            # Structural maintenance
            drift_val = env.g * np.sin(env.omega * env.t) if env.t > 1 else 0.0
            best_err = min(abs(float(preds[i].detach().item()) - pred_target_val) for i in range(K_cur))
            advantage_val = float(abs(soft_pred_val - pred_target_val)) - best_err
            ctrl.step_record(drift_val, advantage_val)
            ctrl.maybe_merge(); ctrl.maybe_prune()

            mse_hist.append(pred_loss_val)
            align_hist.append(align)
            gate_hist.append(gate_val)
        else:
            # Buffer not full: just train on control for bootstrapping
            ctrl_obs = np.array([float(obs_np[0])], dtype=np.float32)
            target_t = torch.tensor(next_np, dtype=torch.float32)
            z, _ = ctrl.route(ctrl_obs)
            weights = z.detach(); K_cur = weights.size(-1)
            preds = [m(ctrl_obs, action_np) for m in ctrl.models]

            ctrl.gating_optimizer.zero_grad()
            for m in ctrl.models: m.optimizer.zero_grad()
            gl = torch.tensor(0.0)
            for i in range(K_cur):
                p_i = ctrl.models[i](ctrl_obs, action_np)
                gl = gl + weights[0,i].detach() * ((p_i - target_t) ** 2).mean()
            gl.backward()
            ctrl.gating_optimizer.step()
            for m in ctrl.models: m.optimizer.step()

            # Just train on control + small switch cost
            policy.optimizer.zero_grad()
            sw = (gate - prev_gate).abs()
            (cost_tensor + SWITCH_BETA * sw).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy.optimizer.step()

            soft_pred_val = float(np.mean([float(p.detach().item()) for p in preds]))
            pred_loss_val = float((soft_pred_val - float(next_np[0])) ** 2)
            mse_hist.append(pred_loss_val)
            align_hist.append(0.0)
            gate_hist.append(1.0)  # bootstrap: assume control

        env.detach_state()
        prev_gate = gate.detach()

        cost_hist.append(float((float(next_state.detach().item()) - TARGET) ** 2))
        state_hist.append(float(next_state.detach().item()))
        action_hist.append(action_np)
        obs = env.observe()

    return cost_hist, state_hist, action_hist, mse_hist, align_hist, gate_hist


# ═════════════════════════════════════════════════════════════════════════════
# Single experiment
# ═════════════════════════════════════════════════════════════════════════════


def experiment_single(g, lam, delay_k, seed=0, silent=False):
    reset_seed(seed)
    env = OscillatingDriftWell(g=g, delay_k=delay_k)
    ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=1); ctrl.init()
    policy = GatedPolicy(obs_dim=1, hidden_dim=16, lr=3e-3)

    total_cost, total_state, total_action, total_align, total_gate = [], [], [], [], []
    for _ in range(EPISODES):
        c_h, s_h, a_h, m_h, al_h, g_h = train_episode_gated(env, ctrl, policy, ROLLOUT_N, lam)
        total_cost.extend(c_h); total_state.extend(s_h)
        total_action.extend(a_h); total_align.extend(al_h)
        total_gate.extend(g_h)

    tail = max(1, len(total_cost) // 3)
    mean_cost = float(np.mean(total_cost[-tail:]))
    in_zone = float(np.mean([1.0 if abs(s) < IN_ZONE_RADIUS else 0.0 for s in total_state[-tail:]]))
    slope = float(np.polyfit(total_state[-tail:], total_action[-tail:], 1)[0]) if len(total_state) > 1 else 0.0
    align_mean = float(np.mean(total_align))
    align_std = float(np.std(total_align))
    neg_frac = float(np.mean([1.0 if a < 0 else 0.0 for a in total_align]))
    exec_ratio = float(np.mean(total_gate))  # fraction of steps using control
    gate_std = float(np.std(total_gate))

    if not silent:
        print(f"  g={g:.1f} lam={lam:.3f} k={delay_k}  "
              f"exec={exec_ratio:.3f}  align={align_mean:+.4f}  "
              f"cost={mean_cost:.3f}  in_zone={in_zone:.3f}")

    return {"g": g, "lambda": lam, "delay_k": delay_k,
            "exec_ratio": exec_ratio, "gate_std": gate_std,
            "alignment_mean": align_mean, "alignment_std": align_std,
            "neg_fraction": neg_frac, "mean_cost": mean_cost,
            "in_zone_rate": in_zone, "policy_slope": slope,
            "gate_series": total_gate, "alignment_series": total_align}


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
    print(f"GATED OBJECTIVE 3D SWEEP (Δ={PRED_DELTA}, ω={DRIFT_OMEGA}, β={SWITCH_BETA})")
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
                for key in ["exec_ratio", "gate_std", "alignment_mean", "alignment_std",
                            "neg_fraction", "mean_cost", "in_zone_rate", "policy_slope"]:
                    avg[key] = float(np.mean([r[key] for r in runs]))
                avg["g"] = g; avg["lambda"] = lam; avg["delay_k"] = k_val
                all_results[(g, lam, k_val)] = avg
                print(f"  [{count}/{total}] g={g:.1f} lam={lam:.3f} k={k_val}  "
                      f"exec={avg['exec_ratio']:.3f}  cost={avg['mean_cost']:.3f}  "
                      f"align={avg['alignment_mean']:+.3f}")
    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Analysis & Plotting
# ═════════════════════════════════════════════════════════════════════════════


def analyze_all(all_results, g_values, lambda_values, delay_values):
    ng, nl = len(g_values), len(lambda_values)
    analysis = {}
    for k_val in delay_values:
        em = np.zeros((ng, nl))  # exec_ratio map
        am = np.zeros((ng, nl))  # alignment map
        cm = np.zeros((ng, nl))  # cost map
        for gi, g in enumerate(g_values):
            for li, lam in enumerate(lambda_values):
                r = all_results[(g, lam, k_val)]
                em[gi, li] = r["exec_ratio"]
                am[gi, li] = r["alignment_mean"]
                cm[gi, li] = r["mean_cost"]

        # Find exec_ratio jump along lambda for each g
        lambda_star = np.full(ng, np.nan)
        for gi in range(ng):
            row = em[gi, :]
            diffs = np.abs(np.diff(row))
            li_max = int(np.argmax(diffs))
            if diffs[li_max] > 0.05:  # significant jump
                lambda_star[gi] = float(lambda_values[li_max + 1])

        analysis[k_val] = {
            "exec_ratio_min": float(em.min()),
            "exec_ratio_max": float(em.max()),
            "exec_ratio_range": float(em.max() - em.min()),
            "lambda_star_exec": {str(g_values[gi]): (float(lambda_star[gi]) if not np.isnan(lambda_star[gi]) else None)
                                 for gi in range(ng)},
            "has_exec_jump": bool(not np.all(np.isnan(lambda_star))),
        }
    return analysis


def plot_all(all_results, g_values, lambda_values, delay_values, save_dir=OUTPUT_DIR):
    ng, nl, nk = len(g_values), len(lambda_values), len(delay_values)

    # ── exec_ratio heatmaps ──
    fig, axes = plt.subplots(1, nk, figsize=(5*nk, 4))
    if nk == 1: axes = [axes]
    fig.suptitle(f"Exec Ratio (control fraction) by delay k", fontsize=12)
    for ki, k_val in enumerate(delay_values):
        em = np.zeros((ng, nl))
        for gi, g in enumerate(g_values):
            for li, lam in enumerate(lambda_values):
                em[gi, li] = all_results[(g, lam, k_val)]["exec_ratio"]
        im = axes[ki].imshow(em.T, aspect="auto", origin="lower",
                              extent=[g_values[0], g_values[-1],
                                      lambda_values[0], lambda_values[-1]],
                              cmap="RdYlGn", vmin=0, vmax=1)
        axes[ki].set_title(f"k={k_val}")
        axes[ki].set_xlabel("g"); axes[ki].set_ylabel("λ")
        plt.colorbar(im, ax=axes[ki], fraction=0.046)
    plt.tight_layout()
    sp = os.path.join(save_dir, "gated_exec_ratio.png")
    plt.savefig(sp, dpi=150, bbox_inches="tight"); plt.close()

    # ── exec_ratio vs λ curves ──
    fig, axes = plt.subplots(1, nk, figsize=(5*nk, 4))
    if nk == 1: axes = [axes]
    for ki, k_val in enumerate(delay_values):
        for gi, g in enumerate(g_values):
            xs = lambda_values
            ys = [all_results[(g, lam, k_val)]["exec_ratio"] for lam in lambda_values]
            axes[ki].plot(xs, ys, 'o-', markersize=3, label=f"g={g}")
        axes[ki].set_xscale("log")
        axes[ki].set_xlabel("λ"); axes[ki].set_ylabel("exec_ratio")
        axes[ki].set_title(f"k={k_val}"); axes[ki].legend(fontsize=7)
        axes[ki].grid(True, alpha=0.3)
    plt.tight_layout()
    sp2 = os.path.join(save_dir, "gated_exec_vs_lambda.png")
    plt.savefig(sp2, dpi=150, bbox_inches="tight"); plt.close()

    # ── cost vs exec_ratio scatter ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for k_val in delay_values:
        xs = []; ys = []
        for g in g_values:
            for lam in lambda_values:
                r = all_results[(g, lam, k_val)]
                xs.append(r["exec_ratio"]); ys.append(r["mean_cost"])
        ax.scatter(xs, ys, alpha=0.6, label=f"k={k_val}", s=15)
    ax.set_xlabel("Exec Ratio (control fraction)"); ax.set_ylabel("mean_cost")
    ax.set_title("Cost vs Control Execution Ratio"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sp3 = os.path.join(save_dir, "gated_cost_vs_exec.png")
    plt.savefig(sp3, dpi=150, bbox_inches="tight"); plt.close()

    # ── alignment vs exec_ratio ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for k_val in delay_values:
        xs = []; ys = []
        for g in g_values:
            for lam in lambda_values:
                r = all_results[(g, lam, k_val)]
                xs.append(r["exec_ratio"]); ys.append(r["alignment_mean"])
        ax.scatter(xs, ys, alpha=0.6, label=f"k={k_val}", s=15)
    ax.set_xlabel("Exec Ratio (control fraction)"); ax.set_ylabel("Alignment")
    ax.set_title("Gradient Alignment vs Control Execution Ratio"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sp4 = os.path.join(save_dir, "gated_align_vs_exec.png")
    plt.savefig(sp4, dpi=150, bbox_inches="tight"); plt.close()

    print(f"  Plots saved to {save_dir}/")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seeds = list(range(SEEDS))
    all_results = run_3d_sweep(seeds=seeds)
    analysis = analyze_all(all_results, G_VALUES, LAMBDA_VALUES, DELAY_VALUES)

    print(f"\n{'='*60}")
    print("STRATEGY SWITCHING ANALYSIS")
    print(f"{'='*60}")
    for k in DELAY_VALUES:
        a = analysis[k]
        flips = [f"g={g}:{ls:.3f}" for g, ls in a["lambda_star_exec"].items() if ls is not None]
        print(f"  k={k}: exec_range=[{a['exec_ratio_min']:.3f}, {a['exec_ratio_max']:.3f}]  "
              f"has_jump={a['has_exec_jump']}")
        if flips: print(f"    exec jump at: {', '.join(flips)}")

    # Correlation: alignment vs exec_ratio
    all_exec = []; all_align = []
    for k in DELAY_VALUES:
        for g in G_VALUES:
            for lam in LAMBDA_VALUES:
                r = all_results[(g, lam, k)]
                all_exec.append(r["exec_ratio"])
                all_align.append(r["alignment_mean"])
    corr = np.corrcoef(all_exec, all_align)[0, 1]
    print(f"\n  corr(alignment, exec_ratio) = {corr:+.4f}")

    any_jump = any(analysis[k]["has_exec_jump"] for k in DELAY_VALUES)
    exec_range = max(analysis[k]["exec_ratio_range"] for k in DELAY_VALUES)

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    if any_jump:
        print(f"  STRATEGY SWITCHING EXISTS")
        print(f"  exec_ratio has significant jumps along lambda axis.")
        print(f"  Time/capacity competition drives a phase transition.")
        verdict = "strategy_switching_found"
    elif exec_range > 0.3:
        print(f"  WEAK SWITCHING (exec_range={exec_range:.3f})")
        print(f"  Gate adapts to lambda but no sharp jump.")
        verdict = "weak_switching"
    else:
        print(f"  NO SWITCHING (exec_range={exec_range:.3f})")
        print(f"  Gate stays in narrow range — no strategy selection.")
        verdict = "no_switching"

    plot_all(all_results, G_VALUES, LAMBDA_VALUES, DELAY_VALUES)

    output = {
        "experiment": "gated_objective_selection",
        "parameters": {"alpha": ALPHA, "noise": NOISE, "target": TARGET,
                       "drift_omega": DRIFT_OMEGA, "pred_delta": PRED_DELTA,
                       "switch_beta": SWITCH_BETA, "episodes": EPISODES,
                       "rollout_N": ROLLOUT_N, "seeds": SEEDS},
        "g_values": G_VALUES, "lambda_values": LAMBDA_VALUES,
        "delay_values": DELAY_VALUES,
        "analysis": analysis,
        "corr_align_exec": float(corr),
        "verdict": verdict,
        "results": {f"{r['g']},{r['lambda']},{r['delay_k']}":
                    {kr: r[kr] for kr in ["exec_ratio", "gate_std", "alignment_mean",
                                           "alignment_std", "neg_fraction",
                                           "mean_cost", "in_zone_rate", "policy_slope"]}
                    for r in all_results.values()},
    }
    with open(os.path.join(OUTPUT_DIR, "gated_objective_selection.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {os.path.join(OUTPUT_DIR, 'gated_objective_selection.json')}")


if __name__ == "__main__":
    main()
