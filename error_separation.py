"""
Error Separation Diagnostic Module.

Pure observer — no gating modification, no eta control, no loss injection.

Steps:
  1. collect per-expert error distributions
  2. compute pairwise MSE distance between distributions → D_sep
  3. EMA-smooth D_sep
  4. per-step log: {drift, D_sep, D_sep_smooth, loss, eta, adv}
  5. generate three plots (D_sep vs drift, loss vs drift, eta vs drift)
  6. detect alignment intervals (D_sep ↑ ∧ loss ↓)
  7. mark candidate W (drift ranges where D_sep > epsilon)
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ErrorSeparationTracker:
    """
    Observational tracker for expert error-separation dynamics.

    Constraints enforced by design:
      - Never touches gating / gating_optimizer
      - Never writes to controller.inertia or any eta
      - Never adds terms to the loss
      - D_sep is never used for branching / decisions
    """

    def __init__(self, window=200, alpha_ema=0.1, epsilon=0.01):
        self.window = window
        self.alpha_ema = alpha_ema
        self.epsilon = epsilon

        self._errors = defaultdict(list)
        self.log = []
        self._D_sep_smooth = None

    def _push_error(self, idx, val):
        buf = self._errors[idx]
        buf.append(float(val))
        if len(buf) > self.window:
            buf.pop(0)

    def _purge_inactive(self, active_ids):
        stale = [i for i in self._errors if i not in active_ids]
        for i in stale:
            del self._errors[i]

    def compute_D_sep(self):
        """
        D_sep = mean over all (i≠j) of ((e_i[-W:] − e_j[-W:]) ** 2).mean()
        """
        active = [i for i in sorted(self._errors) if len(self._errors[i]) >= 2]
        if len(active) < 2:
            return 0.0
        dists = []
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                ea = np.array(self._errors[active[a]])
                eb = np.array(self._errors[active[b]])
                L = min(len(ea), len(eb))
                dists.append(float(((ea[-L:] - eb[-L:]) ** 2).mean()))
        return float(np.mean(dists)) if dists else 0.0

    def step(self, drift, loss, eta, advantage, expert_errors):
        """
        Called once per training / evaluation step.

        Parameters
        ----------
        drift : float
        loss : float          (MSE)
        eta : float           (inertia / learning rate at this step)
        advantage : float     (oracle_err − loss, or similar per-step signal)
        expert_errors : dict  {expert_idx → error_value} for ALL active experts
        """
        for idx, val in expert_errors.items():
            self._push_error(idx, val)
        self._purge_inactive(set(expert_errors.keys()))

        D_sep = self.compute_D_sep()
        if self._D_sep_smooth is None:
            self._D_sep_smooth = D_sep
        else:
            self._D_sep_smooth = (
                self.alpha_ema * D_sep + (1.0 - self.alpha_ema) * self._D_sep_smooth
            )

        self.log.append({
            "drift": float(drift),
            "D_sep": D_sep,
            "D_sep_smooth": self._D_sep_smooth,
            "loss": float(loss),
            "eta": float(eta),
            "adv": float(advantage),
        })

    # ── post-hoc analysis ──────────────────────────────────────────

    def find_alignment_intervals(self, min_len=10, smooth_win=5):
        """
        Intervals where D_sep_smooth ↑ AND loss ↓ simultaneously.
        Returns list of {start, end, drift_start, drift_end, mean_D_sep, mean_loss}.
        """
        if len(self.log) < min_len + smooth_win:
            return []

        D = np.array([e["D_sep_smooth"] for e in self.log])
        L = np.array([e["loss"] for e in self.log])
        drift_vals = np.array([e["drift"] for e in self.log])

        dD = np.diff(D)
        dL = np.diff(L)

        kernel = np.ones(smooth_win) / smooth_win
        D_up = np.convolve(dD > 0, kernel, mode="same") > 0.5
        L_down = np.convolve(dL < 0, kernel, mode="same") > 0.5
        aligned = D_up & L_down

        intervals = []
        inside = False
        start = 0
        for t in range(len(aligned)):
            if aligned[t] and not inside:
                start = t
                inside = True
            elif not aligned[t] and inside:
                if t - start >= min_len:
                    intervals.append({
                        "start": int(start),
                        "end": int(t),
                        "drift_start": float(drift_vals[start]),
                        "drift_end": float(drift_vals[t]),
                        "mean_D_sep": float(np.mean(D[start:t + 1])),
                        "mean_loss": float(np.mean(L[start:t + 1])),
                    })
                inside = False
        if inside and len(D) - start >= min_len:
            intervals.append({
                "start": int(start),
                "end": int(len(D) - 1),
                "drift_start": float(drift_vals[start]),
                "drift_end": float(drift_vals[-1]),
                "mean_D_sep": float(np.mean(D[start:])),
                "mean_loss": float(np.mean(L[start:])),
            })
        return intervals

    def find_W_candidates(self):
        """
        Candidate drift ranges where D_sep_smooth > epsilon.
        W_candidates = drift_ranges_where(D_sep > epsilon)
        """
        if len(self.log) < 2:
            return []
        D = np.array([e["D_sep_smooth"] for e in self.log])
        drift_vals = np.array([e["drift"] for e in self.log])

        candidates = []
        high = D > self.epsilon
        inside = False
        start = 0
        for t in range(len(high)):
            if high[t] and not inside:
                start = t
                inside = True
            elif not high[t] and inside:
                candidates.append({
                    "drift_start": float(drift_vals[start]),
                    "drift_end": float(drift_vals[t - 1]),
                    "duration": int(t - start),
                    "mean_D_sep": float(np.mean(D[start:t])),
                })
                inside = False
        if inside:
            candidates.append({
                "drift_start": float(drift_vals[start]),
                "drift_end": float(drift_vals[-1]),
                "duration": int(len(D) - start),
                "mean_D_sep": float(np.mean(D[start:])),
            })
        return candidates

    def plot(self, save_prefix="error_sep"):
        if len(self.log) < 2:
            print("[error_separation] not enough data for plots")
            return

        drift = np.array([e["drift"] for e in self.log])
        D_sep = np.array([e["D_sep_smooth"] for e in self.log])
        loss = np.array([e["loss"] for e in self.log])
        eta = np.array([e["eta"] for e in self.log])

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

        ax = axes[0]
        ax.scatter(drift, D_sep, s=2, alpha=0.5, color="#1f77b4")
        ax.axhline(self.epsilon, color="red", linestyle="--", alpha=0.35,
                   label=f"ε = {self.epsilon}")
        ax.set_xlabel("drift")
        ax.set_ylabel("D_sep (smooth)")
        ax.set_title("D_sep vs drift")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.scatter(drift, loss, s=2, alpha=0.5, color="#d62728")
        ax.set_xlabel("drift")
        ax.set_ylabel("loss")
        ax.set_title("loss vs drift")

        ax = axes[2]
        ax.scatter(drift, eta, s=2, alpha=0.5, color="#2ca02c")
        ax.set_xlabel("drift")
        ax.set_ylabel("eta")
        ax.set_title("eta vs drift")

        plt.tight_layout()
        path = f"{save_prefix}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[error_separation] plots saved → {path}")

    def summary(self):
        return {
            "n_steps": len(self.log),
            "D_sep_mean": float(np.mean([e["D_sep"] for e in self.log])) if self.log else 0.0,
            "D_sep_smooth_final": self._D_sep_smooth,
            "loss_mean": float(np.mean([e["loss"] for e in self.log])) if self.log else 0.0,
            "eta_mean": float(np.mean([e["eta"] for e in self.log])) if self.log else 0.0,
            "drift_range": [
                float(self.log[0]["drift"]),
                float(self.log[-1]["drift"]),
            ] if self.log else None,
            "alignment_intervals": self.find_alignment_intervals(),
            "W_candidates": self.find_W_candidates(),
        }

    def save_log(self, path):
        with open(path, "w") as f:
            json.dump(self.log, f, indent=1)
        print(f"[error_separation] per-step log saved → {path}")

    def save_summary(self, path):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        print(f"[error_separation] summary saved → {path}")


# ═══════════════════════════════════════════════════════════════════════
# Instrumented training / evaluation loops
# ═══════════════════════════════════════════════════════════════════════

def train_with_tracker(env, ctrl, agent, steps, tracker, z_loss=False, inertia=0.0,
                       online_ctrl=None):
    """
    Training loop instrumented with ErrorSeparationTracker.
    Mirrors train_multi_expert + run_soft in structure.
    """
    obs = env.reset()
    ctrl.gating_reset()
    prev_z = None
    running_error = 1.0

    for t in range(steps):
        ctrl.maybe_split()

        a = agent.act(obs)
        o_next, _, done = env.step(a)
        target = torch.tensor(o_next, dtype=torch.float32)

        drift = float(getattr(env, "drift", 0.0))

        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        z_soft, z_logits, _ = ctrl.gating(s)
        ctrl._last_logits = z_logits
        ctrl._last_z_soft = z_soft

        K_cur = z_soft.size(-1)
        z_hard_idx = z_logits.argmax(dim=-1)
        z_hard = F.one_hot(z_hard_idx, K_cur).float()
        weights = (z_hard - z_soft.detach() + z_soft).squeeze(0)

        preds = [m.predict(obs, a) for m in ctrl.models]
        soft_pred = sum(weights[i] * preds[i] for i in range(K_cur))
        loss_pred = ((soft_pred - target) ** 2).mean()

        entropy = -(weights * torch.log(weights + 1e-8)).sum()
        loss = loss_pred - 0.005 * entropy

        # ── per-expert errors ──
        expert_errors = {}
        with torch.no_grad():
            perr = torch.stack([
                ((preds[i].detach() - target) ** 2).mean()
                for i in range(K_cur)
            ])
            oracle_err = perr.min().item()

        for i in range(K_cur):
            e_i = float(np.mean(np.abs(preds[i].detach().numpy() - o_next)))
            expert_errors[i] = e_i

        # ── optional z-loss (allows replicating stage-10 dynamics) ──
        if z_loss and ctrl.use_z:
            with torch.no_grad():
                perr_t = torch.stack([
                    ((preds[i].detach() - target) ** 2).mean()
                    for i in range(K_cur)
                ])
                perr_t = perr_t - perr_t.min()
                z_target_raw = torch.softmax(-perr_t / 0.05, dim=-1)

            z_kld = F.kl_div(
                F.log_softmax(z_logits, dim=-1),
                z_target_raw.detach(),
                reduction="sum",
            )
            temporal_loss_z = torch.tensor(0.0)
            if prev_z is not None and len(weights) == len(prev_z):
                temporal_loss_z = ((weights - prev_z) ** 2).sum()

            error_soft = float(np.mean(np.abs(soft_pred.detach().numpy() - o_next)))
            running_error = 0.99 * running_error + 0.01 * error_soft
            activate_z = running_error < 0.05

            if activate_z:
                if not ctrl.freeze_structure:
                    loss = loss + 0.5 * z_kld
                else:
                    balance = -(weights * torch.log(weights + 1e-8)).sum()
                    loss = loss + 0.01 * z_kld + 0.005 * balance
                if temporal_loss_z.item() > 0:
                    loss = loss + 0.02 * temporal_loss_z

        # ── tracker step (observational only) ──
        advantage = float(oracle_err - loss_pred.item())
        eta = float(getattr(ctrl, "inertia", inertia))
        tracker.step(
            drift=drift,
            loss=float(loss_pred.item()),
            eta=eta,
            advantage=advantage,
            expert_errors=expert_errors,
        )

        # ── online controller step ──
        if online_ctrl is not None:
            new_eta = online_ctrl.forward(drift, loss=float(loss_pred.item()))
            if hasattr(ctrl.gating, "inertia"):
                ctrl.gating.inertia = float(new_eta)
            ctrl.inertia = float(new_eta)
            eta = float(new_eta)

        # ── backprop ──
        error = float(np.mean(np.abs(soft_pred.detach().numpy() - o_next)))
        ctrl.should_update(error, 0)
        ctrl.record_usage(int(weights.argmax().item()))
        for i in range(K_cur):
            ctrl.track_error(i, expert_errors[i])

        ctrl.gating_optimizer.zero_grad()
        for m in ctrl.models:
            m.optimizer.zero_grad()
        loss.backward()
        ctrl.gating_optimizer.step()
        for m in ctrl.models:
            m.optimizer.step()

        if ctrl.use_z:
            prev_z = weights.detach().clone()

        ctrl.maybe_merge()
        ctrl.maybe_prune()

        if online_ctrl is not None:
            online_ctrl.update(drift, float(loss_pred.item()))

        if done:
            obs = env.reset()
            ctrl.gating_reset()
            prev_z = None
        else:
            obs = o_next

    return tracker


def evaluate_with_tracker(env, ctrl, steps, tracker, agent=None):
    """
    Evaluation loop instrumented with ErrorSeparationTracker.
    """
    obs = env.reset()
    ctrl.gating_reset()

    for t in range(steps):
        a = agent.act(obs) if agent is not None else __import__("random").randint(0, 1)
        o_next, _, done = env.step(a)
        target = torch.tensor(o_next, dtype=torch.float32)

        drift = float(getattr(env, "drift", 0.0))

        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            z_soft, z_logits, _ = ctrl.gating(s)
            K_cur = z_soft.size(-1)
            z_hard_idx = z_logits.argmax(dim=-1)
            z_hard = F.one_hot(z_hard_idx, K_cur).float()
            weights = (z_hard - z_soft.detach() + z_soft).squeeze(0)

            preds = [m.predict(obs, a) for m in ctrl.models]
            soft_pred = sum(weights[i] * preds[i] for i in range(K_cur))
            loss_val = float(((soft_pred - target) ** 2).mean().item())

            perr = torch.stack([
                ((preds[i] - target) ** 2).mean() for i in range(K_cur)
            ])
            oracle_err = perr.min().item()

        expert_errors = {}
        for i in range(K_cur):
            e_i = float(np.mean(np.abs(preds[i].detach().numpy() - o_next)))
            expert_errors[i] = e_i

        eta = float(getattr(ctrl, "inertia", 0.0))
        advantage = float(oracle_err - loss_val)
        tracker.step(
            drift=drift,
            loss=loss_val,
            eta=eta,
            advantage=advantage,
            expert_errors=expert_errors,
        )

        if done:
            obs = env.reset()
            ctrl.gating_reset()
        else:
            obs = o_next

    return tracker


# ═══════════════════════════════════════════════════════════════════════
# Scan-level aggregation & analysis
# ═══════════════════════════════════════════════════════════════════════

def aggregate_scan_results(run_summaries):
    """
    Aggregate per-run summaries grouped by drift_rate.

    Parameters
    ----------
    run_summaries : dict
        {drift_rate: [summary_dict, summary_dict, ...]}
        Each summary_dict is the output of tracker.summary().

    Returns
    -------
    drift_vals : np.array       sorted drift rates
    D_sep_mean, loss_mean, eta_mean : np.array
    D_sep_std,  loss_std,  eta_std  : np.array
    """
    drift_vals = sorted(run_summaries.keys())
    D_sep_arr, loss_arr, eta_arr = [], [], []

    for d in drift_vals:
        summaries = run_summaries[d]
        D_sep_arr.append([s["D_sep_mean"] for s in summaries])
        loss_arr.append([s["loss_mean"] for s in summaries])
        eta_arr.append([s["eta_mean"] for s in summaries])

    D_sep_arr  = np.array(D_sep_arr)
    loss_arr   = np.array(loss_arr)
    eta_arr    = np.array(eta_arr)

    return (
        np.array(drift_vals, dtype=float),
        D_sep_arr.mean(axis=1),  loss_arr.mean(axis=1),  eta_arr.mean(axis=1),
        D_sep_arr.std(axis=1),   loss_arr.std(axis=1),   eta_arr.std(axis=1),
    )


def find_W_struct(drift_vals, D_sep, loss):
    """
    Detect structural intervals W_struct where:

        d/d(drift) D_sep > 0   AND   d/d(drift) loss < 0

    Uses central-difference derivatives on the mean curves.
    Returns list of {drift_start, drift_end, n_points}.
    """
    if len(drift_vals) < 3:
        return []

    dD      = np.gradient(D_sep,  drift_vals)
    dL      = np.gradient(loss,   drift_vals)
    aligned = (dD > 0) & (dL < 0)

    intervals = []
    inside = False
    start_i = 0
    for i in range(len(aligned)):
        if aligned[i] and not inside:
            start_i = i
            inside = True
        elif not aligned[i] and inside:
            if i - start_i >= 1:
                intervals.append({
                    "drift_start": float(drift_vals[start_i]),
                    "drift_end":   float(drift_vals[i - 1]),
                    "n_points":    int(i - start_i),
                })
            inside = False
    if inside and len(aligned) - start_i >= 1:
        intervals.append({
            "drift_start": float(drift_vals[start_i]),
            "drift_end":   float(drift_vals[-1]),
            "n_points":    int(len(aligned) - start_i),
        })
    return intervals


def check_monotonic(drift_vals, D_sep):
    """
    Test whether D_sep(drift) is monotonic.

    Returns dict with:
      - is_monotonic           : bool   (strictly inc or dec overall)
      - is_piecewise_monotonic : bool   (can be split into monotonic segments)
      - segments               : list of {drift_start, drift_end, direction}
      - direction              : "increasing" | "decreasing" | "mixed"
    """
    dD = np.gradient(D_sep, drift_vals)

    if np.all(dD >= 0):
        return {
            "is_monotonic": True,
            "direction": "increasing",
            "is_piecewise_monotonic": True,
            "segments": [{"drift_start": float(drift_vals[0]),
                          "drift_end": float(drift_vals[-1]),
                          "direction": "increasing"}],
        }
    if np.all(dD <= 0):
        return {
            "is_monotonic": True,
            "direction": "decreasing",
            "is_piecewise_monotonic": True,
            "segments": [{"drift_start": float(drift_vals[0]),
                          "drift_end": float(drift_vals[-1]),
                          "direction": "decreasing"}],
        }

    signs = dD > 0
    segments = []
    start_i = 0
    cur_sign = bool(signs[0])
    for i in range(1, len(signs)):
        if signs[i] != cur_sign:
            segments.append({
                "drift_start": float(drift_vals[start_i]),
                "drift_end":   float(drift_vals[i - 1]),
                "direction":   "increasing" if cur_sign else "decreasing",
            })
            start_i = i
            cur_sign = bool(signs[i])
    segments.append({
        "drift_start": float(drift_vals[start_i]),
        "drift_end":   float(drift_vals[-1]),
        "direction":   "increasing" if cur_sign else "decreasing",
    })

    return {
        "is_monotonic": False,
        "direction": "mixed",
        "is_piecewise_monotonic": True,
        "segments": segments,
    }


# ── scan-level plotting ──────────────────────────────────────────────

def plot_scan_curves(drift_vals, D_sep, D_sep_std,
                     loss, loss_std, eta, eta_std,
                     save_path="error_sep_scan_curves.png"):
    """Three final curves: D_sep_mean(d), loss_mean(d), eta_mean(d)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    ax = axes[0]
    ax.errorbar(drift_vals, D_sep, yerr=D_sep_std, marker="o",
                color="#1f77b4", capsize=4, markersize=6)
    ax.set_xlabel("drift rate")
    ax.set_ylabel("D_sep mean")
    ax.set_title("D_sep_mean(d)")

    ax = axes[1]
    ax.errorbar(drift_vals, loss, yerr=loss_std, marker="o",
                color="#d62728", capsize=4, markersize=6)
    ax.set_xlabel("drift rate")
    ax.set_ylabel("loss mean")
    ax.set_title("loss_mean(d)")

    ax = axes[2]
    ax.errorbar(drift_vals, eta, yerr=eta_std, marker="o",
                color="#2ca02c", capsize=4, markersize=6)
    ax.set_xlabel("drift rate")
    ax.set_ylabel("eta mean")
    ax.set_title("eta_mean(d)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[scan] curves plot saved → {save_path}")


def plot_alignment(drift_vals, D_sep, loss,
                   W_struct=None, save_path="error_sep_alignment.png"):
    """
    Key alignment figure: normalized D_sep and -loss overlayed.
    D_sep_norm ∈ [0,1], (-loss)_norm ∈ [0,1].
    W_struct intervals shaded in green.
    """
    def norm(x):
        mn, mx = x.min(), x.max()
        rng = mx - mn
        return (x - mn) / rng if rng > 1e-12 else np.zeros_like(x)

    D_norm = norm(np.array(D_sep))
    L_norm = norm(-np.array(loss))

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(drift_vals, D_norm, "o-", color="#1f77b4", label="D_sep (norm)", markersize=6)
    ax.plot(drift_vals, L_norm, "s--", color="#d62728", label="-loss (norm)", markersize=6)

    if W_struct:
        for w in W_struct:
            ax.axvspan(w["drift_start"], w["drift_end"],
                       color="green", alpha=0.15)
        ax.plot([], [], color="green", alpha=0.3, linewidth=6,
                label="W_struct")

    ax.set_xlabel("drift rate")
    ax.set_ylabel("normalised")
    ax.set_title("D_sep_norm  &  -loss_norm  vs drift rate")
    ax.legend()
    ax.set_ylim(-0.05, 1.08)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[scan] alignment plot saved → {save_path}")
