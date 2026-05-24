"""
Online Stability Controller with state machine, continuous control,
damping, and region memory.

State machine:  SAFE ←→ WARNING ←→ DANGER
  - SAFE:    eta → eta + α * (eta_target - eta)   [approach target]
  - WARNING: eta → eta * 0.9                        [gentle reduction]
  - DANGER:  eta → eta * 0.5                        [aggressive cut]

Damping:     eta = β * eta_prev + (1-β) * eta_new
Region lock: if 3 consecutive DANGER entries → lock_region(drift)

Usage: plug into train/eval loop, call ctrl.step(drift) each timestep.
"""
import json
import numpy as np
from collections import deque


class State:
    SAFE = "SAFE"
    WARNING = "WARNING"
    DANGER = "DANGER"


class OnlineMetricsTracker:
    """Records per-step and aggregate online metrics."""

    def __init__(self):
        self.eta_history = []
        self.eta_target_history = []
        self.state_history = []
        self.drift_history = []
        self.danger_count = 0
        self.warning_streak = 0
        self.max_warning_streak = 0
        self.enter_danger_times = []
        self.exit_danger_times = []
        self.locked_regions = set()
        self.loss_history = []
        self.step = 0

    def record(self, eta, eta_target, state, drift, loss=None):
        self.eta_history.append(float(eta))
        self.eta_target_history.append(float(eta_target))
        self.state_history.append(state)
        self.drift_history.append(float(drift))
        if loss is not None:
            self.loss_history.append(float(loss))
        self.step += 1

        if state == State.DANGER:
            self.danger_count += 1
            self.warning_streak = 0
        elif state == State.WARNING:
            self.warning_streak += 1
            self.max_warning_streak = max(self.max_warning_streak, self.warning_streak)
        else:
            self.warning_streak = 0

    def eta_usage_ratio(self):
        if not self.eta_history:
            return 0
        return float(np.mean([e / max(t, 1e-8)
                for e, t in zip(self.eta_history, self.eta_target_history)]))

    def crash_count(self):
        return self.danger_count

    def avg_warning_duration(self):
        """Average consecutive WARNING streak length."""
        streaks = []
        current = 0
        for s in self.state_history:
            if s == State.WARNING:
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0
        if current > 0:
            streaks.append(current)
        return float(np.mean(streaks)) if streaks else 0.0

    def summary(self):
        return {
            "steps": self.step,
            "eta_usage_ratio": self.eta_usage_ratio(),
            "danger_count": self.danger_count,
            "max_warning_streak": self.max_warning_streak,
            "avg_warning_duration": self.avg_warning_duration(),
            "locked_regions": list(self.locked_regions),
            "final_eta": self.eta_history[-1] if self.eta_history else 0,
            "state_distribution": {
                "SAFE": self.state_history.count(State.SAFE),
                "WARNING": self.state_history.count(State.WARNING),
                "DANGER": self.state_history.count(State.DANGER),
            },
            "loss_mean": float(np.mean(self.loss_history)) if self.loss_history else 0,
            "loss_volatility": float(np.std(self.loss_history)) if self.loss_history else 0,
        }


class OnlineStabilityController:
    """
    Real-time stability controller with:
      - State machine (SAFE/WARNING/DANGER)
      - Continuous eta adjustment
      - Inertial damping
      - Region memory with lock-after-3 rule
      - Online metrics tracking
    """

    def __init__(self, alpha=0.1, beta=0.8, danger_lock_count=3):
        from strategy import StabilityController as BaseController
        self.base = BaseController()
        self.alpha = alpha
        self.beta = beta
        self.danger_lock_count = danger_lock_count

        self.state = State.SAFE
        self.eta = 0.5  # current eta value
        self.locked_drifts = set()

        # Transition hysteresis
        self._danger_streak = 0
        self._safe_streak = 0
        self._danger_entry_window = deque(maxlen=danger_lock_count)
        self._prev_state = None

        self.metrics = OnlineMetricsTracker()

    def _eta_target(self, drift):
        return self.base.eta_max(drift)

    def _detect_state(self, drift):
        """Determine state from drift position and S-derivative."""
        in_unstable = self.base.is_unstable(drift)
        warning = self.base.should_warn(drift)

        if drift in self.locked_drifts:
            in_unstable = True

        if in_unstable:
            return State.DANGER
        elif warning:
            return State.WARNING
        else:
            return State.SAFE

    def _apply_damping(self, eta_new):
        """Inertial damping: smooth transitions."""
        return self.beta * self.eta + (1.0 - self.beta) * eta_new

    def _check_region_lock(self, drift):
        """Lock region if DANGER triggered 3 consecutive times at same drift band."""
        self._danger_entry_window.append(float(drift))
        if len(self._danger_entry_window) >= self.danger_lock_count:
            vals = list(self._danger_entry_window)
            spread = max(vals) - min(vals)
            if spread < 0.02:
                center = np.mean(vals)
                for start, end in self.base.unstable_regions:
                    if start <= center <= end:
                        self.locked_drifts.add((start, end))
                        self.metrics.locked_regions.add(f"{start:.4f}-{end:.4f}")
                        break

    def step(self, drift, loss=None):
        """
        Called each timestep with current drift.
        Returns: new_eta, state, metrics_summary
        """
        # 1. State detection
        new_state = self._detect_state(drift)

        # 2. State transitions with hysteresis
        if new_state == State.DANGER:
            self._danger_streak += 1
            self._safe_streak = 0
        elif new_state == State.SAFE:
            self._safe_streak += 1
            self._danger_streak = 0
        else:
            self._danger_streak = max(0, self._danger_streak - 1)

        self.state = new_state

        # 3. Region memory: lock after 3 consecutive DANGER
        if self.state == State.DANGER:
            self._check_region_lock(drift)

        # 4. Compute eta_new based on state
        eta_target = self._eta_target(drift)

        if self.state == State.SAFE:
            eta_new = self.eta + self.alpha * (eta_target - self.eta)
        elif self.state == State.WARNING:
            eta_new = self.eta * 0.9
        elif self.state == State.DANGER:
            eta_new = self.eta * 0.5

        # 5. Apply damping
        self.eta = self._apply_damping(eta_new)
        self._prev_state = self.state

        # 6. Record metrics
        self.metrics.record(self.eta, eta_target, self.state, drift, loss)

        return self.eta, self.state

    def summary(self):
        return self.metrics.summary()


# ── A/B Test ─────────────────────────────────────────────────────────
def run_ab_test(env_class, kappa=1.0, drift=0.05, steps=500):
    """
    Compare baseline (fixed eta) vs adaptive controller on the same environment.
    Measures: convergence speed, crash count, loss volatility.
    """
    import torch
    import random
    import torch.nn.functional as F
    import torch.optim as optim
    import copy
    from experiment10 import reset_seed, _make_agent, _make_ctrl

    results = {}

    for mode, use_adaptive in [("baseline", False), ("strategy", True)]:
        reset_seed(42)

        env = env_class(kappa=kappa, drift_rate=drift,
                        flip_mode="deterministic", add_context=False)
        agent = _make_agent(env)
        ctrl = _make_ctrl(agent, 4, inertia=0.0)

        if use_adaptive:
            online = OnlineStabilityController()
        else:
            online = None

        loss_history = []
        eta_history = []
        state_counts = {State.SAFE: 0, State.WARNING: 0, State.DANGER: 0}

        obs = env.reset()
        ctrl.gating_reset()

        for t in range(steps):
            # Use adaptive eta if available
            if use_adaptive:
                d = float(getattr(env, 'drift', drift))
                new_eta, state = online.step(d)
                # Apply eta to controller — adjust inertia
                if hasattr(ctrl.gating, 'inertia'):
                    ctrl.gating.inertia = new_eta
                ctrl.inertia = new_eta
                ctrl.gating.inertia = new_eta
                eta_history.append(new_eta)
                state_counts[state] += 1
            else:
                eta_history.append(0.0)

            a = agent.act(obs)
            o_next, _, done = env.step(a)
            target = torch.tensor(o_next, dtype=torch.float32)
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            if ctrl.n_models() > 0:
                z_soft, z_logits, _ = ctrl.gating(s)
                K_cur = z_soft.size(-1)
                z_hard_idx = z_logits.argmax(dim=-1)
                z_hard = F.one_hot(z_hard_idx, K_cur).float()
                weights = (z_hard - z_soft.detach() + z_soft).squeeze(0)
                preds = [m.predict(obs, a) for m in ctrl.models]
                soft_pred = sum(weights[i] * preds[i] for i in range(K_cur))
                loss = ((soft_pred - target) ** 2).mean()

                error_val = float(np.mean(np.abs(soft_pred.detach().numpy() - o_next)))
                ctrl.should_update(error_val, 0)
                ctrl.record_usage(int(weights.argmax().item()))
                for i in range(K_cur):
                    e_i = float(np.mean(np.abs(preds[i].detach().numpy() - o_next)))
                    ctrl.track_error(i, e_i)

                ctrl.gating_optimizer.zero_grad()
                for m in ctrl.models:
                    m.optimizer.zero_grad()
                loss.backward()
                ctrl.gating_optimizer.step()
                for m in ctrl.models:
                    m.optimizer.step()

                loss_history.append(float(loss.item()))
            else:
                loss_history.append(0.0)

            ctrl.maybe_merge()
            ctrl.maybe_prune()

            if done:
                obs = env.reset()
                ctrl.gating_reset()
            else:
                obs = o_next

        # Compute convergence metrics
        window = max(10, steps // 10)
        early_loss = float(np.mean(loss_history[:window]))
        late_loss = float(np.mean(loss_history[-window:]))
        convergence = early_loss / max(late_loss, 1e-8)
        crash_count = sum(1 for l in loss_history if l > 2.0 * np.std(loss_history))

        if use_adaptive:
            summary = online.summary()
        else:
            summary = {"danger_count": 0, "state_distribution": {}}

        results[mode] = {
            "early_loss": early_loss,
            "late_loss": late_loss,
            "convergence_ratio": convergence,
            "crash_count": crash_count,
            "loss_volatility": float(np.std(loss_history)),
            "loss_mean": float(np.mean(loss_history)),
            "state_distribution": state_counts,
            "online_summary": summary,
            "loss_history": [float(l) for l in loss_history],
            "eta_history": [float(e) for e in eta_history],
        }

        print(f"\n{mode:>10}: early_loss={early_loss:.4f} late_loss={late_loss:.4f} "
              f"conv={convergence:.2f}x crashes={crash_count} "
              f"volatility={np.std(loss_history):.4f}")
        if use_adaptive:
            print(f"           states: SAFE={state_counts[State.SAFE]} "
                  f"WARNING={state_counts[State.WARNING]} "
                  f"DANGER={state_counts[State.DANGER]} "
                  f"eta_usage={summary['eta_usage_ratio']:.3f}")

    return results


# ── Main: A/B test ───────────────────────────────────────────────────
if __name__ == "__main__":
    from env_drifting_double_well import DriftingDoubleWell

    print("=" * 60)
    print("A/B TEST: baseline (fixed eta) vs strategy (adaptive)")
    print("=" * 60)

    all_results = {}
    for drift_test in [0.02, 0.05, 0.10]:
        print(f"\n── drift = {drift_test} ──")
        results = run_ab_test(DriftingDoubleWell, kappa=1.0, drift=drift_test, steps=500)
        all_results[drift_test] = results

        base = results["baseline"]
        strat = results["strategy"]

        print(f"\n  COMPARISON at drift={drift_test}:")
        print(f"  convergence:  baseline={base['convergence_ratio']:.2f}x  "
              f"strategy={strat['convergence_ratio']:.2f}x  "
              f"delta={strat['convergence_ratio'] - base['convergence_ratio']:+.2f}")
        print(f"  volatility:   baseline={base['loss_volatility']:.4f}  "
              f"strategy={strat['loss_volatility']:.4f}  "
              f"delta={strat['loss_volatility'] - base['loss_volatility']:+.4f}")
        print(f"  crash_count:  baseline={base['crash_count']}  "
              f"strategy={strat['crash_count']}")

    with open("ab_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nA/B test results saved to ab_test_results.json")
