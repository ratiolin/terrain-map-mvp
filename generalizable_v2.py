"""
Final generalizable controller — v2 with:
  1. Active probing: η_probe = η + ε  (explore upper boundary)
  2. Near-instability learning: only update boundary model near edge
  3. Stability margin head: margin = h(d), η = η_mean · sigmoid(margin)
  4. Danger-aware teacher (non-monotonic, boundary-coupled)
  5. Importance sampling: focus on d ∈ [0.07, 0.15]
"""
import json
import numpy as np
from collections import deque


class ProbeResult:
    def __init__(self):
        self.attempts = 0
        self.survived = 0
        self.diverged = 0
        self.eta_at_divergence = []

    def record(self, eta, survived):
        self.attempts += 1
        if survived:
            self.survived += 1
        else:
            self.diverged += 1
            self.eta_at_divergence.append(float(eta))

    def survival_rate(self):
        return self.survived / max(self.attempts, 1)

    def eta_upper_bound(self, percentile=90):
        if not self.eta_at_divergence:
            return 1.0
        return float(np.percentile(self.eta_at_divergence, percentile))


class GeneralizableMLPv2:
    """MLP with 3 heads: mean, logvar, margin."""

    def __init__(self, input_dim, hidden=32):
        k = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(hidden, input_dim) * k
        self.b1 = np.zeros(hidden)

        self.W2_mean = np.random.randn(1, hidden) * np.sqrt(2.0 / hidden)
        self.b2_mean = np.array([0.3])
        self.W2_logvar = np.random.randn(1, hidden) * 0.01
        self.b2_logvar = np.array([-2.0])
        self.W2_margin = np.random.randn(1, hidden) * np.sqrt(2.0 / hidden)
        self.b2_margin = np.array([0.0])
        self.W2_advantage = np.random.randn(1, hidden) * 0.01
        self.b2_advantage = np.array([0.0])

    def _hidden(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        h_pre = self.W1 @ x_arr + self.b1
        return np.tanh(h_pre), (1 - np.tanh(h_pre)**2), x_arr

    def forward(self, x):
        h, _, _ = self._hidden(x)
        mean = float((self.W2_mean @ h + self.b2_mean)[0])
        logvar = float((self.W2_logvar @ h + self.b2_logvar)[0])
        std = float(np.exp(0.5 * np.clip(logvar, -5, 2)))
        margin = float((self.W2_margin @ h + self.b2_margin)[0])
        advantage = float((self.W2_advantage @ h + self.b2_advantage)[0])
        return mean, std, margin, advantage

    def update_mean(self, x, target, lr=0.01):
        h, dh, x_arr = self._hidden(x)
        pred = float((self.W2_mean @ h + self.b2_mean)[0])
        err = target - pred
        self.W2_mean += lr * err * h.reshape(1, -1)
        self.b2_mean += lr * err
        dW1 = err * (self.W2_mean.T * dh.reshape(-1, 1)) @ x_arr.reshape(1, -1)
        db1 = err * (self.W2_mean.flatten() * dh)
        self.W1 += lr * dW1
        self.b1 += lr * db1

    def update_uncertainty(self, x, pred_error, lr=0.01):
        h, _, _ = self._hidden(x)
        logvar_target = -2.0 + np.log(max(pred_error**2, 1e-4))
        pred_logvar = float((self.W2_logvar @ h + self.b2_logvar)[0])
        self.W2_logvar += lr * (logvar_target - pred_logvar) * h.reshape(1, -1)
        self.b2_logvar += lr * (logvar_target - pred_logvar)

    def update_margin(self, x, target_margin, lr=0.01):
        h, dh, x_arr = self._hidden(x)
        pred = float((self.W2_margin @ h + self.b2_margin)[0])
        err = target_margin - pred
        self.W2_margin += lr * err * h.reshape(1, -1)
        self.b2_margin += lr * err
        dW1 = err * (self.W2_margin.T * dh.reshape(-1, 1)) @ x_arr.reshape(1, -1)
        db1 = err * (self.W2_margin.flatten() * dh)
        self.W1 += lr * 0.1 * dW1
        self.b1 += lr * 0.1 * db1

    def update_advantage(self, x, target_advantage, lr=0.01):
        h, dh, x_arr = self._hidden(x)
        pred = float((self.W2_advantage @ h + self.b2_advantage)[0])
        err = target_advantage - pred
        self.W2_advantage += lr * err * h.reshape(1, -1)
        self.b2_advantage += lr * err


class FinalGeneralizableController:
    """
    v2 controller with stability-margin-gated η and active probing.

    η = η_mean · sigmoid(margin)   where margin = predicted distance to instability.

    Active probe:
      Every N steps, push η slightly higher and observe if system diverges.
     """

    def __init__(self, lr=0.01, gamma=0.3,
                 probe_interval=20, probe_epsilon=0.08,
                 danger_loss_z=2.5, warning_loss_z=1.5,
                 conf_high=0.7, conf_low=0.3):
        self.student = GeneralizableMLPv2(input_dim=6, hidden=32)
        self.lr = lr
        self.gamma = gamma
        self.probe_interval = probe_interval
        self.probe_epsilon = probe_epsilon

        self.danger_loss_z = danger_loss_z
        self.warning_loss_z = warning_loss_z
        self.conf_high = conf_high
        self.conf_low = conf_low

        # Temporal memory
        self.d_prev = 0.0
        self.dd_prev = 0.0
        self.eta_prev = 0.5

        # Probes
        self.probe = ProbeResult()
        self._step_counter = 0
        self._probing = False
        self._probe_eta_saved = 0.0
        self._probe_loss_before = 0.0
        self._probe_active = False

        # History
        self.eta_history = []
        self.confidence_history = []
        self.fallback_history = []
        self.loss_history = []
        self.margin_history = []
        self.state_history = []

        # Teacher reference
        self.teacher_available = False
        self._teacher_etas = {}

        # OOD
        self._d_buffer = deque(maxlen=200)
        self._d_mean = 0.05
        self._d_std = 0.05

        # Near-instability tracking
        self._near_instability_count = 0
        self._safe_count = 0

    def set_teacher(self, teacher_dict):
        self._teacher_etas = dict(teacher_dict)
        self.teacher_available = len(self._teacher_etas) > 0

    def teacher(self, drift):
        if not self.teacher_available:
            return 0.5
        ds = sorted(self._teacher_etas.keys())
        if drift <= ds[0]:
            return self._teacher_etas[ds[0]]
        if drift >= ds[-1]:
            return self._teacher_etas[ds[-1]]
        for i in range(len(ds) - 1):
            if ds[i] <= drift <= ds[i + 1]:
                t = (drift - ds[i]) / (ds[i + 1] - ds[i])
                return self._teacher_etas[ds[i]] * (1 - t) + self._teacher_etas[ds[i + 1]] * t
        return 0.5

    def _is_ood(self, d):
        if len(self._d_buffer) < 20:
            return False
        return abs(d - self._d_mean) / max(self._d_std, 0.001) > 3.0

    def _detect_state(self, loss):
        self._loss_buffer_for_state = getattr(self, '_loss_buffer_for_state',
                                               deque(maxlen=100))
        self._loss_buffer_for_state.append(loss)
        if len(self._loss_buffer_for_state) < 20:
            return "SAFE", False, False

        mu = np.mean(self._loss_buffer_for_state)
        sigma = np.std(self._loss_buffer_for_state) + 1e-6
        z = (loss - mu) / sigma
        near = abs(z) > 1.5
        danger = z > self.danger_loss_z
        if danger:
            return "DANGER", True, near
        elif z > self.warning_loss_z:
            return "WARNING", danger, near
        return "SAFE", danger, near

    def _rule_fallback(self, drift):
        if drift < 0.02:
            return 0.4
        elif drift > 0.15:
            return 0.3
        return 0.5

    def forward(self, drift, loss=None, margin=0.0):
        self._step_counter += 1
        d1 = float(drift - self.d_prev)
        d2 = float(d1 - self.dd_prev)
        self.d_prev = float(drift)
        self.dd_prev = d1

        self._d_buffer.append(drift)
        if len(self._d_buffer) >= 20:
            self._d_mean = float(np.mean(self._d_buffer))
            self._d_std = float(np.std(self._d_buffer)) + 0.001

        state, is_danger, is_near = self._detect_state(loss) if loss is not None else ("SAFE", False, False)
        self.state_history.append(state)

        feat = np.array([drift, d1, d2, self.d_prev, self.eta_prev, margin])
        self._margin_input = float(margin)
        eta_mean, eta_std, mlp_margin, advantage = self.student.forward(feat)
        self.margin_history.append(float(mlp_margin))

        confidence = 1.0 / (1.0 + eta_std)
        self.confidence_history.append(float(confidence))

        # ── 1. Active probing ──
        should_probe = (self._step_counter % self.probe_interval == 0
                        and not is_danger and state == "SAFE")
        probe_eta = 0.0
        if should_probe:
            probe_eta = self.probe_epsilon
            self._probe_active = True
        else:
            self._probe_active = False

        # ── Performance-aware gating: w = sigmoid(advantage) ──
        w = 1.0 / (1.0 + np.exp(-advantage))
        eta_v1 = eta_mean + self.gamma       # aggressive mode: push up
        eta_v2 = eta_mean * (1.0 / (1.0 + np.exp(margin)))  # conservative: margin-gated
        eta_base = w * eta_v1 + (1.0 - w) * eta_v2
        eta = float(np.clip(eta_base + probe_eta, 0.01, 1.0))

        # OOD fallback
        ood = self._is_ood(drift)
        if ood or confidence < self.conf_low:
            eta = self._rule_fallback(drift)
            self.fallback_history.append("rule")
        elif confidence < self.conf_high:
            eta_learned = float(np.clip(eta_base, 0.01, 1.0))
            eta = 0.7 * eta_learned + 0.3 * self._rule_fallback(drift)
            self.fallback_history.append("damped")
        else:
            self.fallback_history.append("learned")

        eta = float(np.clip(eta, 0.01, 1.0))
        # ── Upper-bound protection from probe data ──
        if self.probe.attempts > 0:
            bound = self.probe.eta_upper_bound(percentile=90)
            if bound < 1.0:
                eta = min(eta, bound * 0.95)
        self.eta_prev = eta
        self.eta_history.append(eta)

        if loss is not None:
            self.loss_history.append(float(loss))

        # Near-instability counter
        if is_near:
            self._near_instability_count += 1
            self._safe_count = 0
        else:
            self._safe_count += 1

        return eta

    def update(self, drift, loss):
        """Update with near-instability gating and probe feedback."""
        state, is_danger, is_near = self._detect_state(loss)
        d1 = drift - self.d_prev
        d2 = d1 - self.dd_prev
        feat = np.array([drift, d1, d2, self.d_prev, self.eta_prev,
                         getattr(self, '_margin_input', 0.0)])

        # ── 1. Probe feedback ──
        if self._probe_active:
            survived = not is_danger
            self.probe.record(self.eta_prev, survived)
            # If probe caused divergence, tighten margin
            if not survived:
                self.student.update_margin(feat, -2.0, lr=self.lr * 2.0)
            self._probe_active = False

        # ── 2. Only update boundary model near instability ──
        if is_near:
            # Compute margin target: positive = far from danger, negative = close
            loss_deviation = 0.0
            if len(self.loss_history) >= 20:
                loss_deviation = (loss - np.mean(self.loss_history[-20:])) / max(np.std(self.loss_history[-20:]), 1e-6)
            margin_target = -loss_deviation  # high loss deviation → negative margin
            self.student.update_margin(feat, float(np.clip(margin_target, -3, 3)), lr=self.lr)

        # ── 3. Teacher distillation (always) ──
        if self.teacher_available:
            target = self.teacher(drift)
            if is_near and not is_danger:
                upper_bound = self.probe.eta_upper_bound(percentile=80)
                target = min(target, upper_bound * 0.95)
            if is_danger:
                target *= 0.6
            self.student.update_mean(feat, float(target), lr=self.lr)

        # ── 4. Uncertainty calibration ──
        if len(self.loss_history) >= 10:
            _, eta_std_local, _, _ = self.student.forward(feat)
            pred_err = abs(self.eta_prev - self.eta_history[-2]) if len(self.eta_history) >= 2 else 0.01
            self.student.update_uncertainty(feat, pred_err, lr=self.lr * 0.1)

        # ── 5. Advantage learning: was v1 better than v2? ──
        if len(self.loss_history) >= 20:
            recent = self.loss_history[-10:]
            older = self.loss_history[-20:-10]
            loss_trend = float(np.mean(recent)) - float(np.mean(older))
            # loss decreasing → v1 aggression paid off → positive advantage
            # loss increasing → conservative would have been safer → negative advantage
            adv_target = -np.clip(loss_trend / (np.std(self.loss_history[-20:]) + 1e-6), -2, 2)
            self.student.update_advantage(feat, float(adv_target), lr=self.lr)

        self._d_buffer.append(drift)
        if len(self._d_buffer) >= 20:
            self._d_mean = float(np.mean(self._d_buffer))
            self._d_std = float(np.std(self._d_buffer)) + 0.001

    def param_summary(self):
        fb = self.fallback_history[-200:] if self.fallback_history else []
        return {
            "eta_mean": float(np.mean(self.eta_history[-100:])) if self.eta_history else 0,
            "avg_confidence": float(np.mean(self.confidence_history[-100:])) if self.confidence_history else 0,
            "avg_margin": float(np.mean(self.margin_history[-100:])) if self.margin_history else 0,
            "fallback_dist": {"learned": fb.count("learned"), "damped": fb.count("damped"), "rule": fb.count("rule")},
            "probe_survival_rate": self.probe.survival_rate(),
            "probe_eta_upper_bound": self.probe.eta_upper_bound(),
            "probe_attempts": self.probe.attempts,
            "near_instability_count": self._near_instability_count,
        }


# ── Danger-aware teacher generator ──────────────────────────────────
def generate_danger_aware_teacher(steps_per_drift=200):
    """
    Build teacher that captures the danger boundary, not just monotonic.
    Probes η upward at each drift to find the collapse point.
    """
    import torch, random
    import torch.nn.functional as F
    from experiment10 import reset_seed, _make_agent, _make_ctrl
    from env_drifting_double_well import DriftingDoubleWell

    teacher = {}
    drift_grid = [0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.18]

    for drift_val in drift_grid:
        reset_seed(42)
        env = DriftingDoubleWell(kappa=1.0, drift_rate=drift_val,
                                 flip_mode="deterministic", add_context=False)
        agent = _make_agent(env)
        ctrl = _make_ctrl(agent, 4, inertia=0.0)

        obs = env.reset()
        ctrl.gating_reset()

        eta_current = 0.5
        eta_history = []
        loss_window = deque(maxlen=30)
        survived = True
        probe_eta = 0.0

        for t in range(steps_per_drift):
            if t > 100 and t % 30 == 0 and survived:
                eta_current += 0.05  # push upward

            if hasattr(ctrl.gating, 'inertia'):
                ctrl.gating.inertia = float(eta_current)
                ctrl.inertia = float(eta_current)

            a = agent.act(obs)
            o_next, _, done = env.step(a)
            target_t = torch.tensor(o_next, dtype=torch.float32)
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            if ctrl.n_models() > 0:
                with torch.no_grad():
                    z_soft, z_logits, _ = ctrl.gating(s)
                K_cur = z_soft.size(-1)
                z_hard = F.one_hot(z_logits.argmax(dim=-1), K_cur).float()
                weights = (z_hard - z_soft.detach() + z_soft).squeeze(0)
                preds = [m.predict(obs, a) for m in ctrl.models]
                soft_pred = sum(weights[i] * preds[i] for i in range(K_cur))
                loss_val = ((soft_pred - target_t) ** 2).mean()
                step_loss = float(loss_val.item())

                ctrl.gating_optimizer.zero_grad()
                for m in ctrl.models:
                    m.optimizer.zero_grad()
                loss_val.backward()
                ctrl.gating_optimizer.step()
                for m in ctrl.models:
                    m.optimizer.step()
            else:
                step_loss = 1.0

            loss_window.append(step_loss)
            eta_history.append(eta_current)

            # Detect divergence
            if len(loss_window) >= 20:
                mu = np.mean(loss_window)
                sigma = np.std(loss_window) + 1e-6
                z = (step_loss - mu) / sigma
                if z > 4.0:
                    survived = False
                    eta_current = max(0.1, eta_current - 0.1)

            if done:
                obs = env.reset()
                ctrl.gating_reset()
            else:
                obs = o_next

        final_eta = float(np.mean(eta_history[-30:])) if len(eta_history) >= 30 else float(np.mean(eta_history))
        teacher[float(drift_val)] = final_eta
        print(f"  teacher drift={drift_val:.3f} → η={final_eta:.3f}")

    return teacher


# ── Unified comparison ───────────────────────────────────────────────
def run_final_comparison(env_class, kappa=1.0, drift=0.05, steps=500):
    import torch, random
    import torch.nn.functional as F
    from experiment10 import reset_seed, _make_agent, _make_ctrl
    from online_controller import OnlineStabilityController
    from optimal_controller import OptimalEtaController
    from generalizable_controller import GeneralizableController as GenV1

    configs = [
        ("A_baseline", "fixed"),
        ("C_optimal", "optimal"),
        ("D_gen_v1", "gen_v1"),
        ("E_gen_v2", "gen_v2"),
    ]
    results = {}

    for mode, ctrl_type in configs:
        reset_seed(42)
        env = env_class(kappa=kappa, drift_rate=drift,
                        flip_mode="deterministic", add_context=False)
        agent = _make_agent(env)
        ctrl = _make_ctrl(agent, 4, inertia=0.0)

        if ctrl_type == "optimal":
            online = OptimalEtaController(lr_ctrl=0.01, lr_env=0.005,
                                          lambda_boundary=0.2, lambda_danger=0.5)
        elif ctrl_type == "gen_v1":
            online = GenV1(lr=0.01, gamma=0.3)
            try:
                with open("teacher_policy.json") as f:
                    online.set_teacher({float(k): float(v) for k, v in json.load(f).items()})
            except FileNotFoundError:
                pass
        elif ctrl_type == "gen_v2":
            online = FinalGeneralizableController(lr=0.01, gamma=0.3)
            try:
                with open("danger_teacher.json") as f:
                    online.set_teacher({float(k): float(v) for k, v in json.load(f).items()})
            except FileNotFoundError:
                pass
        else:
            online = None

        loss_history, eta_history = [], []
        obs = env.reset()
        if ctrl_type != "fixed":
            ctrl.gating_reset()
        prev_margin = 0.0

        for t in range(steps):
            drift_val = float(getattr(env, 'drift', drift))

            if ctrl_type == "gen_v2":
                new_eta = online.forward(drift_val, loss=loss_history[-1] if loss_history else None,
                                         margin=prev_margin)
            elif ctrl_type in ("optimal", "gen_v1"):
                new_eta = online.forward(drift_val, loss=loss_history[-1] if loss_history else None)
            else:
                new_eta = 0.0
            eta_history.append(new_eta)

            if ctrl_type != "fixed" and hasattr(ctrl.gating, 'inertia'):
                ctrl.gating.inertia = float(new_eta)
                ctrl.inertia = float(new_eta)

            a = agent.act(obs)
            o_next, _, done = env.step(a)
            target_t = torch.tensor(o_next, dtype=torch.float32)
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            if ctrl.n_models() > 0:
                with torch.no_grad():
                    z_soft, z_logits, _ = ctrl.gating(s)
                K_cur = z_soft.size(-1)
                z_hard = F.one_hot(z_logits.argmax(dim=-1), K_cur).float()
                weights = (z_hard - z_soft.detach() + z_soft).squeeze(0)
                preds = [m.predict(obs, a) for m in ctrl.models]
                soft_pred = sum(weights[i] * preds[i] for i in range(K_cur))
                loss_val = ((soft_pred - target_t) ** 2).mean()
                step_loss = float(loss_val.item())

                expert_errors = {}
                oracle_err = float("inf")
                with torch.no_grad():
                    perr = torch.stack([
                        ((preds[i].detach() - target_t) ** 2).mean()
                        for i in range(K_cur)
                    ])
                    oracle_err = perr.min().item()
                for i in range(K_cur):
                    e_i = float(np.mean(np.abs(preds[i].detach().numpy() - o_next)))
                    expert_errors[i] = e_i
                if K_cur >= 2:
                    sorted_errs = np.sort(list(expert_errors.values()))
                    prev_margin = float(sorted_errs[1] - sorted_errs[0])
                else:
                    prev_margin = 0.0

                ctrl.gating_optimizer.zero_grad()
                for m in ctrl.models:
                    m.optimizer.zero_grad()
                loss_val.backward()
                ctrl.gating_optimizer.step()
                for m in ctrl.models:
                    m.optimizer.step()
            else:
                step_loss = 1.0

            loss_history.append(step_loss)
            if ctrl_type in ("optimal", "gen_v1", "gen_v2"):
                online.update(drift_val, step_loss)

            ctrl.maybe_merge()
            ctrl.maybe_prune()
            obs = env.reset() if done else o_next
            if done and ctrl_type != "fixed":
                ctrl.gating_reset()

        w = max(20, steps // 10)
        conv = float(np.mean(loss_history[:w])) / max(float(np.mean(loss_history[-w:])), 1e-8)
        results[mode] = {
            "convergence_ratio": conv,
            "loss_volatility": float(np.std(loss_history)),
            "eta_avg": float(np.mean(eta_history)),
        }
        if ctrl_type == "gen_v2":
            results[mode]["params"] = online.param_summary()

        info = f"conv={conv:.2f}x vol={np.std(loss_history):.4f} eta={np.mean(eta_history):.3f}"
        if ctrl_type == "gen_v2":
            ps = online.param_summary()
            info += (f"  probe_surv={ps['probe_survival_rate']:.2f} "
                     f"η_bound={ps['probe_eta_upper_bound']:.3f} "
                     f"margin={ps['avg_margin']:.3f} "
                     f"near={ps['near_instability_count']}")
        print(f"{mode:>15}: {info}")

    return results


if __name__ == "__main__":
    from env_drifting_double_well import DriftingDoubleWell

    print("=" * 60)
    print("FINAL GENERALIZABLE CONTROLLER v2")
    print("  + active probing + margin head + danger-aware teacher")
    print("=" * 60)

    # Step 1: Build danger-aware teacher
    print("\n── Building danger-aware teacher ──")
    teacher = generate_danger_aware_teacher(steps_per_drift=200)
    with open("danger_teacher.json", "w") as f:
        json.dump(teacher, f, indent=2)
    print(f"Danger-aware teacher saved ({len(teacher)} points)")

    # Step 2: Importance-sampled comparison
    print(f"\n{'='*60}")
    print("A/C/D/E: baseline vs optimal vs gen_v1 vs gen_v2")
    print(f"{'='*60}")

    all_results = {}
    # Focus on the complex regime + edges
    for d_test in [0.02, 0.05, 0.08, 0.10, 0.12, 0.14]:
        print(f"\n── drift = {d_test} ──")
        res = run_final_comparison(DriftingDoubleWell, kappa=1.0, drift=d_test, steps=500)
        all_results[d_test] = res

    print(f"\n{'='*75}")
    print("FINAL v2 COMPARISON")
    print(f"{'='*75}")
    hdr = f"{'drift':>6} {'A(base)':>11} {'C(opt)':>11} {'D(gen_v1)':>11} {'E(gen_v2)':>11}  best"
    print(hdr)
    print("-" * 75)
    for d_test in all_results:
        A = all_results[d_test].get("A_baseline", {"convergence_ratio": 0, "loss_volatility": 0})
        C = all_results[d_test].get("C_optimal", {"convergence_ratio": 0, "loss_volatility": 0})
        D = all_results[d_test].get("D_gen_v1", {"convergence_ratio": 0, "loss_volatility": 0})
        E = all_results[d_test].get("E_gen_v2", {"convergence_ratio": 0, "loss_volatility": 0})
        best = max([(A["convergence_ratio"], "A"), (C["convergence_ratio"], "C"),
                     (D["convergence_ratio"], "D"), (E["convergence_ratio"], "E")],
                   key=lambda x: x[0])
        print(f"{d_test:>6.3f}  {A['convergence_ratio']:>5.2f}x {A['loss_volatility']:>5.3f}  "
              f"{C['convergence_ratio']:>5.2f}x {C['loss_volatility']:>5.3f}  "
              f"{D['convergence_ratio']:>5.2f}x {D['loss_volatility']:>5.3f}  "
              f"{E['convergence_ratio']:>5.2f}x {E['loss_volatility']:>5.3f}  → {best[1]}")

    with open("final_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to final_v2_results.json")
