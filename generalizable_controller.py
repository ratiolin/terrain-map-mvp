"""
Generalizable Stability Controller — the final unified system.

Architecture:
  Teacher:   pi_teacher(d) = optimal η from previous controller (distilled)
  Student:   f_θ(d, d', d'', d_prev, η_prev) → (η_mean, η_std)
  Policy:    η = η_mean - β * η_std   (conservative under uncertainty)
  OOD:       if confidence < threshold → fallback to rule-based
  Temporal:  η_t = f(d_t, d_{t-1}, η_{t-1}) with second-order inputs

Unified fallback:
  high_confidence → learned
  medium_confidence → damped learned  
  low_confidence / OOD → rule-based
"""
import json
import numpy as np
from collections import deque


class GeneralizableMLP:
    """MLP with uncertainty head: outputs (mean, logvar)."""

    def __init__(self, input_dim, hidden=32):
        k = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(hidden, input_dim) * k
        self.b1 = np.zeros(hidden)
        self.W2_mean = np.random.randn(1, hidden) * np.sqrt(2.0 / hidden)
        self.b2_mean = np.array([0.3])
        self.W2_logvar = np.random.randn(1, hidden) * 0.01
        self.b2_logvar = np.array([-2.0])

    def forward(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        h = np.tanh(self.W1 @ x_arr + self.b1)
        mean = float((self.W2_mean @ h + self.b2_mean)[0])
        logvar = float((self.W2_logvar @ h + self.b2_logvar)[0])
        std = float(np.exp(0.5 * np.clip(logvar, -5, 2)))
        return mean, std

    def update(self, x, target_mean, lr=0.01):
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        h_pre = self.W1 @ x_arr + self.b1
        h = np.tanh(h_pre)
        dh = (1 - h**2)

        pred_mean = float((self.W2_mean @ h + self.b2_mean)[0])
        err = target_mean - pred_mean

        self.W2_mean += lr * err * h.reshape(1, -1)
        self.b2_mean += lr * err

        dW1 = err * (self.W2_mean.T * dh.reshape(-1, 1)) @ x_arr.reshape(1, -1)
        db1 = err * (self.W2_mean.flatten() * dh)
        self.W1 += lr * dW1
        self.b1 += lr * db1

        # Uncertainty update: push logvar toward prediction error
        logvar_target = -2.0 + np.log(max(err**2, 1e-4))
        pred_logvar = float((self.W2_logvar @ h + self.b2_logvar)[0])
        self.W2_logvar += lr * 0.1 * (logvar_target - pred_logvar) * h.reshape(1, -1)
        self.b2_logvar += lr * 0.1 * (logvar_target - pred_logvar)


class GeneralizableController:
    """
    Student model: f_θ(d, d', d'', η_prev) → (η_mean, η_std)

    Trained by distilling teacher policy + online loss feedback.

    Fallback strategy:
      confidence = 1 / (1 + η_std)
      high   (>0.7) → learned
      medium (0.3-0.7) → damped learned
      low    (<0.3) → rule-based
    """

    def __init__(self, lr=0.01, gamma=0.3,
                 conf_high=0.7, conf_low=0.3):
        self.student = GeneralizableMLP(input_dim=5, hidden=32)  # d, d', d'', d_prev, η_prev
        self.lr = lr
        self.gamma = gamma
        self.conf_high = conf_high
        self.conf_low = conf_low

        # Temporal memory
        self.d_prev = 0.0
        self.dd_prev = 0.0
        self.eta_prev = 0.5

        # OOD detection
        self._d_buffer = deque(maxlen=200)
        self._d_mean = 0.05
        self._d_std = 0.05

        # Teacher reference (initialized from optimal controller or rule)
        self.teacher_available = False
        self._teacher_etas = {}

        # History
        self.eta_history = []
        self.confidence_history = []
        self.fallback_history = []
        self.loss_history = []

    def set_teacher(self, teacher_dict):
        """Set teacher policy: {drift: eta_optimal}."""
        self._teacher_etas = dict(teacher_dict)
        self.teacher_available = len(self._teacher_etas) > 0

    def teacher(self, drift):
        """Interpolate teacher policy."""
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

    def _compute_derivatives(self, d):
        """Compute first and second derivatives of drift."""
        d1 = d - self.d_prev
        d2 = d1 - self.dd_prev
        self.d_prev = d
        self.dd_prev = d1
        return d1, d2

    def _is_ood(self, d):
        """Check if drift is out of training distribution."""
        if len(self._d_buffer) < 20:
            return False
        z = abs(d - self._d_mean) / max(self._d_std, 0.001)
        return z > 3.0

    def forward(self, drift, loss=None):
        """
        Compute η(d) with:
          1. Student model: η_mean, η_std = f(d, d', d'', d_prev, η_prev)
          2. Conservative: η = η_mean - β * η_std
          3. OOD detection → fallback
          4. Unified confidence-gating
        """
        d1, d2 = self._compute_derivatives(drift)
        self._d_buffer.append(drift)
        if len(self._d_buffer) >= 20:
            self._d_mean = float(np.mean(self._d_buffer))
            self._d_std = float(np.std(self._d_buffer)) + 0.001

        # Student prediction
        feat = np.array([drift, d1, d2, self.d_prev, self.eta_prev])
        eta_mean, eta_std = self.student.forward(feat)
        confidence = 1.0 / (1.0 + eta_std)
        self.confidence_history.append(float(confidence))

        # OOD check
        ood = self._is_ood(drift)

        # Unified fallback
        if ood or confidence < self.conf_low:
            # Rule-based fallback
            eta = self._rule_fallback(drift)
            self.fallback_history.append("rule")
        elif confidence < self.conf_high:
            # Damped learned: η = η_mean + γ·sigmoid(confidence-0.5)
            margin = confidence - 0.5
            eta_learned = eta_mean + self.gamma * (1.0 / (1.0 + np.exp(-margin * 10)))
            eta = 0.7 * eta_learned + 0.3 * self._rule_fallback(drift)
            self.fallback_history.append("damped")
        else:
            # Pure learned: η = η_mean + γ·sigmoid(implicit_margin)
            margin = confidence - 0.5
            eta = eta_mean + self.gamma * (1.0 / (1.0 + np.exp(-margin * 10)))
            self.fallback_history.append("learned")

        eta = float(np.clip(eta, 0.01, 1.0))
        self.eta_prev = eta
        self.eta_history.append(eta)

        if loss is not None:
            self.loss_history.append(float(loss))

        return eta

    def _rule_fallback(self, drift):
        """Simple rule-based fallback: lower η at extreme drifts."""
        if drift < 0.02:
            return 0.4
        elif drift > 0.15:
            return 0.3
        else:
            return 0.5

    def update(self, drift, loss):
        """Distillation update + online refinement."""
        d1, d2 = self._compute_derivatives(drift)
        feat = np.array([drift, d1, d2, self.d_prev, self.eta_prev])

        # Teacher target
        if self.teacher_available:
            target = self.teacher(drift)
            # Blend with loss-based feedback
            if len(self.loss_history) >= 20:
                loss_z = (loss - np.mean(self.loss_history[-20:])) / max(np.std(self.loss_history[-20:]), 1e-6)
                if loss_z > 2.0:
                    target *= 0.7  # Reduce target when loss spiking
            self.student.update(feat, target, lr=self.lr)

        # Online OOD updates: expand known distribution
        self._d_buffer.append(drift)
        if len(self._d_buffer) >= 20:
            self._d_mean = float(np.mean(self._d_buffer))
            self._d_std = float(np.std(self._d_buffer)) + 0.001

    def param_summary(self):
        fb = self.fallback_history[-200:] if self.fallback_history else []
        return {
            "eta_mean": float(np.mean(self.eta_history[-100:])) if self.eta_history else 0,
            "avg_confidence": float(np.mean(self.confidence_history[-100:])) if self.confidence_history else 0,
            "fallback_dist": {
                "learned": fb.count("learned"),
                "damped": fb.count("damped"),
                "rule": fb.count("rule"),
            },
            "ood_d_mean": float(self._d_mean),
            "ood_d_std": float(self._d_std),
        }


# ── Teacher generation ────────────────────────────────────────────────
def generate_teacher_policy(steps_per_drift=300):
    """
    Run optimal controller on multiple drifts to build teacher policy.
    Returns: {drift: optimal_eta, ...}
    """
    from optimal_controller import OptimalEtaController
    import torch, random
    import torch.nn.functional as F
    from experiment10 import reset_seed, _make_agent, _make_ctrl
    from env_drifting_double_well import DriftingDoubleWell

    teacher = {}
    drift_grid = np.linspace(0.01, 0.18, 10)

    for drift in drift_grid:
        reset_seed(42)
        env = DriftingDoubleWell(kappa=1.0, drift_rate=drift,
                                 flip_mode="deterministic", add_context=False)
        agent = _make_agent(env)
        ctrl = _make_ctrl(agent, 4, inertia=0.0)
        online = OptimalEtaController(lr_ctrl=0.01, lr_env=0.005,
                                      lambda_boundary=0.2, lambda_danger=0.5)

        obs = env.reset()
        ctrl.gating_reset()
        eta_history = []

        for t in range(steps_per_drift):
            drift_val = float(drift)
            new_eta = online.forward(drift_val, loss=eta_history[-1] if eta_history else None)
            eta_history.append(new_eta)

            if hasattr(ctrl.gating, 'inertia'):
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

                ctrl.gating_optimizer.zero_grad()
                for m in ctrl.models:
                    m.optimizer.zero_grad()
                loss_val.backward()
                ctrl.gating_optimizer.step()
                for m in ctrl.models:
                    m.optimizer.step()
            else:
                step_loss = 1.0

            online.update(drift_val, step_loss)
            if done:
                obs = env.reset()
                ctrl.gating_reset()
            else:
                obs = o_next

        final_eta = float(np.mean(eta_history[-50:])) if len(eta_history) >= 50 else float(np.mean(eta_history))
        teacher[float(drift)] = final_eta
        print(f"  teacher drift={drift:.3f} → η={final_eta:.3f}")

    return teacher


# ── Full comparison ────────────────────────────────────────────────────
def run_unified_comparison(env_class, kappa=1.0, drift=0.05, steps=500):
    import torch, random
    import torch.nn.functional as F
    from experiment10 import reset_seed, _make_agent, _make_ctrl
    from online_controller import OnlineStabilityController
    from optimal_controller import OptimalEtaController

    configs = [
        ("A_baseline", "fixed"),
        ("B_rule", "rule"),
        ("C_optimal", "optimal"),
        ("D_generalizable", "generalizable"),
    ]
    results = {}

    for mode, ctrl_type in configs:
        reset_seed(42)
        env = env_class(kappa=kappa, drift_rate=drift,
                        flip_mode="deterministic", add_context=False)
        agent = _make_agent(env)
        ctrl = _make_ctrl(agent, 4, inertia=0.0)

        if ctrl_type == "rule":
            online = OnlineStabilityController()
        elif ctrl_type == "optimal":
            online = OptimalEtaController(lr_ctrl=0.01, lr_env=0.005,
                                          lambda_boundary=0.2, lambda_danger=0.5)
        elif ctrl_type == "generalizable":
            online = GeneralizableController(lr=0.01, beta_conservative=0.5)
            # Load teacher if available
            try:
                with open("teacher_policy.json") as f:
                    teacher = json.load(f)
                    teacher = {float(k): float(v) for k, v in teacher.items()}
                online.set_teacher(teacher)
            except FileNotFoundError:
                pass
        else:
            online = None

        loss_history, eta_history = [], []
        obs = env.reset()
        if ctrl_type != "fixed":
            ctrl.gating_reset()

        fb_counts = {"learned": 0, "damped": 0, "rule": 0}

        for t in range(steps):
            drift_val = float(getattr(env, 'drift', drift))

            if ctrl_type == "rule":
                new_eta, state = online.step(drift_val)
            elif ctrl_type in ("optimal", "generalizable"):
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

            if ctrl_type == "optimal":
                online.update(drift_val, step_loss)
            elif ctrl_type == "generalizable":
                online.update(drift_val, step_loss)

            ctrl.maybe_merge()
            ctrl.maybe_prune()

            if done:
                obs = env.reset()
                if ctrl_type != "fixed":
                    ctrl.gating_reset()
            else:
                obs = o_next

        w = max(20, steps // 10)
        early = float(np.mean(loss_history[:w]))
        late = float(np.mean(loss_history[-w:]))
        conv = early / max(late, 1e-8)

        results[mode] = {
            "early_loss": early, "late_loss": late,
            "convergence_ratio": conv,
            "loss_volatility": float(np.std(loss_history)),
            "loss_mean": float(np.mean(loss_history)),
            "eta_avg": float(np.mean(eta_history)),
            "loss_history": [float(l) for l in loss_history],
            "eta_history": [float(e) for e in eta_history],
        }

        if ctrl_type == "generalizable":
            results[mode]["params"] = online.param_summary()

        fb_label = ""
        if ctrl_type == "generalizable":
            ps = online.param_summary()
            fb = ps["fallback_dist"]
            fb_label = (f"  fb:L={fb['learned']} D={fb['damped']} R={fb['rule']} "
                        f"conf={ps['avg_confidence']:.3f}")

        print(f"{mode:>20}: conv={conv:.2f}x  vol={np.std(loss_history):.4f}  "
              f"eta_avg={np.mean(eta_history):.3f}{fb_label}")

    return results


if __name__ == "__main__":
    from env_drifting_double_well import DriftingDoubleWell

    print("=" * 60)
    print("GENERALIZABLE CONTROLLER — Teacher Distillation")
    print("=" * 60)

    # Step 1: Generate teacher policy
    print("\n── Generating teacher policy (optimal controller on drift grid) ──")
    teacher = generate_teacher_policy(steps_per_drift=200)
    with open("teacher_policy.json", "w") as f:
        json.dump(teacher, f, indent=2)
    print(f"Teacher policy saved to teacher_policy.json ({len(teacher)} points)")

    # Step 2: Unified comparison
    print(f"\n{'='*60}")
    print("A/B/C/D: baseline vs rule vs optimal vs GENERALIZABLE")
    print(f"{'='*60}")

    all_results = {}
    for d_test in [0.02, 0.05, 0.10, 0.14]:
        print(f"\n── drift = {d_test} ──")
        res = run_unified_comparison(DriftingDoubleWell, kappa=1.0, drift=d_test, steps=500)
        all_results[d_test] = res

    print(f"\n{'='*70}")
    print("FINAL UNIFIED COMPARISON")
    print(f"{'='*70}")
    print(f"{'drift':>6} {'A(base)':>10} {'B(rule)':>10} {'C(opt)':>10} {'D(gen)':>10}  winner")
    print(f"{'':>6} {'conv':>5} {'vol':>5} {'conv':>5} {'vol':>5} {'conv':>5} {'vol':>5} {'conv':>5} {'vol':>5}")
    print("-" * 70)

    for d_test in all_results:
        A = all_results[d_test]["A_baseline"]
        B = all_results[d_test]["B_rule"]
        C = all_results[d_test]["C_optimal"]
        D = all_results[d_test]["D_generalizable"]

        best = max([(A["convergence_ratio"], "A"), (B["convergence_ratio"], "B"),
                     (C["convergence_ratio"], "C"), (D["convergence_ratio"], "D")],
                   key=lambda x: x[0])
        best_vol = min([(A["loss_volatility"], "A"), (B["loss_volatility"], "B"),
                         (C["loss_volatility"], "C"), (D["loss_volatility"], "D")],
                        key=lambda x: x[0])

        print(f"{d_test:>6.3f} "
              f"{A['convergence_ratio']:>5.2f} {A['loss_volatility']:>5.4f} "
              f"{B['convergence_ratio']:>5.2f} {B['loss_volatility']:>5.4f} "
              f"{C['convergence_ratio']:>5.2f} {C['loss_volatility']:>5.4f} "
              f"{D['convergence_ratio']:>5.2f} {D['loss_volatility']:>5.4f}  "
              f"conv={best[1]} vol={best_vol[1]}")

    with open("generalizable_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to generalizable_results.json")
