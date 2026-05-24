"""
Learned η(d) controller — online gradient optimization of the stability policy.

Parameterization:  η(d) = clip( a · sigmoid(b·d + c), 0, eta_max(d) )
Learnable:  θ = (a, b, c)

Objective:  reward = -loss
            penalty = λ1·danger_flag + λ2·warning_flag
            maximize:  reward - penalty

Online update:  θ ← θ + lr · ∂obj/∂θ   (via score-function gradient estimator)

Exploration:  η ← η · (1 + ε),  ε ~ U(-0.05, 0.05)

Comparison:  baseline (fixed η) vs rule-based vs learned
"""
import json
import numpy as np
from collections import deque

from strategy import StabilityController as BaseStabilityController


class LearnedEtaController:
    """
    Learned η(d) with parameterized sigmoid + online gradient updates.
    """

    def __init__(self, lr=0.01, lambda_danger=0.5, lambda_warning=0.1,
                 explore_std=0.05):
        self.base = BaseStabilityController()

        # Parameterized policy: η(d) = a · sigmoid(b·d + c)
        self.a = 1.0
        self.b = -10.0
        self.c = 0.5

        self.lr = lr
        self.lambda_danger = lambda_danger
        self.lambda_warning = lambda_warning
        self.explore_std = explore_std

        # History for learning curve
        self.theta_history = [(self.a, self.b, self.c)]
        self.loss_history = []
        self.eta_history = []
        self.reward_history = []
        self.danger_rate_history = []

        self.step_count = 0
        self._loss_buffer = deque(maxlen=20)
        self._eta_buffer = deque(maxlen=20)
        self._danger_buffer = deque(maxlen=50)

    def eta_raw(self, drift):
        """Raw parameterized eta before constraints."""
        return float(self.a * self._sigmoid(self.b * drift + self.c))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _sigmoid_deriv(self, x):
        s = self._sigmoid(x)
        return s * (1.0 - s)

    def forward(self, drift, explore=True):
        """
        Compute η(d) with:
        1. Parameterized function
        2. Hard constraint: η ≤ η_max(d)
        3. Danger zone: η *= 0.5
        4. Exploration noise
        """
        eta_raw = self.eta_raw(drift)
        eta_max = self.base.eta_max(drift)

        # Hard constraint
        eta = min(eta_raw, eta_max)

        # Danger zone protection
        if self.base.is_unstable(drift):
            eta *= 0.5

        # Exploration noise
        if explore:
            eps = np.random.uniform(-self.explore_std, self.explore_std)
            eta *= (1.0 + eps)

        eta = max(0.01, min(eta, 1.0))
        self.eta_history.append(float(eta))
        return float(eta)

    def _compute_objective(self, loss, is_danger, is_warning):
        """Compute objective = -loss - λ1·danger - λ2·warning."""
        return -loss - self.lambda_danger * float(is_danger) - self.lambda_warning * float(is_warning)

    def update(self, drift, loss, is_danger, is_warning):
        """
        Online gradient update using score-function estimator.
        θ ← θ + lr · obj · ∇_θ log π_θ(η|d)

        Simplified: use REINFORCE-style on the continuous action.
        ∇_θ η ≈ [η, η·d·(1-sigmoid), η·(1-sigmoid)]
        """
        self.step_count += 1
        self._loss_buffer.append(loss)
        self._danger_buffer.append(int(is_danger))

        # Compute objective
        obj = self._compute_objective(loss, is_danger, is_warning)
        self.reward_history.append(float(obj))

        # Current η and sigmoid for gradient computation
        x = self.b * drift + self.c
        s = self._sigmoid(x)
        ds = self._sigmoid_deriv(x)

        # Gradient directions (chain rule through η(d) = a·sigmoid(b·d + c))
        grad_a = s
        grad_b = self.a * ds * drift
        grad_c = self.a * ds

        # Apply learning rule: θ ← θ + lr * (objective_baselined) * grad
        baseline = np.mean(self.reward_history[-50:]) if len(self.reward_history) >= 50 else 0.0
        advantage = obj - baseline

        self.a += self.lr * advantage * grad_a
        self.b += self.lr * advantage * grad_b
        self.c += self.lr * advantage * grad_c

        # Clamp params
        self.a = max(0.1, min(self.a, 2.0))
        self.b = max(-50, min(self.b, 0.0))
        self.c = max(-1.0, min(self.c, 3.0))

        self.theta_history.append((float(self.a), float(self.b), float(self.c)))
        self.loss_history.append(float(loss))

    def danger_rate(self):
        if not self._danger_buffer:
            return 0.0
        return float(np.mean(self._danger_buffer))

    def param_summary(self):
        return {
            "a": float(self.a),
            "b": float(self.b),
            "c": float(self.c),
            "eta_at_d01": self.forward(0.01, explore=False),
            "eta_at_d05": self.forward(0.05, explore=False),
            "eta_at_d10": self.forward(0.10, explore=False),
            "eta_at_d15": self.forward(0.15, explore=False),
            "danger_rate": self.danger_rate(),
            "avg_reward": float(np.mean(self.reward_history[-100:])) if self.reward_history else 0,
        }


# ── A/B/C comparison ─────────────────────────────────────────────────
def run_comparison(env_class, kappa=1.0, drift=0.05, steps=800):
    """
    Compare three controllers:
      A: baseline (fixed η = 0.0)
      B: rule-based (OnlineStabilityController from online_controller.py)
      C: learned (LearnedEtaController)
    """
    import torch
    import random
    import torch.nn.functional as F

    from experiment10 import reset_seed, _make_agent, _make_ctrl
    from online_controller import OnlineStabilityController

    results = {}

    for mode, ctrl_type in [("A_baseline", "fixed"),
                              ("B_rule_based", "rule"),
                              ("C_learned", "learned")]:
        reset_seed(42)

        env = env_class(kappa=kappa, drift_rate=drift,
                        flip_mode="deterministic", add_context=False)
        agent = _make_agent(env)
        ctrl = _make_ctrl(agent, 4, inertia=0.0)

        if ctrl_type == "rule":
            online = OnlineStabilityController()
        elif ctrl_type == "learned":
            online = LearnedEtaController(lr=0.02, lambda_danger=0.5, lambda_warning=0.1)
        else:
            online = None

        loss_history = []
        eta_history = []
        danger_flags = []
        warning_flags = []

        obs = env.reset()
        if ctrl_type != "fixed":
            ctrl.gating_reset()

        for t in range(steps):
            drift_val = float(getattr(env, 'drift', drift))

            if ctrl_type == "rule":
                new_eta, state = online.step(drift_val)
                eta_history.append(new_eta)
            elif ctrl_type == "learned":
                new_eta = online.forward(drift_val)
                eta_history.append(new_eta)
            else:
                new_eta = 0.0
                eta_history.append(0.0)

            if ctrl_type != "fixed":
                if hasattr(ctrl.gating, 'inertia'):
                    ctrl.gating.inertia = float(new_eta)
                ctrl.inertia = float(new_eta)
                ctrl.gating.inertia = float(new_eta)

            a = agent.act(obs)
            o_next, _, done = env.step(a)
            target = torch.tensor(o_next, dtype=torch.float32)
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            if ctrl.n_models() > 0:
                with torch.no_grad():
                    z_soft, z_logits, _ = ctrl.gating(s)
                K_cur = z_soft.size(-1)
                z_hard_idx = z_logits.argmax(dim=-1)
                z_hard = F.one_hot(z_hard_idx, K_cur).float()
                weights = (z_hard - z_soft.detach() + z_soft).squeeze(0)
                preds = [m.predict(obs, a) for m in ctrl.models]
                soft_pred = sum(weights[i] * preds[i] for i in range(K_cur))
                loss_val = ((soft_pred - target) ** 2).mean()

                error_val = float(np.mean(np.abs(soft_pred.detach().numpy() - o_next)))
                ctrl.should_update(error_val, 0)
                ctrl.record_usage(int(weights.argmax().item()))
                for i in range(K_cur):
                    e_i = float(np.mean(np.abs(preds[i].detach().numpy() - o_next)))
                    ctrl.track_error(i, e_i)

                ctrl.gating_optimizer.zero_grad()
                for m in ctrl.models:
                    m.optimizer.zero_grad()
                loss_val.backward()
                ctrl.gating_optimizer.step()
                for m in ctrl.models:
                    m.optimizer.step()

                step_loss = float(loss_val.item())
            else:
                step_loss = 1.0

            loss_history.append(step_loss)

            # Determine danger/warning for learning
            is_danger = step_loss > 3.0 * np.mean(loss_history[-50:]) if len(loss_history) >= 50 else False
            is_warning = step_loss > 1.5 * np.mean(loss_history[-50:]) if len(loss_history) >= 50 else False
            danger_flags.append(int(is_danger))
            warning_flags.append(int(is_warning))

            if ctrl_type == "learned":
                online.update(drift_val, step_loss, is_danger, is_warning)

            ctrl.maybe_merge()
            ctrl.maybe_prune()

            if done:
                obs = env.reset()
                if ctrl_type != "fixed":
                    ctrl.gating_reset()
            else:
                obs = o_next

        window = max(20, steps // 10)
        early_loss = float(np.mean(loss_history[:window]))
        late_loss = float(np.mean(loss_history[-window:]))
        convergence = early_loss / max(late_loss, 1e-8)

        results[mode] = {
            "early_loss": early_loss,
            "late_loss": late_loss,
            "convergence_ratio": convergence,
            "loss_volatility": float(np.std(loss_history)),
            "loss_mean": float(np.mean(loss_history)),
            "danger_rate": float(np.mean(danger_flags[-200:])),
            "loss_history": [float(l) for l in loss_history],
            "eta_history": [float(e) for e in eta_history],
        }

        if ctrl_type == "learned":
            results[mode]["final_params"] = online.param_summary()
            results[mode]["theta_history"] = online.theta_history
            results[mode]["reward_history"] = [float(r) for r in online.reward_history]

        print(f"\n{mode:>15}: early={early_loss:.4f} late={late_loss:.4f} "
              f"conv={convergence:.2f}x vol={np.std(loss_history):.4f} "
              f"danger={np.mean(danger_flags[-200:]):.3f}")

        if ctrl_type == "learned":
            ps = online.param_summary()
            print(f"  learned params: a={ps['a']:.3f} b={ps['b']:.3f} c={ps['c']:.3f}")
            print(f"  eta curve: d=0.01→{ps['eta_at_d01']:.3f} d=0.05→{ps['eta_at_d05']:.3f} "
                  f"d=0.10→{ps['eta_at_d10']:.3f} d=0.15→{ps['eta_at_d15']:.3f}")
            print(f"  danger_rate={ps['danger_rate']:.3f} avg_reward={ps['avg_reward']:.3f}")

    return results


if __name__ == "__main__":
    from env_drifting_double_well import DriftingDoubleWell

    print("=" * 60)
    print("A/B/C COMPARISON: baseline vs rule-based vs learned")
    print("=" * 60)

    all_results = {}
    for drift_test in [0.02, 0.05, 0.10]:
        print(f"\n{'─'*60}")
        print(f"  DRIFT = {drift_test}")
        print(f"{'─'*60}")
        res = run_comparison(DriftingDoubleWell, kappa=1.0, drift=drift_test, steps=500)
        all_results[drift_test] = res

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'drift':>8} {'baseline':>12} {'rule-based':>12} {'learned':>12} {'winner':>12}")
    print(f"{'':>8} {'conv':>6} {'vol':>6} {'conv':>6} {'vol':>6} {'conv':>6} {'vol':>6}")
    print("-" * 70)

    for drift_test in all_results:
        A = all_results[drift_test]["A_baseline"]
        B = all_results[drift_test]["B_rule_based"]
        C = all_results[drift_test]["C_learned"]

        best_conv = max((A["convergence_ratio"], "A"),
                         (B["convergence_ratio"], "B"),
                         (C["convergence_ratio"], "C"),
                         key=lambda x: x[0])
        best_vol = min((A["loss_volatility"], "A"),
                        (B["loss_volatility"], "B"),
                        (C["loss_volatility"], "C"),
                        key=lambda x: x[0])

        print(f"{drift_test:>8.3f} "
              f"{A['convergence_ratio']:>6.2f} {A['loss_volatility']:>6.4f} "
              f"{B['convergence_ratio']:>6.2f} {B['loss_volatility']:>6.4f} "
              f"{C['convergence_ratio']:>6.2f} {C['loss_volatility']:>6.4f} "
              f"conv={best_conv[1]} vol={best_vol[1]}")

    with open("learned_vs_rule_vs_baseline.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to learned_vs_rule_vs_baseline.json")
