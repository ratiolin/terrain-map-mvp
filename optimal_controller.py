"""
Optimal Control System — learned η(d) with dual-model architecture.

Architecture:
  Controller:  η(d) = f_θ(d)     ← MLP(1→16→1) with uncertainty
  Env model:   η̂_max(d) = g_φ(d) ← MLP(1→8→1) learned online

Dual-objective:
  objective = -loss + λ₁·boundary_reward - λ₂·danger
  boundary_reward = -|η - η̂_max|

Band-wise: train independent η(d) per discovered band.
Active exploration: push η↑ in SAFE, pull η↓ near boundaries.
"""
import json
import numpy as np
from collections import defaultdict, deque


class MLP:
    """Tiny MLP for 1D → 1D mapping. No torch dependency."""
    def __init__(self, hidden=16):
        # He init
        self.W1 = np.random.randn(hidden, 1) * np.sqrt(2.0 / 1)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(1, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(1)
        self.W1_var = np.random.randn(hidden, 1) * 0.01  # uncertainty head
        self.b1_var = np.zeros(hidden)
        self.W2_var = np.random.randn(1, hidden) * 0.01
        self.b2_var = np.zeros(1)

    def forward(self, x):
        h = np.tanh(self.W1 @ np.array([x]) + self.b1).flatten()
        mean = float((self.W2 @ h + self.b2)[0])
        var_h = np.tanh(self.W1_var @ np.array([x]) + self.b1_var).flatten()
        logvar = float((self.W2_var @ var_h + self.b2_var)[0])
        std = np.exp(0.5 * np.clip(logvar, -5, 2))
        return mean, max(std, 0.01)

    def grad(self, x):
        """Returns d(mean)/d(x) for gradient computation."""
        arr_x = np.array([x])
        h_pre = self.W1 @ arr_x + self.b1
        h = np.tanh(h_pre).flatten()
        dh = (1 - h**2)
        grad_h = dh.reshape(-1, 1) * self.W1  # (hidden, 1)
        grad_out = self.W2 @ grad_h          # (1, 1)
        return float(grad_out[0, 0])

    def update(self, x, target, lr=0.01):
        """Single SGD step on MSE."""
        pred, _ = self.forward(x)
        err = target - pred

        arr_x = np.array([x])
        h_pre = self.W1 @ arr_x + self.b1
        h = np.tanh(h_pre).flatten()
        dh = (1 - h**2)

        dW2 = err * h.reshape(1, -1)
        db2 = err * np.ones(1)
        dW1 = err * (self.W2.T * dh.reshape(-1, 1)) @ arr_x.reshape(1, -1)
        db1 = err * (self.W2.flatten() * dh)

        self.W2 += lr * dW2
        self.b2 += lr * db2
        self.W1 += lr * dW1
        self.b1 += lr * db1


class BandDecomposer:
    """Discover and maintain independent η(d) per drift band."""
    def __init__(self, min_band_width=0.02, merge_distance=0.03):
        self.bands = []  # list of (start, end, ctrl_mlp, env_mlp)
        self.min_width = min_band_width
        self.merge_distance = merge_distance

    def find_or_create_band(self, drift):
        # Check existing bands with overlap tolerance
        for start, end, ctrl_mlp, env_mlp in self.bands:
            if start - self.merge_distance <= drift <= end + self.merge_distance:
                return start, end, ctrl_mlp, env_mlp
        # Create new band
        hw = self.min_width / 2
        start = drift - hw
        end = drift + hw
        # Merge with nearby bands
        merged = False
        for i, (s, e, cm, em) in enumerate(self.bands):
            if abs(s - start) < self.merge_distance or abs(e - end) < self.merge_distance:
                # Extend existing band
                self.bands[i] = (min(s, start), max(e, end), cm, em)
                merged = True
                break
        if not merged:
            ctrl = MLP(hidden=16)
            ctrl.b2 = np.array([0.3])  # initialize to ~0.3
            env = MLP(hidden=8)
            env.b2 = np.array([0.5])   # initialize η̂_max ~0.5
            self.bands.append((start, end, ctrl, env))
        return self.find_or_create_band(drift)

    def get_all_bands(self):
        return [(s, e) for s, e, _, _ in self.bands]


class OptimalEtaController:
    """
    Dual-model optimal controller:
      f_θ(d) → η(d)  with uncertainty: η = μ(d) + β·σ(d)
      g_φ(d) → η̂_max(d)  learned online from stability feedback
    """

    def __init__(self, lr_ctrl=0.02, lr_env=0.01,
                 lambda_boundary=0.3, lambda_danger=0.5,
                 explore_push=0.05, explore_pull=0.1,
                 beta_uncertainty=0.3):
        self.bands = BandDecomposer(min_band_width=0.02)

        self.lr_ctrl = lr_ctrl
        self.lr_env = lr_env
        self.lambda_boundary = lambda_boundary
        self.lambda_danger = lambda_danger
        self.explore_push = explore_push
        self.explore_pull = explore_pull
        self.beta_uncertainty = beta_uncertainty

        # History
        self.eta_history = []
        self.eta_max_history = []
        self.loss_history = []
        self.boundary_reward_history = []
        self.objective_history = []
        self.state_history = []

        # Danger tracking
        self._loss_buffer = deque(maxlen=100)
        self._consecutive_safe = 0
        self._consecutive_danger = 0

    def _detect_state(self, loss):
        """Adaptive danger detection based on loss statistics."""
        self._loss_buffer.append(loss)
        if len(self._loss_buffer) < 20:
            return "SAFE"

        mu = np.mean(self._loss_buffer)
        sigma = np.std(self._loss_buffer) + 1e-6
        z = (loss - mu) / sigma

        if z > 3.0:
            self._consecutive_danger += 1
            self._consecutive_safe = 0
            return "DANGER"
        elif z > 1.5:
            self._consecutive_danger += 1
            self._consecutive_safe = 0
            return "WARNING"
        else:
            self._consecutive_safe += 1
            self._consecutive_danger = max(0, self._consecutive_danger - 1)
            return "SAFE"

    def forward(self, drift, loss=None):
        """
        Compute η(d) with:
          1. MLP mean + uncertainty → η = μ + β·σ
          2. Clamped to [0.01, 1.0]
        """
        _, _, ctrl_mlp, env_mlp = self.bands.find_or_create_band(drift)

        mu, sigma = ctrl_mlp.forward(drift)
        eta = float(np.clip(mu + self.beta_uncertainty * sigma, 0.01, 1.0))

        # Env model prediction (for logging / boundary reward)
        eta_max_hat_raw, _ = env_mlp.forward(drift)
        eta_max_hat = float(np.clip(eta_max_hat_raw, 0.01, 1.0))

        # Track state for exploration in update()
        state = self._detect_state(loss) if loss is not None else "SAFE"
        self.state_history.append(state)

        self.eta_history.append(eta)
        self.eta_max_history.append(eta_max_hat)
        return eta

    def update(self, drift, loss):
        """
        Dual-model update with active boundary exploration.
        """
        state = self._detect_state(loss)
        _, _, ctrl_mlp, env_mlp = self.bands.find_or_create_band(drift)

        eta, _ = ctrl_mlp.forward(drift)
        eta = float(np.clip(eta, 0.01, 1.0))
        eta_max_hat_raw, _ = env_mlp.forward(drift)
        eta_max_hat = float(np.clip(eta_max_hat_raw, 0.01, 1.0))

        # ── Active exploration ──
        if state == "SAFE":
            eta_explore = eta * (1.0 + self.explore_push)
        elif state == "WARNING":
            eta_explore = eta * (1.0 - self.explore_pull)
        else:
            eta_explore = eta * 0.5

        # ── Boundary reward (tightens as env model converges) ──
        boundary_reward = -abs(eta_explore - eta_max_hat)
        self.boundary_reward_history.append(float(boundary_reward))

        # ── Danger penalty ──
        is_danger = 1.0 if state == "DANGER" else 0.0

        # ── Total objective ──
        objective = -loss + self.lambda_boundary * boundary_reward - self.lambda_danger * is_danger
        self.objective_history.append(float(objective))
        self.loss_history.append(float(loss))

        # ── Update controller MLP (f_θ) ──
        baseline = np.mean(self.objective_history[-50:]) if len(self.objective_history) >= 50 else 0.0
        advantage = objective - baseline
        grad_eta = ctrl_mlp.grad(drift)
        step_size = self.lr_ctrl * advantage * grad_eta * 0.05
        ctrl_mlp.update(drift, eta + step_size, lr=self.lr_ctrl)

        # ── Update environment model (g_φ) ──
        # Only update in SAFE (eta ≈ η_max when stable)
        if state == "SAFE" and self._consecutive_safe > 5:
            env_mlp.update(drift, eta, lr=self.lr_env * 0.5)

        # ── Track safety streak ──
        if state == "SAFE":
            self._consecutive_safe += 1
            self._consecutive_danger = 0
        elif state == "DANGER":
            self._consecutive_danger += 1
            self._consecutive_safe = 0

    def param_summary(self):
        return {
            "n_bands": len(self.bands.bands),
            "bands": [(float(s), float(e)) for s, e, _, _ in self.bands.bands],
            "avg_eta": float(np.mean(self.eta_history[-100:])) if self.eta_history else 0,
            "avg_eta_max_hat": float(np.mean(self.eta_max_history[-100:])) if self.eta_max_history else 0,
            "avg_boundary_reward": float(np.mean(self.boundary_reward_history[-100:])) if self.boundary_reward_history else 0,
            "avg_objective": float(np.mean(self.objective_history[-100:])) if self.objective_history else 0,
            "state_dist": {
                "SAFE": self.state_history[-200:].count("SAFE") if self.state_history else 0,
                "WARNING": self.state_history[-200:].count("WARNING") if self.state_history else 0,
                "DANGER": self.state_history[-200:].count("DANGER") if self.state_history else 0,
            },
        }


# ── A/B/C comparison with optimal controller ─────────────────────────
def run_full_comparison(env_class, kappa=1.0, drift=0.05, steps=500):
    import torch, random
    import torch.nn.functional as F
    from experiment10 import reset_seed, _make_agent, _make_ctrl
    from online_controller import OnlineStabilityController
    from learned_controller import LearnedEtaController as LearnedSigmoidController

    configs = [
        ("A_baseline", "fixed"),
        ("B_rule_based", "rule"),
        ("C_sigmoid", "sigmoid"),
        ("D_optimal", "optimal"),
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
        elif ctrl_type == "sigmoid":
            online = LearnedSigmoidController(lr=0.02, lambda_danger=0.5, lambda_warning=0.1)
        elif ctrl_type == "optimal":
            online = OptimalEtaController(lr_ctrl=0.02, lr_env=0.01,
                                          lambda_boundary=0.3, lambda_danger=0.5)
        else:
            online = None

        loss_history, eta_history = [], []
        obs = env.reset()
        if ctrl_type != "fixed":
            ctrl.gating_reset()

        for t in range(steps):
            drift_val = float(getattr(env, 'drift', drift))

            if ctrl_type == "rule":
                new_eta, state = online.step(drift_val)
                eta_history.append(new_eta)
            elif ctrl_type == "sigmoid":
                new_eta = online.forward(drift_val)
                eta_history.append(new_eta)
            elif ctrl_type == "optimal":
                new_eta = online.forward(drift_val, loss=loss_history[-1] if loss_history else None)
                eta_history.append(new_eta)
            else:
                eta_history.append(0.0)
                new_eta = 0.0

            if ctrl_type != "fixed" and hasattr(ctrl.gating, 'inertia'):
                ctrl.gating.inertia = float(new_eta)
                ctrl.inertia = float(new_eta)

            a = agent.act(obs)
            o_next, _, done = env.step(a)
            target = torch.tensor(o_next, dtype=torch.float32)
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            if ctrl.n_models() > 0:
                with torch.no_grad():
                    z_soft, z_logits, _ = ctrl.gating(s)
                K_cur = z_soft.size(-1)
                z_hard = F.one_hot(z_logits.argmax(dim=-1), K_cur).float()
                weights = (z_hard - z_soft.detach() + z_soft).squeeze(0)
                preds = [m.predict(obs, a) for m in ctrl.models]
                soft_pred = sum(weights[i] * preds[i] for i in range(K_cur))
                loss_t = ((soft_pred - target) ** 2).mean()
                step_loss = float(loss_t.item())

                ctrl.gating_optimizer.zero_grad()
                for m in ctrl.models:
                    m.optimizer.zero_grad()
                loss_t.backward()
                ctrl.gating_optimizer.step()
                for m in ctrl.models:
                    m.optimizer.step()
            else:
                step_loss = 1.0

            loss_history.append(step_loss)

            if ctrl_type == "sigmoid":
                is_danger = step_loss > 3.0 * np.mean(loss_history[-50:]) if len(loss_history) >= 50 else False
                online.update(drift_val, step_loss, is_danger, step_loss > 1.5 * np.mean(loss_history[-50:]) if len(loss_history) >= 50 else False)
            elif ctrl_type == "optimal":
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
            "loss_history": [float(l) for l in loss_history],
            "eta_history": [float(e) for e in eta_history],
        }

        if ctrl_type == "optimal":
            results[mode]["final_params"] = online.param_summary()

        print(f"{mode:>15}: conv={conv:.2f}x  vol={np.std(loss_history):.4f}  "
              f"eta_avg={np.mean(eta_history):.3f}")

        if ctrl_type == "optimal":
            ps = online.param_summary()
            print(f"  bands={ps['n_bands']}  η̂_max_avg={ps['avg_eta_max_hat']:.3f}  "
                  f"boundary_r={ps['avg_boundary_reward']:.3f}  obj={ps['avg_objective']:.3f}")

    return results


if __name__ == "__main__":
    from env_drifting_double_well import DriftingDoubleWell

    print("=" * 60)
    print("A/B/C/D: baseline vs rule vs sigmoid vs OPTIMAL")
    print("=" * 60)

    all_results = {}
    for d_test in [0.02, 0.05, 0.10]:
        print(f"\n{'─'*50}")
        print(f"  DRIFT = {d_test}")
        print(f"{'─'*50}")
        res = run_full_comparison(DriftingDoubleWell, kappa=1.0, drift=d_test, steps=500)
        all_results[d_test] = res

    print(f"\n{'='*65}")
    print("FINAL COMPARISON")
    print(f"{'='*65}")
    hdr = f"{'drift':>6} {'A(base)':>10} {'B(rule)':>10} {'C(sigm)':>10} {'D(opt)':>10}  winner"
    print(hdr)
    print(f"{'':>6} {'conv':>5} {'vol':>5} {'conv':>5} {'vol':>5} {'conv':>5} {'vol':>5} {'conv':>5} {'vol':>5}")
    print("-" * 65)

    for d_test in all_results:
        A = all_results[d_test]["A_baseline"]
        B = all_results[d_test]["B_rule_based"]
        C = all_results[d_test]["C_sigmoid"]
        D = all_results[d_test]["D_optimal"]

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

    with open("optimal_controller_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to optimal_controller_results.json")
