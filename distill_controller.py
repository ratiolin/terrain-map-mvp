"""
Hybrid controller: FSM safety overrides + f_safe(drift) for continuous control.

  state ∈ {DANGER, WARNING, OOD, SAFE}
  DANGER  → η = 0.3
  WARNING → η = 0.4
  OOD     → η = fallback(drift)
  SAFE    → η = f_safe(drift)  [piecewise linear, monotonic, smoothed]
"""
import json, random, numpy as np, torch, torch.nn.functional as F
from collections import deque
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from env_drifting_double_well import DriftingDoubleWell
from generalizable_v2 import FinalGeneralizableController
from experiment10 import _make_agent, _make_ctrl, reset_seed


# ═════════════════════════════════════════════════════════════════════
# 1. Record (drift, state, η) — only drift as feature
# ═════════════════════════════════════════════════════════════════════

def record_drift_state_eta(drift_list, seeds, steps_per_run=1200):
    data = []
    for drift_rate in drift_list:
        for seed in seeds:
            reset_seed(seed)
            env = DriftingDoubleWell(kappa=1.0, drift_rate=drift_rate,
                                     flip_mode="deterministic", add_context=False)
            agent = _make_agent(env)
            ctrl = _make_ctrl(agent, 4, inertia=0.0)
            online = FinalGeneralizableController(lr=0.0, gamma=0.3)
            loss_hist, obs = [], env.reset()
            ctrl.gating_reset()
            prev_margin = 0.0
            for _ in range(steps_per_run):
                drift_val = float(getattr(env, "drift", drift_rate))
                eta_out = online.forward(drift_val, loss=loss_hist[-1] if loss_hist else None, margin=prev_margin)
                state = online.state_history[-1] if online.state_history else "SAFE"
                is_ood = online._is_ood(drift_val)
                if online._probe_active:
                    label = "PROBE"
                elif is_ood:
                    label = "OOD"
                else:
                    label = state
                data.append({"drift": drift_val, "state": label, "eta": float(eta_out)})
                if hasattr(ctrl.gating, "inertia"): ctrl.gating.inertia = float(eta_out)
                ctrl.inertia = float(eta_out)
                a = agent.act(obs)
                o_next, _, done = env.step(a)
                target_t = torch.tensor(o_next, dtype=torch.float32)
                if ctrl.n_models() > 0:
                    weights = ctrl.gating_weights(obs)
                    K_cur = len(weights)
                    preds = [m.predict(obs, a) for m in ctrl.models]
                    soft_pred = sum(weights[i] * preds[i] for i in range(K_cur))
                    loss_val = ((soft_pred - target_t)**2).mean()
                    step_loss = float(loss_val.item())
                    errs = [float(np.mean(np.abs(p.detach().numpy()-o_next))) for p in preds]
                    prev_margin = float(np.sort(errs)[1]-np.sort(errs)[0]) if len(errs)>=2 else 0.0
                    ctrl.gating_optimizer.zero_grad()
                    for m in ctrl.models: m.optimizer.zero_grad()
                    loss_val.backward(); ctrl.gating_optimizer.step()
                    for m in ctrl.models: m.optimizer.step()
                else: step_loss = 1.0
                loss_hist.append(step_loss)
                ctrl.maybe_merge(); ctrl.maybe_prune()
                obs = env.reset() if done else o_next
                if done: ctrl.gating_reset()
    return data


# ═════════════════════════════════════════════════════════════════════
# 2. Piecewise linear monotonic fit on SAFE data
# ═════════════════════════════════════════════════════════════════════

def fit_piecewise_linear(drifts, etas, n_segments=4):
    """Fit n_segments piecewise linear segments with monotonic trend detection."""
    drifts = np.array(drifts); etas = np.array(etas)
    idx = np.argsort(drifts)
    drifts, etas = drifts[idx], etas[idx]

    # Determine monotonic direction
    corr = np.corrcoef(drifts, etas)[0, 1]
    decreasing = corr < 0

    # Equal-frequency binning
    n = len(drifts)
    boundaries = [drifts[0]]
    for i in range(1, n_segments):
        boundaries.append(float(drifts[int(i * n / n_segments)]))
    boundaries.append(drifts[-1] + 1e-8)

    segments = []
    for i in range(n_segments):
        mask = (drifts >= boundaries[i]) & (drifts < boundaries[i + 1])
        if mask.sum() < 5:
            continue
        d_seg = drifts[mask]; e_seg = etas[mask]
        if len(np.unique(d_seg)) < 2:
            k, b = 0.0, float(np.mean(e_seg))
        else:
            X = np.c_[d_seg, np.ones(len(d_seg))]
            k, b = np.linalg.lstsq(X, e_seg, rcond=None)[0]
        # Monotonic enforcement
        if decreasing and k > 0: k = 0.0
        if not decreasing and k < 0: k = 0.0
        segments.append({"d_start": float(boundaries[i]), "d_end": float(boundaries[i+1]),
                         "k": float(k), "b": float(b), "n": int(mask.sum()),
                         "mean_eta": float(np.mean(e_seg))})

    # Merge adjacent segments with similar slopes
    merged = []
    for s in segments:
        if not merged:
            merged.append(s)
            continue
        prev = merged[-1]
        if abs(s["k"] - prev["k"]) < 0.05:
            prev["d_end"] = s["d_end"]
            prev["n"] += s["n"]
        else:
            merged.append(s)

    return merged, decreasing


def predict_pwl(drift, segments):
    """Evaluate piecewise linear function."""
    for s in segments:
        if s["d_start"] <= drift < s["d_end"]:
            return float(np.clip(s["k"] * drift + s["b"], 0.01, 1.0))
    # Fallback: nearest segment
    return float(np.clip(segments[-1]["k"] * drift + segments[-1]["b"], 0.01, 1.0))


def fallback_eta(drift):
    """OOD rule fallback."""
    if drift < 0.02: return 0.4
    if drift > 0.15: return 0.3
    return 0.5


# ═════════════════════════════════════════════════════════════════════
# 3. Hybrid controller: FSM safety + f_safe(drift) + smoothing
# ═════════════════════════════════════════════════════════════════════

class HybridController:
    def __init__(self, safe_segments, decreasing, gamma=0.2, lip=0.5):
        self.segments = safe_segments
        self.decreasing = decreasing
        self.gamma = gamma
        self.lip = lip  # Lipschitz bound

        self._loss_buffer = deque(maxlen=100)
        self._drift_buffer = deque(maxlen=200)
        self._prev_eta = 0.5
        self._prev_drift = 0.0

        self._danger_streak = 0
        self._warning_streak = 0
        self._prev_state = "SAFE"
        self._state_streak = 0

    def _detect_state(self, loss, drift):
        if len(self._loss_buffer) < 20:
            return "SAFE"
        mu = float(np.mean(self._loss_buffer))
        sigma = float(np.std(self._loss_buffer)) + 1e-6
        z = (loss - mu) / sigma

        # OOD detection
        if len(self._drift_buffer) >= 20:
            d_mu = float(np.mean(self._drift_buffer))
            d_sigma = float(np.std(self._drift_buffer)) + 0.001
            if abs(drift - d_mu) / d_sigma > 3.0:
                return "OOD"

        if z > 2.5: return "DANGER"
        if z > 1.5: return "WARNING"
        return "SAFE"

    def step(self, drift, loss):
        self._drift_buffer.append(drift)
        self._loss_buffer.append(loss)

        raw = self._detect_state(loss, drift)

        if raw == self._prev_state:
            self._state_streak += 1
        else:
            self._state_streak = 0
            self._prev_state = raw

        state = raw if self._state_streak >= 3 else self._prev_state

        # Safety overrides
        if state == "DANGER":
            eta = 0.3
        elif state == "WARNING":
            eta = 0.4
        elif state == "OOD":
            eta = fallback_eta(drift)
        else:  # SAFE
            eta = predict_pwl(drift, self.segments)

        # Lipschitz constraint
        drift_delta = abs(drift - self._prev_drift)
        max_change = self.lip * drift_delta + 0.02
        if abs(eta - self._prev_eta) > max_change:
            eta = self._prev_eta + np.sign(eta - self._prev_eta) * max_change

        # Smoothing
        eta_smooth = (1.0 - self.gamma) * eta + self.gamma * self._prev_eta

        self._prev_eta = float(np.clip(eta_smooth, 0.01, 1.0))
        self._prev_drift = float(drift)
        return self._prev_eta


# ═════════════════════════════════════════════════════════════════════
# 4. Closed-loop test
# ═════════════════════════════════════════════════════════════════════

def run_hybrid_loop(env, ctrl, agent, steps, hybrid):
    loss_hist = []
    hybrid._loss_buffer.clear(); hybrid._drift_buffer.clear()
    hybrid._prev_eta = 0.5; hybrid._prev_drift = 0.0
    hybrid._prev_state = "SAFE"; hybrid._state_streak = 0
    obs = env.reset(); ctrl.gating_reset()
    prev_margin = 0.0
    for _ in range(steps):
        drift_val = float(getattr(env, "drift", 0.02))
        cur_loss = loss_hist[-1] if loss_hist else 1.0
        new_eta = hybrid.step(drift_val, cur_loss)
        if hasattr(ctrl.gating, "inertia"): ctrl.gating.inertia = float(new_eta)
        ctrl.inertia = float(new_eta)
        a = agent.act(obs); o_next, _, done = env.step(a)
        target_t = torch.tensor(o_next, dtype=torch.float32)
        if ctrl.n_models() > 0:
            weights = ctrl.gating_weights(obs)
            K_cur = len(weights)
            preds = [m.predict(obs, a) for m in ctrl.models]
            soft_pred = sum(weights[i]*preds[i] for i in range(K_cur))
            loss_val = ((soft_pred - target_t)**2).mean(); step_loss = float(loss_val.item())
            errs = [float(np.mean(np.abs(p.detach().numpy()-o_next))) for p in preds]
            prev_margin = float(np.sort(errs)[1]-np.sort(errs)[0]) if len(errs)>=2 else 0.0
            ctrl.gating_optimizer.zero_grad()
            for m in ctrl.models: m.optimizer.zero_grad()
            loss_val.backward(); ctrl.gating_optimizer.step()
            for m in ctrl.models: m.optimizer.step()
        else: step_loss = 1.0
        loss_hist.append(step_loss)
        ctrl.maybe_merge(); ctrl.maybe_prune()
        obs = env.reset() if done else o_next
        if done: ctrl.gating_reset()
    return loss_hist


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 1. Collect data ──
    print("="*55); print("  1. Collecting (drift, state, η)")
    print("="*55)
    drift_list = [0.02, 0.05, 0.08, 0.11, 0.14, 0.17]
    seeds = [42, 43, 44, 45, 46]
    data = record_drift_state_eta(drift_list, seeds, steps_per_run=800)
    print(f"  {len(data)} samples")

    # State distribution
    from collections import Counter
    sc = Counter(d["state"] for d in data)
    for s, c in sc.most_common(): print(f"    {s}: {c} ({100*c/len(data):.1f}%)")

    # ── 2. Fit f_safe(drift) ──
    print(f"\n{'='*55}"); print(f"  2. Fitting f_safe(drift) on SAFE subset")
    print(f"{'='*55}")
    safe = [(d["drift"], d["eta"]) for d in data if d["state"] == "SAFE"]
    drifts_safe = np.array([s[0] for s in safe])
    etas_safe = np.array([s[1] for s in safe])
    print(f"  SAFE samples: {len(safe)}")

    best_mse = float("inf")
    best_segments = None
    best_dec = False
    for n_seg in [3, 4, 5, 6]:
        segs, dec = fit_piecewise_linear(drifts_safe, etas_safe, n_segments=n_seg)
        preds = np.array([predict_pwl(d, segs) for d in drifts_safe])
        mse = float(np.mean((etas_safe - preds)**2))
        r2 = 1 - mse / max(float(np.var(etas_safe)), 1e-12)
        print(f"    n_seg={n_seg}  MSE={mse:.4f}  R²={r2:.4f}  {'← best' if mse<best_mse else ''}")
        if mse < best_mse:
            best_mse = mse
            best_segments = segs
            best_dec = dec

    print(f"\n  Selected: {len(best_segments)} segments, {'decreasing' if best_dec else 'increasing'}")
    for s in best_segments:
        print(f"    [{s['d_start']:.3f}, {s['d_end']:.3f}]  "
              f"η = {s['k']:+.3f}·d {s['b']:+.3f}  (n={s['n']})")

    # ── 3. Build hybrid controller ──
    hybrid = HybridController(best_segments, best_dec, gamma=0.2, lip=0.5)

    # ── 4. Closed-loop test ──
    print(f"\n{'='*55}"); print(f"  4. Closed-loop Hybrid vs MLP")
    print(f"{'='*55}")
    test_drift = [0.02, 0.05, 0.08, 0.11, 0.14]
    test_steps = 600

    for d in test_drift:
        # Hybrid
        reset_seed(42)
        env = DriftingDoubleWell(kappa=1.0, drift_rate=d, flip_mode="deterministic", add_context=False)
        agent = _make_agent(env); ctrl = _make_ctrl(agent, 4, inertia=0.0)
        loss_h = run_hybrid_loop(env, ctrl, agent, test_steps, hybrid)
        tail = int(test_steps*0.3)
        mean_h = float(np.mean(loss_h[-tail:])) if tail>0 else float(np.mean(loss_h))

        # MLP baseline
        reset_seed(42)
        env2 = DriftingDoubleWell(kappa=1.0, drift_rate=d, flip_mode="deterministic", add_context=False)
        agent2 = _make_agent(env2); ctrl2 = _make_ctrl(agent2, 4, inertia=0.0)
        online = FinalGeneralizableController(lr=0.01, gamma=0.3)
        loss_mlp = []; obs2 = env2.reset(); ctrl2.gating_reset(); prev_m = 0.0
        for _ in range(test_steps):
            dv = float(getattr(env2, "drift", d))
            new_eta = online.forward(dv, loss=loss_mlp[-1] if loss_mlp else None, margin=prev_m)
            if hasattr(ctrl2.gating, "inertia"): ctrl2.gating.inertia = float(new_eta)
            ctrl2.inertia = float(new_eta)
            a = agent2.act(obs2); o_next, _, done = env2.step(a)
            target_t = torch.tensor(o_next, dtype=torch.float32)
            if ctrl2.n_models()>0:
                weights = ctrl2.gating_weights(obs2); K_cur = len(weights)
                preds = [m.predict(obs2, a) for m in ctrl2.models]
                soft_pred = sum(weights[i]*preds[i] for i in range(K_cur))
                loss_val = ((soft_pred-target_t)**2).mean(); step_loss = float(loss_val.item())
                errs = [float(np.mean(np.abs(p.detach().numpy()-o_next))) for p in preds]
                prev_m = float(np.sort(errs)[1]-np.sort(errs)[0]) if len(errs)>=2 else 0.0
                ctrl2.gating_optimizer.zero_grad()
                for m in ctrl2.models: m.optimizer.zero_grad()
                loss_val.backward(); ctrl2.gating_optimizer.step()
                for m in ctrl2.models: m.optimizer.step()
            else: step_loss = 1.0
            loss_mlp.append(step_loss); online.update(dv, step_loss)
            ctrl2.maybe_merge(); ctrl2.maybe_prune()
            obs2 = env2.reset() if done else o_next
            if done: ctrl2.gating_reset()
        mean_mlp = float(np.mean(loss_mlp[-tail:])) if tail>0 else float(np.mean(loss_mlp))

        delta = 100*(mean_h-mean_mlp)/max(abs(mean_mlp),1e-8)
        tag = "PASS ✓" if abs(delta)<0.5 else f"FAIL ({delta:+.2f}%)"
        print(f"  drift={d:.2f}  MLP={mean_mlp:.4f}  Hybrid={mean_h:.4f}  Δ={delta:+.2f}%  {tag}")

    # ── 5. Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: PWL fit overlay
    d_all = np.linspace(0, 3.5, 200)
    eta_pred = [predict_pwl(x, best_segments) for x in d_all]
    safe_data = [(d["drift"], d["eta"]) for d in data if d["state"] == "SAFE"]
    axes[0].scatter([s[0] for s in safe_data[::50]], [s[1] for s in safe_data[::50]],
                    s=1, alpha=0.3, color="gray")
    axes[0].plot(d_all, eta_pred, "r-", linewidth=2, label="f_safe(drift)")
    for s in best_segments:
        axes[0].axvline(s["d_start"], color="blue", linestyle="--", alpha=0.3)
    axes[0].set_xlabel("drift"); axes[0].set_ylabel("η")
    axes[0].set_title("Piecewise linear fit on SAFE subset")
    axes[0].legend()

    # Right: η vs drift for controller output
    drifts_data = np.array([d["drift"] for d in data])
    etas_data = np.array([d["eta"] for d in data])
    colors = {"SAFE": "green", "WARNING": "orange", "DANGER": "red", "OOD": "purple", "PROBE": "blue"}
    for state in ["SAFE", "WARNING", "DANGER", "OOD", "PROBE"]:
        mask = [d["state"]==state for d in data]
        if sum(mask) > 0:
            axes[1].scatter(drifts_data[mask][::30], etas_data[mask][::30],
                            s=2, alpha=0.4, color=colors.get(state,"gray"), label=state)
    axes[1].plot(d_all, eta_pred, "k-", linewidth=2)
    axes[1].axhline(0.3, color="red", linestyle=":", alpha=0.5, label="DANGER=0.3")
    axes[1].axhline(0.4, color="orange", linestyle=":", alpha=0.5, label="WARNING=0.4")
    axes[1].set_xlabel("drift"); axes[1].set_ylabel("η")
    axes[1].set_title("Controller output by state")
    axes[1].legend(fontsize=6, loc="upper right")
    plt.tight_layout(); plt.savefig("hybrid_controller.png", dpi=150); plt.close()
    print("\n  Plot → hybrid_controller.png")

    # ── 6. Output ──
    print(f"\n{'='*55}"); print(f"  6. Final hybrid control law")
    print(f"{'='*55}")
    print(f"  Safety overrides:")
    print(f"    DANGER  → η = 0.3")
    print(f"    WARNING → η = 0.4")
    print(f"    OOD     → η = fallback(drift)")
    print(f"  SAFE (piecewise linear, {len(best_segments)} segments):")
    for s in best_segments:
        print(f"    d ∈ [{s['d_start']:.3f}, {s['d_end']:.3f})  "
              f"η = {s['k']:+.4f}·d {s['b']:+.4f}")
    print(f"  Smoothing: γ = {hybrid.gamma}")
    print(f"  Lipschitz: L = {hybrid.lip}")
    print(f"  Hysteresis: 3 consecutive agreements")

    with open("hybrid_controller.json","w") as f:
        json.dump({"type":"hybrid_fsm_pwl","segments":best_segments,
                   "decreasing":best_dec,"gamma":hybrid.gamma,"lip":hybrid.lip,
                   "danger_eta":0.3,"warning_eta":0.4,"fallback_desc":"OOD rule"},f,indent=2)
    print("\n  → hybrid_controller.json, hybrid_controller.png")
