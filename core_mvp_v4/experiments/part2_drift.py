import json
import numpy as np
from core_mvp_v4.env import ContinuousDriftEnv
from core_mvp_v4.models import V4Model, train_model, compute_jacobian
from core_mvp_v4.metrics import (
    compute_k80, effective_rank, R2_probe, alignment, analyze_jacobian,
)


def run_part2_continuous_drift(n_seeds=8, d=4, k=2, hidden_dim=32,
                               n_episodes=5, episode_length=2000,
                               A=1.0, T=2000, sigma_obs=0.1):
    """Part 2: Continuous drift.

    Environment with sinusoidal drift: g = A * sin(2*pi*t / T).
    Measures how the controllability subspace tracks the drifting parameter.

    Windows: W1 = T/4, W2 = T/2 for panic signal estimation.
    """
    results = {
        "config": {"d": d, "k": k, "A": A, "T": T, "sigma_obs": sigma_obs},
        "seeds": [],
    }

    for seed in range(n_seeds):
        env = ContinuousDriftEnv(d=d, k=k, A=A, T=T, seed=seed)
        model = V4Model(state_dim=d, hidden_dim=hidden_dim, action_dim=k)

        train_model(model, env, num_episodes=n_episodes,
                    episode_length=episode_length, seed=seed)

        W1 = T // 4
        W2 = T // 2

        time_points = []
        drift_values = []
        k80_values = []
        R2_values = []
        angle_change_values = []
        hysteresis_data = []

        env.reset()
        prev_V = None
        hidden_buffer = []
        risk_buffer = []

        for t_step in range(episode_length * 2):
            s = env.get_state()
            g = env.get_current_drift()
            drift_values.append(g)

            h = model.f_numpy(s)
            hidden_buffer.append(h)
            a = model.act_numpy(s)
            ns, risk, _, _ = env.step(a)
            risk_buffer.append(risk)

            time_points.append(t_step)

            if len(hidden_buffer) >= 50:
                hidden_buffer = hidden_buffer[-50:]
                risk_buffer = risk_buffer[-50:]

                if t_step % 20 == 0:
                    J = compute_jacobian(model, s)
                    _, S, Vt = np.linalg.svd(J, full_matrices=False)
                    V = Vt.T
                    k80 = compute_k80(S)
                    k80_values.append(k80)

                    if prev_V is not None:
                        k_align = min(k, V.shape[1], prev_V.shape[1])
                        angle_change = 1.0 - alignment(V, prev_V, k=k_align)
                        angle_change_values.append(angle_change)
                    prev_V = V.copy()

            if len(risk_buffer) >= W1 and t_step % W1 == 0:
                recent_h = np.array(hidden_buffer[-W1:])
                recent_r = np.array(risk_buffer[-W1:]).reshape(-1, 1)
                try:
                    r2_short = R2_probe(recent_h, recent_r)
                    R2_values.append(r2_short)
                except Exception:
                    pass

            if t_step > W2 and t_step % 50 == 0:
                hist_drift = drift_values[-W2:]
                hist_k80 = k80_values[-W2:] if k80_values else []
                if hist_k80:
                    hysteresis_data.append({
                        "t": t_step,
                        "g": float(g),
                        "k80": float(np.mean(hist_k80)),
                    })

        summary = {
            "seed": seed,
            "drift_range": [float(np.min(drift_values)), float(np.max(drift_values))],
            "k80_mean": float(np.mean(k80_values)) if k80_values else 0.0,
            "k80_std": float(np.std(k80_values)) if k80_values else 0.0,
            "R2_mean": float(np.mean(R2_values)) if R2_values else 0.0,
            "angle_change_mean": float(np.mean(angle_change_values)) if angle_change_values else 0.0,
            "n_hysteresis_samples": len(hysteresis_data),
        }

        if hysteresis_data:
            g_vals = [h["g"] for h in hysteresis_data]
            k80_vals = [h["k80"] for h in hysteresis_data]
            from scipy.stats import pearsonr
            if len(g_vals) > 2:
                corr, pval = pearsonr(g_vals, k80_vals)
                summary["R2_g"] = float(corr ** 2)
                summary["correlation_significance"] = float(pval)

        results["seeds"].append(summary)

    agg = {
        "k80_mean": f"{float(np.mean([s['k80_mean'] for s in results['seeds']])):.4f}",
        "k80_std": f"{float(np.std([s['k80_mean'] for s in results['seeds']])):.4f}",
        "R2_g": f"{float(np.mean([s.get('R2_g', 0) for s in results['seeds']])):.4f}",
        "angle_change_mean": f"{float(np.mean([s['angle_change_mean'] for s in results['seeds']])):.4f}"
        if any(s.get('angle_change_mean') for s in results['seeds'])
        else "N/A",
    }
    results["aggregate"] = agg

    return results


if __name__ == "__main__":
    import os
    os.makedirs("core_mvp_v4/results", exist_ok=True)
    r = run_part2_continuous_drift(n_seeds=8)
    with open("core_mvp_v4/results/part2_continuous_drift.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("Part 2 complete.")
    agg = r.get("aggregate", {})
    print(f"  k80={agg.get('k80_mean')}±{agg.get('k80_std')}, "
          f"R2(g)={agg.get('R2_g')}, angle_change={agg.get('angle_change_mean')}")
