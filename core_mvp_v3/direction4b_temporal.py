"""Direction 4b: Temporal Structure Analysis.

Cross-correlation, FFT power spectra, phase synchronization, and
before/after noise comparison for multi-agent action sequences.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def load_policy(tag=""):
    suffix = f"_{tag}" if tag else ""
    net = PolicyNetwork(hidden_dim=32)
    state_dict = torch.load(Path(f"results_final/phase0_policy_net{suffix}.pt"),
                            map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def rollout_actions(net_A, net_B, T=4000):
    schedule = [
        (1000, (0.1, 0.3)),
        (1000, (1.0, 2.0)),
        (1000, (0.1, 0.3)),
        (1000, (1.0, 2.0)),
    ]
    env = MultiAgentDriftingEnv(schedule=schedule, noise=0.05,
                                state_clip=5.0, force_scale=0.1,
                                action_scale=0.1, action_mix=0.5)
    env.reset()
    a_A_list, a_B_list, drift_list = [], [], []
    for _ in range(T):
        state = env.state.copy()
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            as_A, _, _ = net_A(x)
            as_B, _, _ = net_B(x)
        a_A = float(as_A.item())
        a_B = float(as_B.item())
        env.step(a_A, a_B)
        a_A_list.append(a_A)
        a_B_list.append(a_B)
        drift_list.append(env.current_drift)
    return np.array(a_A_list), np.array(a_B_list), np.array(drift_list)


def cross_corr(a, b, max_lag=100):
    lags = np.arange(-max_lag, max_lag + 1)
    corr = []
    n = len(a)
    for lag in lags:
        if lag < 0:
            corr.append(np.corrcoef(a[:lag], b[-lag:])[0, 1])
        elif lag > 0:
            corr.append(np.corrcoef(a[lag:], b[:-lag])[0, 1])
        else:
            corr.append(np.corrcoef(a, b)[0, 1])
    return lags, np.array(corr)


def power_spectrum(x):
    fft = np.fft.fft(x)
    power = np.abs(fft) ** 2
    return power[:len(power) // 2]


def main():
    print("=" * 60)
    print("  DIRECTION 4b: TEMPORAL STRUCTURE")
    print("=" * 60)

    print("\n--- loading agents ---")
    net_A = load_policy()
    net_B = load_policy("seed1")

    print("\n--- rolling out (clean) ---")
    a_A, a_B, drift = rollout_actions(net_A, net_B, T=4000)
    print(f"  {len(a_A)} steps, drift range [{drift.min():.1f}, {drift.max():.1f}]")

    print("\n--- STEP 1-2: cross-correlation ---")
    lags, cc = cross_corr(a_A, a_B, max_lag=100)
    peak_idx = np.argmax(cc)
    peak_corr = float(cc[peak_idx])
    peak_lag = int(lags[peak_idx])
    lag0_corr = float(cc[100])
    print(f"  lag=0 corr: {lag0_corr:.4f}")
    print(f"  peak corr:   {peak_corr:.4f}  at lag={peak_lag}")
    print(f"  asymmetry:   peak at lag {peak_lag:+d} → "
          f"{'B lags A' if peak_lag < 0 else 'A lags B' if peak_lag > 0 else 'synchronous'}")

    print("\n--- STEP 3: FFT power spectrum ---")
    ps_A = power_spectrum(a_A)
    ps_B = power_spectrum(a_B)
    ps_corr = float(np.corrcoef(ps_A, ps_B)[0, 1])
    print(f"  power spectrum corr: {ps_corr:.4f}")

    print("\n--- STEP 4: phase synchronization ---")
    phase_A = np.angle(np.fft.fft(a_A))
    phase_B = np.angle(np.fft.fft(a_B))
    phase_diff = float(np.mean(np.abs(phase_A - phase_B)))
    phase_circular_diff = float(np.mean(np.abs(np.angle(np.exp(1j * (phase_A - phase_B))))))
    print(f"  mean |Δphase|: {phase_diff:.2f} rad")
    print(f"  circular |Δphase|: {phase_circular_diff:.2f} rad")

    print("\n--- STEP 5: noise comparison ---")
    noise_lvls = [0.0, 0.02, 0.05, 0.1]
    noise_results = []
    for sigma in noise_lvls:
        a_B_noisy = a_B + np.random.normal(0, sigma, size=a_B.shape)
        lags_n, cc_n = cross_corr(a_A, a_B_noisy, max_lag=100)
        pk_n = float(cc_n[np.argmax(cc_n)])
        phase_An = np.angle(np.fft.fft(a_A))
        phase_Bn = np.angle(np.fft.fft(a_B_noisy))
        pd_n = float(np.mean(np.abs(np.angle(np.exp(1j * (phase_An - phase_Bn))))))
        noise_results.append({
            "sigma": sigma,
            "lag0_corr": float(cc_n[100]),
            "peak_corr": pk_n,
            "phase_diff": pd_n,
        })
        print(f"  σ={sigma:.2f}: lag0_corr={cc_n[100]:.4f} "
              f"peak={pk_n:.4f} phase_diff={pd_n:.2f}")

    results = {
        "cross_correlation": {
            "lag0_corr": lag0_corr,
            "peak_corr": peak_corr,
            "peak_lag": peak_lag,
            "lags": lags.tolist(),
            "corr_curve": cc.tolist(),
        },
        "frequency": {
            "power_spectrum_corr": ps_corr,
        },
        "phase": {
            "mean_abs_diff": phase_diff,
            "circular_diff": phase_circular_diff,
        },
        "noise_degradation": noise_results,
    }

    out_path = Path("results_final/direction4b_temporal.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
