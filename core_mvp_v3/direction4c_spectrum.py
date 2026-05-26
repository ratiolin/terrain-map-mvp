"""Direction 4c-spectrum: Anti-Correlation Spectral Structure.

Loads co-trained agents and analyzes the spectral signature of
anti-correlated action patterns: cross-correlation lag structure,
phase difference, and anti-phase type classification.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_mvp_v3.env import MultiAgentDriftingEnv
from core_mvp_v3.models import PolicyNetwork


def cross_corr(a, b, max_lag=100):
    lags = np.arange(-max_lag, max_lag + 1)
    corr = []
    n = len(a)
    for lag in lags:
        if lag < 0:
            corr.append(float(np.corrcoef(a[:lag], b[-lag:])[0, 1]))
        elif lag > 0:
            corr.append(float(np.corrcoef(a[lag:], b[:-lag])[0, 1]))
        else:
            corr.append(float(np.corrcoef(a, b)[0, 1]))
    return lags, np.array(corr)


def load_cotrained(tag):
    net = PolicyNetwork(hidden_dim=32)
    path = Path(f"results_final/direction4c_policy_{tag}.pt")
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def main():
    print("=" * 60)
    print("  DIRECTION 4c-SPECTRUM: ANTI-CORRELATION STRUCTURE")
    print("=" * 60)

    net_A = load_cotrained("A")
    net_B = load_cotrained("B")

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

    a_A, a_B = [], []
    for _ in range(4000):
        state = env.state.copy()
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            as_A, _, _ = net_A(x)
            as_B, _, _ = net_B(x)
        aA = float(as_A.item())
        aB = float(as_B.item())
        env.step(aA, aB)
        a_A.append(aA)
        a_B.append(aB)

    aA = np.array(a_A)
    aB = np.array(a_B)

    print(f"\n  action stats: A [{aA.min():.3f}, {aA.max():.3f}] μ={aA.mean():.4f}")
    print(f"                 B [{aB.min():.3f}, {aB.max():.3f}] μ={aB.mean():.4f}")

    print("\n--- cross-correlation ---")
    lags, cc = cross_corr(aA, aB, max_lag=50)

    idx_max = np.argmax(cc)
    idx_min = np.argmin(cc)
    lag_pos = int(lags[idx_max])
    lag_neg = int(lags[idx_min])
    peak_pos = float(cc[idx_max])
    peak_neg = float(cc[idx_min])

    print(f"  lag=0 corr:   {float(cc[50]):+.4f}")
    print(f"  max (+):      {peak_pos:+.4f}  at lag={lag_pos}")
    print(f"  min (-):      {peak_neg:+.4f}  at lag={lag_neg}")

    if lag_neg == 0 and peak_neg < -0.9:
        mode = "instant anti-phase"
    elif abs(lag_neg) > 0 and peak_neg < -0.5:
        mode = f"delayed anti-phase (lag={lag_neg:+d})"
    else:
        mode = "weak / no anti-phase"

    print(f"  type: {mode}")

    print("\n--- phase analysis ---")
    fft_A = np.fft.fft(aA)
    fft_B = np.fft.fft(aB)
    phase_diff = float(np.mean(np.abs(np.angle(np.exp(
        1j * (np.angle(fft_A) - np.angle(fft_B)))))))
    print(f"  mean circular |Δphase|: {phase_diff:.3f} rad")
    near_pi = np.abs(phase_diff - np.pi) < 0.5
    print(f"  near π? {'YES (anti-phase)' if near_pi else 'no'}")

    ps_A = np.abs(fft_A)**2
    ps_B = np.abs(fft_B)**2
    ps_corr = float(np.corrcoef(ps_A[:len(ps_A)//2], ps_B[:len(ps_B)//2])[0, 1])
    print(f"  power spectrum corr: {ps_corr:.4f}")

    results = {
        "cross_correlation": {
            "lag0": float(cc[50]),
            "max_corr": peak_pos, "max_lag": lag_pos,
            "min_corr": peak_neg, "min_lag": lag_neg,
            "lag_type": mode,
        },
        "phase": {
            "circular_diff": float(phase_diff),
            "near_pi": bool(near_pi),
            "power_spectrum_corr": float(ps_corr),
        },
    }

    out_path = Path("results_final/direction4c_spectrum.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
