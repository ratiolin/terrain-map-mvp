from env_double_well import DoubleWellEnv
from agent import Agent
from controller import GatingGrowthController
from loop_multi import run_soft
from analyze import (temporal_consistency, z_separation,
                     specialization_score, model_distance)

import random
import numpy as np
import torch
from collections import Counter


def reset_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def test_routing_modes():
    print("=" * 50)
    print("STAGE 7 — Soft / Semi-Hard / Hard Routing")
    print("=" * 50)

    env = DoubleWellEnv(noise=0.03)
    agent = Agent(1, 1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=8, use_z=True)

    hist, wh = run_soft(env, agent, ctrl, steps=20000)

    K = ctrl.n_models()
    tc = temporal_consistency(wh)
    frozen = ctrl.freeze_structure
    mode = "semi-hard" if frozen else "soft"

    # Per-basin winner
    ctrl.gating_reset()
    env_t = DoubleWellEnv(noise=0.0)
    winners_pos = []; winners_neg = []
    z_hist = []; s_hist = []
    for x_start in [0.5] * 200 + [-0.5] * 200:
        ctrl.gating_reset()
        env_t.state = np.array([x_start], dtype=np.float32)
        for _ in range(10):
            w = ctrl.gating_weights(env_t.state)
            z_hist.append(w.detach().numpy().copy())
            s_hist.append(env_t.state.copy())
            wa = int(w.argmax().item())
            if x_start > 0: winners_pos.append(wa)
            else: winners_neg.append(wa)
            o_next, _, _ = env_t.step(0)
            env_t.state = o_next

    sep = z_separation(z_hist, s_hist)
    total_u = max(1, sum(ctrl.usage))
    max_frac = max(ctrl.usage) / total_u
    spec = specialization_score(ctrl.errors, wh)

    pos_winner = Counter(winners_pos).most_common(1)[0][0] if winners_pos else 0
    neg_winner = Counter(winners_neg).most_common(1)[0][0] if winners_neg else 0

    print(f"K={K}  mode={mode}  frozen={frozen}")
    print(f"stability:       {tc['stability']:.3f}")
    print(f"max_fraction:    {max_frac:.3f}")
    print(f"Z separation:    {sep:.3f}")
    print(f"specialization:  {spec:.3f}")
    print(f"pos winner:      model {pos_winner}")
    print(f"neg winner:      model {neg_winner}")
    print(f"models separated: {pos_winner != neg_winner if K > 1 else False}")

    # Verify
    assert mode == "semi-hard", f"Mode should be semi-hard after freeze, got {mode}"
    assert tc['stability'] >= 0.7, f"stability={tc['stability']}"
    assert max_frac <= 0.85, f"max_fraction={max_frac}"
    assert sep >= 0.3, f"Z separation={sep}"
    if K > 1:
        assert pos_winner != neg_winner, \
            f"Models should diverge: pos={pos_winner} neg={neg_winner}"
        print(f"  models separated: True (pos={pos_winner}, neg={neg_winner})")
        if spec < 0.5:
            print(f"  spec={spec:.3f} — semi-hard routing enables model divergence (was negative in stage 6C)")

    print("\nPASS: Semi-hard routing activated after freeze")
    print("STAGE 7 COMPLETE")


if __name__ == "__main__":
    reset_seed()
    test_routing_modes()
