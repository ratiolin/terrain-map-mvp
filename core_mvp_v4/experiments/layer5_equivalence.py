"""Layer 5: Functional Equivalence Classes.

Steps 18-22:
  Step 18: Multi-seed training (>=8 models at fixed d=16).
  Step 19: Behavioral consistency (action differences).
  Step 20: Representation differences (CKA).
  Step 21: Random baseline CKA.
  Step 22: Interpolation path (barrier height).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from core_mvp_v4.core_env import MultiDimEnv
from core_mvp_v4.core_models import V4Model, get_designed_hidden_dim
from core_mvp_v4.core_training import train_closed_loop, rollout
from core_mvp_v4.core_metrics import compute_cka
from core_mvp_v4.core_logger import ExperimentLogger

K = 2
DEFAULT_D = 16
N_SEEDS = 12
DEFAULT_LR = 1e-3
LAMBDA_CTRL = 0.1


def run_layer5(results_dir, d=None, n_seeds=None, episodes=5, steps=2000):
    if d is None:
        d = DEFAULT_D
    if n_seeds is None:
        n_seeds = N_SEEDS

    layer_dir = os.path.join(results_dir, "layer5")
    os.makedirs(layer_dir, exist_ok=True)

    hidden_dim = get_designed_hidden_dim(d)

    models = {}
    for seed in range(n_seeds):
        print(f"  Layer 5 - training model seed={seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = MultiDimEnv(d=d, k=K, drift=0.5, seed=seed)
        env.calibrate_noise_scales()
        model = V4Model(state_dim=d, hidden_dim=hidden_dim, action_dim=K)
        train_closed_loop(model, env, num_episodes=episodes,
                          episode_length=steps, lr=DEFAULT_LR,
                          lambda_ctrl=LAMBDA_CTRL, seed=seed)
        model.eval()
        models[seed] = model

    eval_env = MultiDimEnv(d=d, k=K, drift=0.5, seed=5000)
    eval_env.calibrate_noise_scales()

    trajectory_states = _generate_shared_trajectory(eval_env, steps=steps, seed=6000)

    action_diffs = _compute_action_differences(models, trajectory_states)
    print(f"  Mean action diff: {np.mean(action_diffs):.6f}, std: {np.std(action_diffs):.6f}")

    cka_matrix = _compute_cka_matrix(models, trajectory_states)
    cka_values = [ck for i, row in enumerate(cka_matrix) for j, ck in enumerate(row) if i < j]
    print(f"  CKA mean: {np.mean(cka_values):.4f}, std: {np.std(cka_values):.4f}")

    random_cka = _compute_random_cka_baseline(models, trajectory_states, n_seeds // 2)
    print(f"  Random baseline CKA: {np.mean(random_cka):.4f}")

    barrier_results = _compute_interpolation_barriers(models, eval_env, trajectory_states)

    results = {
        "d": d,
        "n_seeds": n_seeds,
        "action_diff_mean": float(np.mean(action_diffs)),
        "action_diff_std": float(np.std(action_diffs)),
        "cka_mean": float(np.mean(cka_values)),
        "cka_std": float(np.std(cka_values)),
        "random_cka_mean": float(np.mean(random_cka)),
        "barriers": barrier_results,
    }

    with open(os.path.join(layer_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    cka_save = {"cka_matrix": [[float(ck) for ck in row] for row in cka_matrix]}
    with open(os.path.join(layer_dir, "cka_matrix.json"), "w") as f:
        json.dump(cka_save, f, indent=2)

    print(f"Layer 5 complete. Results in {layer_dir}")
    return results


def _generate_shared_trajectory(env, steps, seed):
    env.set_seed(seed)
    env.reset()
    states = []
    state = env.get_state()
    for _ in range(steps):
        states.append(state.copy())
        a = np.zeros(K)
        ns, _, _, _ = env.step(a)
        state = ns
    return states


def _compute_action_differences(models, states):
    model_seeds = sorted(models.keys())
    all_diffs = []

    for s in states[:min(500, len(states))]:
        actions = []
        for seed in model_seeds:
            a = models[seed].act_numpy(s)
            actions.append(a)
        actions = np.array(actions)
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                all_diffs.append(float(np.linalg.norm(actions[i] - actions[j])))

    return all_diffs


def _compute_cka_matrix(models, states):
    model_seeds = sorted(models.keys())
    n = len(model_seeds)

    sample_states = states[:min(300, len(states))]

    hidden_collections = {}
    for seed in model_seeds:
        h_stack = []
        for s in sample_states:
            h = models[seed].f_numpy(s)
            h_stack.append(h)
        hidden_collections[seed] = np.array(h_stack)

    cka_matrix = np.zeros((n, n))
    for i, si in enumerate(model_seeds):
        for j, sj in enumerate(model_seeds):
            cka_matrix[i, j] = compute_cka(hidden_collections[si], hidden_collections[sj])

    return cka_matrix


def _compute_random_cka_baseline(models, states, n_random):
    hidden_dim = list(models.values())[0].hidden_dim
    n_states = min(200, len(states))

    real_hiddens = []
    for _, m in models.items():
        h_stack = []
        for s in states[:n_states]:
            h_stack.append(m.f_numpy(s))
        real_hiddens.append(np.array(h_stack))

    cka_vals = []
    for _ in range(n_random):
        random_h = np.random.randn(n_states, hidden_dim).astype(np.float64)
        cka_vals.append(compute_cka(real_hiddens[0], random_h))

    return cka_vals


def _compute_interpolation_barriers(models, env, states, alpha_steps=11):
    model_seeds = sorted(models.keys())
    alphas = np.linspace(0, 1, alpha_steps)

    n_pairs = min(5, len(model_seeds) * (len(model_seeds) - 1) // 2)
    barriers = []

    pair_idx = 0
    for i in range(len(model_seeds)):
        for j in range(i + 1, len(model_seeds)):
            if pair_idx >= n_pairs:
                break
            si, sj = model_seeds[i], model_seeds[j]
            barrier = _compute_single_barrier(models[si], models[sj], env, states, alphas)
            barriers.append(barrier)
            pair_idx += 1

    return {
        "barrier_mean": float(np.mean(barriers)),
        "barrier_std": float(np.std(barriers)),
        "barrier_max": float(np.max(barriers)),
        "barrier_min": float(np.min(barriers)),
        "n_pairs": len(barriers),
    }


def _compute_single_barrier(model_a, model_b, env, states, alphas):
    state_dict_a = {k: v.clone() for k, v in model_a.state_dict().items()}
    state_dict_b = {k: v.clone() for k, v in model_b.state_dict().items()}

    test_states = states[:min(100, len(states))]

    costs = []
    for alpha in alphas:
        state_dict_interp = {}
        for key in state_dict_a:
            if key in state_dict_b and isinstance(state_dict_a[key], torch.Tensor):
                state_dict_interp[key] = (1 - alpha) * state_dict_a[key] + alpha * state_dict_b[key]

        interp_model = V4Model(state_dim=model_a.state_dim,
                               hidden_dim=model_a.hidden_dim,
                               action_dim=model_a.action_dim)
        interp_model.load_state_dict(state_dict_interp, strict=False)
        interp_model.eval()

        total_risk = 0.0
        for s in test_states:
            env.state[:] = s.copy()
            a = interp_model.act_numpy(s)
            ns, risk, _, _ = env.step(a)
            total_risk += risk
        avg_cost = total_risk / max(len(test_states), 1)
        costs.append(avg_cost)

    if max(costs) > 0:
        barrier = (max(costs) - min(costs)) / max(costs)
    else:
        barrier = 0.0

    return float(barrier)
