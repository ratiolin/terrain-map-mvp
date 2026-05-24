import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from env_drifting_double_well import DriftingDoubleWell
from baseline_single import (
    MLP,
    count_multi_params,
    make_single_large,
    train_single_model,
)
from agent import Agent
from controller import GatingGrowthController
from metrics import compute_metrics_from_seed, compute_stability, compute_dwell_time, compute_prediction_variance


def reset_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _state_dim_from_env(env):
    return 2 if env.add_context else 1


def _make_agent(env):
    sd = _state_dim_from_env(env)
    return Agent(obs_dim=sd, act_dim=1, hidden_dim=2, lr=1e-3)


def _make_ctrl(agent, K_budget, inertia=0.0):
    ctrl = GatingGrowthController(
        check_interval=200,
        env_type="doublewell",
        merge_thresh=0.2,
        prune_thresh=0.03,
        max_models=8,
        use_z=True,
        inertia=inertia,
    )
    ctrl.init_models(agent)
    for _ in range(K_budget - 1):
        child = copy.deepcopy(ctrl.models[0])
        child.optimizer = optim.Adam(child.parameters(), lr=1e-3)
        with torch.no_grad():
            for p in child.predictor.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        ctrl.models.append(child)
        ctrl.gating.expand()
        ctrl.gating_optimizer = optim.Adam(ctrl.gating.parameters(), lr=1e-3)
    K = len(ctrl.models)
    ctrl.usage = [0] * K
    ctrl.errors = [[] for _ in range(K)]
    ctrl.birth_step = [0] * K
    ctrl.region_bias = [0.0] * K
    ctrl.weight_history = [[] for _ in range(K)]
    return ctrl


def train_multi_expert(env, ctrl, agent, steps):
    obs = env.reset()
    ctrl.gating_reset()
    mse_history = []
    oracle_history = []
    z_history = []
    state_history = []
    sign_history = []

    for t in range(steps):
        k_before = ctrl.n_models()
        ctrl.maybe_split()
        if k_before != ctrl.n_models():
            k_before = ctrl.n_models()

        a = agent.act(obs)
        o_next, _, done = env.step(a)

        target = torch.tensor(o_next, dtype=torch.float32)

        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        z_soft, z_logits, _ = ctrl.gating(s)
        ctrl._last_logits = z_logits
        ctrl._last_z_soft = z_soft
        K_cur = z_soft.size(-1)

        z_hard_idx = z_logits.argmax(dim=-1)
        z_hard = F.one_hot(z_hard_idx, K_cur).float()
        weights = (z_hard - z_soft.detach() + z_soft).squeeze(0)

        preds = [m.predict(obs, a) for m in ctrl.models]
        soft_pred = sum(weights[i] * preds[i] for i in range(K_cur))

        loss = ((soft_pred - target) ** 2).mean()

        with torch.no_grad():
            perr = torch.stack([
                ((preds[i].detach() - target) ** 2).mean()
                for i in range(K_cur)
            ])
            oracle_err = perr.min().item()

        error = float(np.mean(np.abs(soft_pred.detach().numpy() - o_next)))
        ctrl.should_update(error, 0)
        ctrl.record_usage(int(weights.argmax().item()))
        for i in range(K_cur):
            e_i = float(np.mean(np.abs(preds[i].detach().numpy() - o_next)))
            ctrl.track_error(i, e_i)

        ctrl.gating_optimizer.zero_grad()
        for m in ctrl.models:
            m.optimizer.zero_grad()
        loss.backward()
        ctrl.gating_optimizer.step()
        for m in ctrl.models:
            m.optimizer.step()

        mse_history.append(loss.item())
        oracle_history.append(oracle_err)
        z_history.append(weights.detach().numpy().copy())
        state_history.append(float(o_next[0]))
        sign_history.append(float(env.sign))

        ctrl.maybe_merge()
        ctrl.maybe_prune()

        if done:
            obs = env.reset()
            ctrl.gating_reset()
        else:
            obs = o_next

    K_max = max(len(z) for z in z_history) if z_history else 1
    return mse_history, oracle_history, z_history, state_history, sign_history, ctrl.n_models(), K_max


def evaluate_multi(env, ctrl, steps):
    obs = env.reset()
    ctrl.gating_reset()
    mse_history = []
    z_history = []
    state_history = []
    sign_history = []
    phase_history = []
    oracle_mse_history = []
    routing_gap_history = []

    flip_period = getattr(env, 'flip_period', 500)
    for t_idx in range(steps):
        a = random.randint(0, 1)
        o_next, _, done = env.step(a)
        target = torch.tensor(o_next, dtype=torch.float32)
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            z_soft, z_logits, _ = ctrl.gating(s)
            K_cur = z_soft.size(-1)
            z_hard_idx = z_logits.argmax(dim=-1)
            z_hard = F.one_hot(z_hard_idx, K_cur).float()
            weights = (z_hard - z_soft.detach() + z_soft).squeeze(0)
            preds = [m.predict(obs, a) for m in ctrl.models]
            soft_pred = sum(weights[i] * preds[i] for i in range(K_cur))

            perr = torch.stack([
                ((preds[i] - target) ** 2).mean() for i in range(K_cur)
            ])
            oracle_err = perr.min().item()
            chosen_k = int(weights.argmax().item())
            chosen_err = perr[chosen_k].item()

        loss = ((soft_pred - target) ** 2).mean()
        mse_history.append(loss.item())
        oracle_mse_history.append(oracle_err)
        routing_gap_history.append(chosen_err - oracle_err)
        z_history.append(weights.numpy().copy())
        state_history.append(float(o_next[0]))
        sign_history.append(float(env.sign))
        phase_history.append((int(env.t % flip_period), int(weights.argmax().item())))
        if done:
            obs = env.reset()
            ctrl.gating_reset()
        else:
            obs = o_next
    K_max = max(len(z) for z in z_history) if z_history else 1
    return mse_history, z_history, state_history, sign_history, phase_history, K_max, \
        oracle_mse_history, routing_gap_history


def evaluate_single(model, env, steps):
    obs = env.reset()
    mse_history = []
    state_history = []
    for _ in range(steps):
        action = random.randint(0, 1)
        o_next, _, done = env.step(action)
        pred = model.predict(obs, action)
        target = torch.tensor(o_next, dtype=torch.float32)
        loss = ((pred - target) ** 2).mean()
        mse_history.append(loss.item())
        state_history.append(float(o_next[0]))
        if done:
            obs = env.reset()
        else:
            obs = o_next
    return mse_history, state_history


def compute_routing_stability(z_history, K_max):
    padded = np.zeros((len(z_history), K_max))
    for t, z in enumerate(z_history):
        L = min(len(z), K_max)
        padded[t, :L] = z[:L]
    z_arr = padded
    max_fraction = float(np.max(z_arr.mean(axis=0)))
    switches = 0
    for t in range(1, len(z_arr)):
        if np.argmax(z_arr[t]) != np.argmax(z_arr[t - 1]):
            switches += 1
    switch_rate = switches / max(1, len(z_arr) - 1)
    return max_fraction, switch_rate


def compute_expert_regions(state_history, z_history, K_max):
    regions = {k: [] for k in range(K_max)}
    for s, z in zip(state_history, z_history):
        k_best = int(np.argmax(z)) if len(z) > 0 else 0
        regions[k_best].append(s)
    expert_means = {k: float(np.mean(v)) if v else 0.0 for k, v in regions.items()}
    return expert_means


def compute_separation(expert_means):
    vals = [v for v in expert_means.values() if v != 0.0 or len(expert_means) <= 2]
    if len(vals) < 2:
        return False
    return (max(vals) - min(vals)) > 0.5


def compute_ctx_align(z_history, sign_history, K_max):
    z_arr = np.zeros((len(z_history), K_max))
    for t, z in enumerate(z_history):
        L = min(len(z), K_max)
        z_arr[t, :L] = z[:L]
    expert_idx = z_arr.argmax(axis=1)
    signs = np.array(sign_history)

    best_align = 0.0
    for k in range(K_max):
        sel_k = (expert_idx == k)
        if sel_k.sum() < 5:
            continue
        pos_when_k = (signs[sel_k] > 0).mean()
        align = max(pos_when_k, 1 - pos_when_k)
        if align > best_align:
            best_align = align
    return float(best_align)


def run_condition_multi(kappa, drift, flip_mode, add_context,
                         K_budget, expert_hidden, gating_hidden,
                         train_steps, test_steps, seeds, inertia=0.0):
    seed_results = []
    for seed in seeds:
        reset_seed(seed)
        env_train = DriftingDoubleWell(
            kappa=kappa, drift_rate=drift,
            flip_mode=flip_mode, add_context=add_context,
        )
        agent = _make_agent(env_train)
        ctrl = _make_ctrl(agent, K_budget, inertia=inertia)

        (_mse_train, oracle_train_hist, _z_train, _st_train,
         _sgn_train, _K_train, _Kmax_train) = train_multi_expert(
            env_train, ctrl, agent, train_steps,
        )
        oracle_train_mean = float(np.mean(oracle_train_hist[-500:])) if len(oracle_train_hist) >= 500 else float(np.mean(oracle_train_hist))

        reset_seed(seed + 10000)
        env_test = DriftingDoubleWell(
            kappa=kappa, drift_rate=drift,
            flip_mode=flip_mode, add_context=add_context,
        )
        (mse_test, z_test, st_test, sgn_test, phase_test, K_max,
         oracle_mse_list, routing_gap_list) = evaluate_multi(
            env_test, ctrl, test_steps,
        )
        test_mse = float(np.mean(mse_test))
        oracle_mse = float(np.mean(oracle_mse_list))
        routing_gap = float(np.mean(routing_gap_list))
        max_frac, switch_rate = compute_routing_stability(z_test, K_max)
        expert_means = compute_expert_regions(st_test, z_test, K_max)
        separation = compute_separation(expert_means)
        ctx_align = compute_ctx_align(z_test, sgn_test, K_max)

        seed_result = {
            "test_mse": test_mse,
            "mse_test_history": mse_test,
            "oracle_mse": oracle_mse,
            "oracle_train_mse": oracle_train_mean,
            "routing_gap": routing_gap,
            "max_fraction": max_frac,
            "switch_rate": switch_rate,
            "expert_means": expert_means,
            "separation": separation,
            "ctx_align": ctx_align,
            "K_final": ctrl.n_models(),
            "phase_expert": phase_test,
            "state_history": st_test,
            "z_history": z_test,
            "sign_history": sgn_test,
        }

        metrics = compute_metrics_from_seed(seed_result)
        seed_result["variance"] = metrics["variance"]
        seed_result["consistency"] = metrics["consistency"]
        seed_result["dwell_time"] = metrics["dwell_time"]
        seed_result["response_time"] = metrics["response_time"]

        seed_results.append(seed_result)

    return seed_results


def run_condition_single_large(kappa, drift, flip_mode, add_context,
                                K_budget, expert_hidden, gating_hidden,
                                train_steps, test_steps, seeds):
    state_dim = 2 if add_context else 1
    test_mses = []
    mse_histories = []
    state_histories = []
    H_large = None
    params_large = None
    for seed in seeds:
        reset_seed(seed)
        model_large, H_large = make_single_large(
            K_budget, expert_hidden, gating_hidden, state_dim=state_dim,
        )
        params_large = model_large.count_params()
        env_train = DriftingDoubleWell(
            kappa=kappa, drift_rate=drift,
            flip_mode=flip_mode, add_context=add_context,
        )
        train_single_model(model_large, env_train, train_steps, lr=1e-3, seed=seed)

        reset_seed(seed + 10000)
        env_test = DriftingDoubleWell(
            kappa=kappa, drift_rate=drift,
            flip_mode=flip_mode, add_context=add_context,
        )
        mse_test, st_test = evaluate_single(model_large, env_test, test_steps)
        test_mses.append(float(np.mean(mse_test)))
        mse_histories.append(mse_test)
        state_histories.append(st_test)

    return test_mses, params_large, H_large, mse_histories, state_histories


def run_condition_single_small(kappa, drift, flip_mode, add_context,
                                expert_hidden, train_steps, test_steps, seed):
    state_dim = 2 if add_context else 1
    reset_seed(seed)
    model_small = MLP(hidden_dim=expert_hidden, state_dim=state_dim)
    env_train = DriftingDoubleWell(
        kappa=kappa, drift_rate=drift,
        flip_mode=flip_mode, add_context=add_context,
    )
    train_single_model(model_small, env_train, train_steps, lr=1e-3, seed=seed)

    reset_seed(seed + 10000)
    env_test = DriftingDoubleWell(
        kappa=kappa, drift_rate=drift,
        flip_mode=flip_mode, add_context=add_context,
    )
    mse_test, _ = evaluate_single(model_small, env_test, test_steps)
    return float(np.mean(mse_test)), model_small.count_params()


def run_triplet(kappa, drift, K_budget=4, expert_hidden=2, gating_hidden=8,
                train_steps=1200, test_steps=300, seeds=(42, 43, 44), inertia=0.0):
    configs = [
        ("det", "deterministic", False),
        ("rand_noctx", "random", False),
        ("rand_ctx", "random", True),
    ]

    triplet = {}

    test_small_1d, params_small_1d = run_condition_single_small(
        kappa, drift, flip_mode="deterministic", add_context=False,
        expert_hidden=expert_hidden,
        train_steps=train_steps, test_steps=test_steps, seed=seeds[0],
    )
    triplet["test_small_1d"] = test_small_1d
    triplet["params_small_1d"] = params_small_1d

    for tag, flip_mode, add_ctx in configs:
        seed_results = run_condition_multi(
            kappa=kappa, drift=drift,
            flip_mode=flip_mode, add_context=add_ctx,
            K_budget=K_budget, expert_hidden=expert_hidden,
            gating_hidden=gating_hidden,
            train_steps=train_steps, test_steps=test_steps,
            seeds=seeds,
            inertia=inertia,
        )

        test_large_list, params_large, H_large, mse_hist_large, state_hist_large = run_condition_single_large(
            kappa=kappa, drift=drift,
            flip_mode=flip_mode, add_context=add_ctx,
            K_budget=K_budget, expert_hidden=expert_hidden,
            gating_hidden=gating_hidden,
            train_steps=train_steps, test_steps=test_steps,
            seeds=seeds,
        )

        test_large_mean = float(np.mean(test_large_list))

        S_single_per_seed = []
        for i in range(len(seeds)):
            var_s = compute_prediction_variance(mse_hist_large[i])
            dwell_s = compute_dwell_time(state_hist_large[i])
            S_single_per_seed.append(compute_stability(var_s, 1.0, dwell_s))
        S_single_mean = float(np.mean(S_single_per_seed))

        for i, r in enumerate(seed_results):
            r["mse_gap"] = test_large_list[i] - r["test_mse"]
            r["single_large_mse"] = test_large_list[i]

        mse_gaps = [s["mse_gap"] for s in seed_results]
        switch_rates = [s["switch_rate"] for s in seed_results]
        separations = [s["separation"] for s in seed_results]
        oracle_mses = [s["oracle_mse"] for s in seed_results]
        oracle_train_mses = [s["oracle_train_mse"] for s in seed_results]
        routing_gaps = [s["routing_gap"] for s in seed_results]
        variances = [s["variance"] for s in seed_results]
        consistencies = [s["consistency"] for s in seed_results]
        dwell_times = [s["dwell_time"] for s in seed_results]
        response_times = [s["response_time"] for s in seed_results]

        triplet[tag] = {
            "seeds": seed_results,
            "test_large_mean": test_large_mean,
            "params_large": params_large,
            "H_large": H_large,
            "mse_gap_mean": float(np.mean(mse_gaps)),
            "switch_rate_mean": float(np.mean(switch_rates)),
            "separation_majority": sum(separations) >= 2,
            "mse_gap_pass": sum(1 for g in mse_gaps if g > 0.1) >= 2,
            "switch_rate_pass": sum(1 for s in switch_rates if s < 0.05) >= 2,
            "oracle_mse_mean": float(np.mean(oracle_mses)),
            "oracle_train_mse_mean": float(np.mean(oracle_train_mses)),
            "routing_gap_mean": float(np.mean(routing_gaps)),
            "variance_mean": float(np.mean(variances)),
            "consistency_mean": float(np.mean(consistencies)),
            "dwell_time_mean": float(np.mean(dwell_times)),
            "response_time_mean": float(np.mean(response_times)),
            "S_single_mean": S_single_mean,
        }

        of_str = ""
        if triplet[tag]["oracle_train_mse_mean"] > 0:
            ratio = triplet[tag]["oracle_train_mse_mean"] / max(triplet[tag]["oracle_mse_mean"], 1e-8)
            of_str = f" overfit={ratio:.2f}" if ratio < 0.5 else ""

        print(f"  {tag:15s}: gap={triplet[tag]['mse_gap_mean']:+.4f}  "
              f"switch={triplet[tag]['switch_rate_mean']:.4f}  "
              f"sep={'Y' if triplet[tag]['separation_majority'] else 'N'}  "
              f"large_mse={test_large_mean:.4f}  "
              f"oracle={triplet[tag]['oracle_mse_mean']:.4f}  "
              f"r_gap={triplet[tag]['routing_gap_mean']:+.4f}"
              f"{of_str}")

    return triplet


def run_experiment_grid(kappa_list=None, drift_list=None, seeds=(42, 43, 44),
                         expert_hidden=2, K_budget=4, gating_hidden=8,
                         train_steps=1200, test_steps=300, inertia=0.0):
    if kappa_list is None:
        kappa_list = [0.5, 1.0, 2.0, 4.0]
    if drift_list is None:
        drift_list = [0.005, 0.02, 0.08]

    all_results = {}
    for kappa in kappa_list:
        for drift in drift_list:
            key = f"k{kappa}_d{drift}"
            print(f"\n{'='*60}")
            print(f"TRIPLET: kappa={kappa}, drift={drift}")
            print(f"{'='*60}")
            triplet = run_triplet(
                kappa=kappa, drift=drift,
                K_budget=K_budget, expert_hidden=expert_hidden,
                gating_hidden=gating_hidden,
                train_steps=train_steps, test_steps=test_steps,
                seeds=seeds,
                inertia=inertia,
            )
            all_results[key] = triplet

    return all_results
