import copy
import random
import numpy as np
import torch
import torch.optim as optim

from env_drifting_double_well import DriftingDoubleWell
from experiment10 import (
    reset_seed,
    _make_agent,
    _make_ctrl,
    train_multi_expert,
    evaluate_multi,
    evaluate_single,
    compute_routing_stability,
    compute_expert_regions,
    compute_separation,
    compute_ctx_align,
    run_condition_single_large,
)
from gating_multi_scale import (
    find_matched_sizes,
    DualGRUGating,
)
from gating import ZGatingNet
from baseline_single import MLP, train_single_model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def fit_linear_ensemble(env, experts, steps):
    obs = env.reset()
    X_rows = []
    y_rows = []

    for _ in range(steps):
        a = random.randint(0, 1)
        o_next, _, done = env.step(a)
        with torch.no_grad():
            preds = [m.predict(obs, a).numpy().flatten() for m in experts]
        X_rows.append(np.concatenate(preds))
        y_rows.append(o_next.flatten())
        obs = o_next if not done else env.reset()

    X = np.array(X_rows)
    y = np.array(y_rows)
    w, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    return w


def eval_linear_ensemble(env, experts, w, steps):
    obs = env.reset()
    mse_list = []
    for _ in range(steps):
        a = random.randint(0, 1)
        o_next, _, done = env.step(a)
        with torch.no_grad():
            preds = [m.predict(obs, a).numpy().flatten() for m in experts]
        x = np.concatenate(preds)
        pred = x @ w
        target = o_next.flatten()
        mse_list.append(float(np.mean((pred - target) ** 2)))
        obs = o_next if not done else env.reset()
    return float(np.mean(mse_list))


def run_linear_ensemble_test(kappa, drift, K_budget, expert_hidden, gating_hidden,
                              state_dim, train_steps, test_steps, seed):
    """Train model A, freeze experts, fit linear weights, evaluate."""
    reset_seed(seed)
    env_train = DriftingDoubleWell(
        kappa=kappa, drift_rate=drift,
        flip_mode="deterministic", add_context=False,
    )
    agent = _make_agent(env_train)
    ctrl = _make_ctrl(agent, K_budget)
    train_multi_expert(env_train, ctrl, agent, train_steps)

    experts = ctrl.models
    K_final = len(experts)

    reset_seed(seed)
    env_fit = DriftingDoubleWell(
        kappa=kappa, drift_rate=drift,
        flip_mode="deterministic", add_context=False,
    )
    w = fit_linear_ensemble(env_fit, experts, train_steps)

    reset_seed(seed + 10000)
    env_test = DriftingDoubleWell(
        kappa=kappa, drift_rate=drift,
        flip_mode="deterministic", add_context=False,
    )
    linear_mse = eval_linear_ensemble(env_test, experts, w, test_steps)

    return {
        "linear_mse": linear_mse,
        "K_final": K_final,
        "weights": w.flatten().tolist(),
    }


def run_model_A(kappa, drift, K_budget, expert_hidden, gating_hidden,
                state_dim, train_steps, test_steps, seed):
    """Baseline: single GRU ZGatingNet"""
    reset_seed(seed)
    env_train = DriftingDoubleWell(
        kappa=kappa, drift_rate=drift,
        flip_mode="deterministic", add_context=False,
    )
    agent = _make_agent(env_train)
    ctrl = _make_ctrl(agent, K_budget)

    (_mse_train, oracle_train_hist, _z_train, _st_train,
     _sgn_train, _K_train, _Kmax_train) = train_multi_expert(
        env_train, ctrl, agent, train_steps,
    )
    oracle_train_mean = float(np.mean(oracle_train_hist[-500:])) if len(oracle_train_hist) >= 500 else float(np.mean(oracle_train_hist))

    reset_seed(seed + 10000)
    env_test = DriftingDoubleWell(
        kappa=kappa, drift_rate=drift,
        flip_mode="deterministic", add_context=False,
    )
    (mse_test, z_test, st_test, sgn_test, phase_test, K_max,
     oracle_mse_list, routing_gap_list) = evaluate_multi(env_test, ctrl, test_steps)

    test_mse = float(np.mean(mse_test))
    oracle_mse = float(np.mean(oracle_mse_list))
    routing_gap = float(np.mean(routing_gap_list))
    max_frac, switch_rate = compute_routing_stability(z_test, K_max)
    expert_means = compute_expert_regions(st_test, z_test, K_max)
    separation = compute_separation(expert_means)
    ctx_align = compute_ctx_align(z_test, sgn_test, K_max)
    params = count_params(ctrl.gating)

    K_final = ctrl.n_models()

    return {
        "test_mse": test_mse,
        "oracle_train_mse": oracle_train_mean,
        "oracle_mse": oracle_mse,
        "routing_gap": routing_gap,
        "switch_rate": switch_rate,
        "separation": separation,
        "ctx_align": ctx_align,
        "K_final": K_final,
        "max_fraction": max_frac,
        "expert_means": expert_means,
        "gating_params": params,
        "phase_expert": phase_test,
    }


def run_model_dual(kappa, drift, K_budget, expert_hidden,
                   h1, h2, state_dim, train_steps, test_steps, seed):
    """Dual GRU gating (capacity-control or fast+slow)"""
    reset_seed(seed)
    env_train = DriftingDoubleWell(
        kappa=kappa, drift_rate=drift,
        flip_mode="deterministic", add_context=False,
    )
    agent = _make_agent(env_train)
    ctrl = _make_ctrl(agent, K_budget)

    ctrl.gating = DualGRUGating(state_dim, K_budget, h1, h2, temperature=0.5)
    ctrl.gating_optimizer = optim.Adam(ctrl.gating.parameters(), lr=1e-3)

    (_mse_train, oracle_train_hist, _z_train, _st_train,
     _sgn_train, _K_train, _Kmax_train) = train_multi_expert(
        env_train, ctrl, agent, train_steps,
    )
    oracle_train_mean = float(np.mean(oracle_train_hist[-500:])) if len(oracle_train_hist) >= 500 else float(np.mean(oracle_train_hist))

    reset_seed(seed + 10000)
    env_test = DriftingDoubleWell(
        kappa=kappa, drift_rate=drift,
        flip_mode="deterministic", add_context=False,
    )
    (mse_test, z_test, st_test, sgn_test, phase_test, K_max,
     oracle_mse_list, routing_gap_list) = evaluate_multi(env_test, ctrl, test_steps)

    test_mse = float(np.mean(mse_test))
    oracle_mse = float(np.mean(oracle_mse_list))
    routing_gap = float(np.mean(routing_gap_list))
    max_frac, switch_rate = compute_routing_stability(z_test, K_max)
    expert_means = compute_expert_regions(st_test, z_test, K_max)
    separation = compute_separation(expert_means)
    ctx_align = compute_ctx_align(z_test, sgn_test, K_max)
    params = count_params(ctrl.gating)

    K_final = ctrl.n_models()

    return {
        "test_mse": test_mse,
        "oracle_train_mse": oracle_train_mean,
        "oracle_mse": oracle_mse,
        "routing_gap": routing_gap,
        "switch_rate": switch_rate,
        "separation": separation,
        "ctx_align": ctx_align,
        "K_final": K_final,
        "max_fraction": max_frac,
        "expert_means": expert_means,
        "gating_params": params,
        "phase_expert": phase_test,
    }


def judge_improvement(baseline, variant, name, label):
    r_gap_base = baseline["routing_gap"]
    r_gap_var = variant["routing_gap"]
    t_mse_base = baseline["test_mse"]
    t_mse_var = variant["test_mse"]
    o_mse_base = baseline["oracle_mse"]
    o_mse_var = variant["oracle_mse"]

    gap_drop = (r_gap_base - r_gap_var) / max(r_gap_base, 1e-8)
    t_gap_base = t_mse_base - o_mse_base
    t_gap_var = t_mse_var - o_mse_var
    t_gap_drop = (t_gap_base - t_gap_var) / max(t_gap_base, 1e-8) if t_gap_base > 0 else 0.0

    improved = (gap_drop >= 0.30) and (t_gap_drop >= 0.30)

    print(f"  {label:8s} | rout_gap: {r_gap_base:.4f} → {r_gap_var:.4f} (Δ={gap_drop*100:+.0f}%)  "
          f"t_mse: {t_mse_base:.4f} → {t_mse_var:.4f}  "
          f"t_gap(Δ>={0.30})={'Y' if t_gap_drop >= 0.3 else 'N'}  "
          f"IMPROVED={'YES' if improved else 'NO '}")

    return {
        "label": label,
        "gap_drop": gap_drop,
        "t_gap_drop": t_gap_drop,
        "improved": improved,
    }


def run_ablation():
    kappa = 1.0
    drift = 0.08
    K_budget = 4
    expert_hidden = 2
    gating_hidden = 8
    state_dim = 1
    train_steps = 1200
    test_steps = 300
    seeds = [0, 1, 2]

    sizes = find_matched_sizes(
        state_dim=state_dim, K=K_budget,
        H_baseline=gating_hidden, ratio_fast=0.25, tolerance=0.05,
    )
    print("=" * 70)
    print("GATING PARAMETER MATCHING")
    print(f"  Baseline (ZGatingNet H={gating_hidden}):  {sizes['baseline_params']} params")
    print(f"  Ctrl (DualGRU h={sizes['capacity_ctrl_h']}×2):       "
          f"{sizes['capacity_ctrl_params']} params  (err={sizes['capacity_ctrl_err']*100:.1f}%)")
    print(f"  FastSlow (h_fast={sizes['fast_h']}, h_slow={sizes['slow_h']}): "
          f"{sizes['fast_slow_params']} params  (err={sizes['fast_slow_err']*100:.1f}%)")
    print("=" * 70)

    all_A = []
    all_B = []
    all_C = []
    all_linear = []

    test_large_list, _, _ = run_condition_single_large(
        kappa=kappa, drift=drift,
        flip_mode="deterministic", add_context=False,
        K_budget=K_budget, expert_hidden=expert_hidden,
        gating_hidden=gating_hidden,
        train_steps=train_steps, test_steps=test_steps,
        seeds=tuple(seeds),
    )
    single_large_mean = float(np.mean(test_large_list))

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        rA = run_model_A(
            kappa, drift, K_budget, expert_hidden, gating_hidden,
            state_dim, train_steps, test_steps, seed,
        )
        rA["single_large_mse"] = test_large_list[seed]
        all_A.append(rA)

        rLin = run_linear_ensemble_test(
            kappa, drift, K_budget, expert_hidden, gating_hidden,
            state_dim, train_steps, test_steps, seed,
        )
        rLin["single_large_mse"] = test_large_list[seed]
        all_linear.append(rLin)

        h_ctrl = sizes["capacity_ctrl_h"]
        rB = run_model_dual(
            kappa, drift, K_budget, expert_hidden,
            h_ctrl, h_ctrl, state_dim, train_steps, test_steps, seed,
        )
        rB["single_large_mse"] = test_large_list[seed]
        all_B.append(rB)

        rC = run_model_dual(
            kappa, drift, K_budget, expert_hidden,
            sizes["fast_h"], sizes["slow_h"], state_dim,
            train_steps, test_steps, seed,
        )
        rC["single_large_mse"] = test_large_list[seed]
        all_C.append(rC)

    avg = lambda lst, key: float(np.mean([r[key] for r in lst]))

    print(f"\n{'='*70}")
    print("RESULTS (mean across seeds)")
    print(f"{'='*70}")
    print(f"  single_large_mse = {single_large_mean:.4f}")
    print()
    print(f"  {'':8s} {'test_mse':>10s} {'oracle_tr':>10s} {'oracle':>10s} {'r_gap':>10s} {'switch':>10s} {'params':>10s}")
    print(f"  {'A base':8s} {avg(all_A,'test_mse'):10.4f} "
          f"{avg(all_A,'oracle_train_mse'):10.4f} {avg(all_A,'oracle_mse'):10.4f} "
          f"{avg(all_A,'routing_gap'):10.4f} {avg(all_A,'switch_rate'):10.4f} "
          f"{int(avg(all_A,'gating_params')):10d}")
    print(f"  {'B ctrl':8s} {avg(all_B,'test_mse'):10.4f} "
          f"{avg(all_B,'oracle_train_mse'):10.4f} {avg(all_B,'oracle_mse'):10.4f} "
          f"{avg(all_B,'routing_gap'):10.4f} {avg(all_B,'switch_rate'):10.4f} "
          f"{int(avg(all_B,'gating_params')):10d}")
    print(f"  {'C fast':8s} {avg(all_C,'test_mse'):10.4f} "
          f"{avg(all_C,'oracle_train_mse'):10.4f} {avg(all_C,'oracle_mse'):10.4f} "
          f"{avg(all_C,'routing_gap'):10.4f} {avg(all_C,'switch_rate'):10.4f} "
          f"{int(avg(all_C,'gating_params')):10d}")

    lin_mse = avg(all_linear, "linear_mse")
    lin_k = avg(all_linear, "K_final")
    print(f"  {'D linear':8s} {lin_mse:10.4f}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  K={int(lin_k)}")

    print(f"\n  {'Overfitting check (oracle_train / oracle_test):':>55s}")
    for label, lst in [("A base", all_A), ("B ctrl", all_B), ("C fast", all_C)]:
        ot = avg(lst, "oracle_train_mse")
        os_ = avg(lst, "oracle_mse")
        ratio = ot / max(os_, 1e-8)
        flag = "OVERFIT" if ratio < 0.5 else "ok"
        print(f"    {label:8s}: {ot:.4f} / {os_:.4f} = {ratio:.3f}  → {flag}")

    print(f"\n  {'Linear ensemble vs single_large:':>55s}")
    print(f"    linear_mse = {lin_mse:.4f}  vs  single_large_mse = {single_large_mean:.4f}")
    if lin_mse < single_large_mean * 1.1:
        print(f"    → ❗ 分化结构错误：冻结专家+线性权重 ≈ 单一大模型")
        print(f"    → 专家集合有足够容量，GRU routing 未能找到正确组合")
    else:
        print(f"    → ❌ 真正容量不足：即使最优线性组合也无法超越 single_large")

    baseline_mean = {
        "routing_gap": avg(all_A, "routing_gap"),
        "test_mse": avg(all_A, "test_mse"),
        "oracle_mse": avg(all_A, "oracle_mse"),
    }
    B_mean = {
        "routing_gap": avg(all_B, "routing_gap"),
        "test_mse": avg(all_B, "test_mse"),
        "oracle_mse": avg(all_B, "oracle_mse"),
    }
    C_mean = {
        "routing_gap": avg(all_C, "routing_gap"),
        "test_mse": avg(all_C, "test_mse"),
        "oracle_mse": avg(all_C, "oracle_mse"),
    }

    print(f"\n{'='*70}")
    print("JUDGMENT")
    print(f"{'='*70}")
    res_B = judge_improvement(baseline_mean, B_mean, all_B, "B (ctrl)")
    res_C = judge_improvement(baseline_mean, C_mean, all_C, "C (fast)")

    print()
    if res_B["improved"] and res_C["improved"]:
        print("→ ❌ 容量效应（增加GRU数量即可改善，非时间尺度特化）")
    elif res_C["improved"] and not res_B["improved"]:
        print("→ ✅ 时间尺度效应（异构时间尺度关键，同质扩展无效）")
    elif not res_B["improved"] and not res_C["improved"]:
        print("→ ❌ 不是时间尺度瓶颈（routing failure有其他原因）")
    elif res_B["improved"] and not res_C["improved"]:
        print("→ ⚠️  结构被破坏（异质时间尺度反而干扰）")
    else:
        print("→ — 无法判定")

    print(f"\n{'='*70}")
    print("ABLATION COMPLETE")
    print(f"{'='*70}")

    return all_A, all_B, all_C, sizes


if __name__ == "__main__":
    run_ablation()
