import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy

from env_drifting_double_well import DriftingDoubleWell
from experiment10 import (
    reset_seed, _make_agent, _make_ctrl,
    evaluate_multi, evaluate_single,
    compute_routing_stability, compute_expert_regions, compute_separation,
)
from baseline_single import MLP, make_single_large, train_single_model
from agent import Agent
from controller import GatingGrowthController


def train_multi_openloop(env, ctrl, agent, steps):
    """Train multi-expert with random actions (open-loop, no feedback)."""
    obs = env.reset()
    ctrl.gating_reset()
    mse_history = []
    oracle_history = []
    z_history = []

    for t in range(steps):
        ctrl.maybe_split()
        ctrl.maybe_merge()
        ctrl.maybe_prune()

        a = random.randint(0, 1)
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
                ((preds[i].detach() - target) ** 2).mean() for i in range(K_cur)
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

        if done:
            obs = env.reset()
            ctrl.gating_reset()
        else:
            obs = o_next

    return mse_history, oracle_history, z_history, ctrl.n_models()


def fit_linear_ensemble(env, experts, steps):
    obs = env.reset()
    X_rows, y_rows = [], []
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
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
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
        mse_list.append(float(np.mean((pred - o_next.flatten()) ** 2)))
        obs = o_next if not done else env.reset()
    return float(np.mean(mse_list))


def run_single_large(kappa, drift, K_budget, expert_hidden, gating_hidden,
                     train_steps, test_steps, seeds):
    test_mses = []
    for seed in seeds:
        reset_seed(seed)
        model_large, _ = make_single_large(K_budget, expert_hidden, gating_hidden, state_dim=1)
        env_train = DriftingDoubleWell(kappa=kappa, drift_rate=drift,
                                        flip_mode="deterministic", add_context=False)
        train_single_model(model_large, env_train, train_steps, lr=1e-3, seed=seed)
        reset_seed(seed + 10000)
        env_test = DriftingDoubleWell(kappa=kappa, drift_rate=drift,
                                       flip_mode="deterministic", add_context=False)
        mse_test, _ = evaluate_single(model_large, env_test, test_steps)
        test_mses.append(float(np.mean(mse_test)))
    return test_mses


def run_one_condition(kappa, drift, K_budget, expert_hidden, gating_hidden,
                      train_steps, test_steps, seed, open_loop=False):
    reset_seed(seed)
    env_train = DriftingDoubleWell(kappa=kappa, drift_rate=drift,
                                    flip_mode="deterministic", add_context=False)
    agent = _make_agent(env_train)
    ctrl = _make_ctrl(agent, K_budget)

    if open_loop:
        mse_hist, oracle_hist, z_hist, K_final = train_multi_openloop(
            env_train, ctrl, agent, train_steps)
    else:
        from experiment10 import train_multi_expert
        (mse_hist, oracle_hist, z_hist, _st, _sg, K_final, _Kmax) = train_multi_expert(
            env_train, ctrl, agent, train_steps)

    oracle_train_mean = float(np.mean(oracle_hist[-500:])) if len(oracle_hist) >= 500 else float(np.mean(oracle_hist))

    reset_seed(seed + 10000)
    env_test = DriftingDoubleWell(kappa=kappa, drift_rate=drift,
                                   flip_mode="deterministic", add_context=False)
    (mse_test, z_test, st_test, sgn_test, phase_test, K_max,
     oracle_mse_list, routing_gap_list) = evaluate_multi(env_test, ctrl, test_steps)

    test_mse = float(np.mean(mse_test))
    oracle_mse = float(np.mean(oracle_mse_list))
    routing_gap = float(np.mean(routing_gap_list))
    max_frac, switch_rate = compute_routing_stability(z_test, K_max)
    expert_means = compute_expert_regions(st_test, z_test, K_max)
    separation = compute_separation(expert_means)

    experts = ctrl.models
    reset_seed(seed)
    env_fit = DriftingDoubleWell(kappa=kappa, drift_rate=drift,
                                  flip_mode="deterministic", add_context=False)
    w = fit_linear_ensemble(env_fit, experts, train_steps)

    reset_seed(seed + 10000)
    env_lin = DriftingDoubleWell(kappa=kappa, drift_rate=drift,
                                  flip_mode="deterministic", add_context=False)
    linear_mse = eval_linear_ensemble(env_lin, experts, w, test_steps)

    return {
        "test_mse": test_mse,
        "oracle_train_mse": oracle_train_mean,
        "oracle_mse": oracle_mse,
        "routing_gap": routing_gap,
        "switch_rate": switch_rate,
        "separation": separation,
        "K_final": K_final,
        "linear_mse": linear_mse,
        "weights": w.flatten().tolist(),
    }


def run_benchmark(drifts, seeds, kappa=1.0, K_budget=4, expert_hidden=2,
                   gating_hidden=8, train_steps=1200, test_steps=300):
    results = {}

    for drift in drifts:
        key = f"d{drift}"
        print(f"\n{'='*60}")
        print(f"DRIFT={drift}")
        print(f"{'='*60}")

        large_mses = run_single_large(kappa, drift, K_budget, expert_hidden,
                                       gating_hidden, train_steps, test_steps, seeds)

        closed = []
        opened = []
        for seed in seeds:
            r_cl = run_one_condition(kappa, drift, K_budget, expert_hidden,
                                      gating_hidden, train_steps, test_steps, seed,
                                      open_loop=False)
            r_cl["single_large_mse"] = large_mses[seed]
            closed.append(r_cl)

            if drift == 0.02:
                r_op = run_one_condition(kappa, drift, K_budget, expert_hidden,
                                          gating_hidden, train_steps, test_steps, seed,
                                          open_loop=True)
                r_op["single_large_mse"] = large_mses[seed]
                opened.append(r_op)
            else:
                opened.append(None)

        avg = lambda lst, key: float(np.mean([r[key] for r in lst]))

        results[key] = {
            "closed": closed,
            "opened": opened if opened[0] is not None else None,
            "single_large_mean": float(np.mean(large_mses)),
            "linear_mean": avg(closed, "linear_mse"),
            "test_mse_mean": avg(closed, "test_mse"),
            "oracle_mean": avg(closed, "oracle_mse"),
        }

        print(f"  single_large = {results[key]['single_large_mean']:.4f}")
        print(f"  linear_mse   = {results[key]['linear_mean']:.4f}")
        print(f"  oracle_test  = {results[key]['oracle_mean']:.4f}")
        print(f"  A test_mse   = {results[key]['test_mse_mean']:.4f}")
        gap = results[key]["linear_mean"] - results[key]["single_large_mean"]
        print(f"  gap          = {gap:+.4f}  ({'✅ 分化有益' if gap < 0 else '❌ 分化失效'})")

        n_success = sum(1 for r in closed if r["linear_mse"] < r["single_large_mse"])
        print(f"  success_rate = {n_success}/{len(seeds)}")
        results[key]["success_rate"] = n_success / len(seeds)

    return results


def print_table(results):
    print("\n" + "=" * 80)
    print("PHASE TRANSITION TABLE")
    print("=" * 80)
    print(f"  {'drift':>8s} | {'linear':>10s} | {'single_l':>10s} | {'gap':>10s} | {'success':>8s} | {'oracle':>10s} | {'A_test':>10s}")
    print("  " + "-" * 78)
    for key in sorted(results.keys()):
        r = results[key]
        gap = r["linear_mean"] - r["single_large_mean"]
        sr = f"{r['success_rate']:.2f}"
        print(f"  {key:>8s} | {r['linear_mean']:10.4f} | {r['single_large_mean']:10.4f} | "
              f"{gap:+10.4f} | {sr:>8s} | {r['oracle_mean']:10.4f} | {r['test_mse_mean']:10.4f}")

    if results.get("d0.02") and results["d0.02"].get("opened"):
        op = results["d0.02"]["opened"]
        avg_op = lambda key: float(np.mean([r[key] for r in op]))
        print(f"\n  {'Open-loop (drift=0.02)':>40s}")
        print(f"  {'':>8s} | {'linear':>10s} | {'single_l':>10s} | {'gap':>10s}")
        lin_op = avg_op("linear_mse")
        sl_op = avg_op("single_large_mse")
        gap_op = lin_op - sl_op
        print(f"  {'open':>8s} | {lin_op:10.4f} | {sl_op:10.4f} | {gap_op:+10.4f}")
        n_op = sum(1 for r in op if r["linear_mse"] < r["single_large_mse"])
        print(f"  success_rate = {n_op}/{len(op)}")

    print("=" * 80)


if __name__ == "__main__":
    drifts = [0.005, 0.02, 0.04, 0.08]
    seeds = [0, 1, 2]

    print("=" * 60)
    print("STAGE 10 FINAL: Phase Transition Sweep")
    print("=" * 60)

    results = run_benchmark(
        drifts=drifts,
        seeds=seeds,
        kappa=1.0,
        K_budget=4,
        expert_hidden=2,
        gating_hidden=8,
        train_steps=1200,
        test_steps=300,
    )

    print_table(results)

    print()
    print("=" * 60)
    print("STAGE 10 FINAL COMPLETE")
