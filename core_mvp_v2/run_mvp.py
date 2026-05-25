import random
import json
import pickle
import copy
import numpy as np
import torch

from core_mvp_v2.env import DriftingDoubleWell
from core_mvp_v2.agent import Expert
from core_mvp_v2.controller import Controller
from core_mvp_v2.metrics import (compute_stability, compute_dwell_time,
                                  compute_prediction_variance, compute_success,
                                  check_structure_stable, classify_structure_type,
                                  state_coverage, transition_entropy, visitation_uniformity)


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def routing_entropy(z):
    if z.dim() == 2:
        z = z.squeeze(0)
    z_np = z.detach().numpy()
    z_np = np.clip(z_np, 1e-10, 1.0)
    return float(-np.sum(z_np * np.log(z_np)))


def routing_entropy_grad(z):
    if z.dim() == 2:
        z = z.squeeze(0)
    z_clamped = z.clamp(1e-10, 1.0)
    return -(z_clamped * z_clamped.log()).sum()


def expert_utilization(z_history, K):
    counts = np.zeros(K)
    for z in z_history:
        if z.shape[-1] >= 1:
            k = int(np.argmax(z))
            if k < K:
                counts[k] += 1
    total = max(1, counts.sum())
    return (counts / total).tolist()


def specialization_index(preds, target):
    if len(preds) < 2:
        return 0.0
    pred_vals = np.array([float(p.item()) for p in preds])
    pairwise_dists = []
    for i in range(len(pred_vals)):
        for j in range(i + 1, len(pred_vals)):
            pairwise_dists.append(abs(pred_vals[i] - pred_vals[j]))
    if not pairwise_dists:
        return 0.0
    return float(np.mean(pairwise_dists) / max(0.001, abs(float(target.mean().item()))))


def get_action(mode):
    if mode == "zero":
        return 0
    elif mode == "random_normal":
        return float(np.random.randn())
    else:
        return random.randint(0, 1)


def evaluate_single(model, env, steps, action_mode="model"):
    obs = env.reset()
    mse_history = []
    state_history = []
    for _ in range(steps):
        action = get_action(action_mode)
        o_next, _, _ = env.step(action)
        with torch.no_grad():
            pred = model(obs, action)
        target = torch.tensor(o_next, dtype=torch.float32)
        loss = ((pred - target) ** 2).mean().item()
        mse_history.append(loss)
        state_history.append(float(o_next[0]))
        obs = o_next
    return mse_history, state_history


def train_single(model, env, steps, action_mode="model"):
    obs = env.reset()
    mse_history = []
    for _ in range(steps):
        action = get_action(action_mode)
        o_next, _, _ = env.step(action)
        pred = model(obs, action)
        target = torch.tensor(o_next, dtype=torch.float32)
        loss = ((pred - target) ** 2).mean()
        if action_mode != "frozen_policy":
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
        mse_history.append(loss.item())
        obs = o_next
    return mse_history


def train_multi(env, ctrl, steps, eta=0.5, structure_beta=0.0, regime_visible=False, action_mode="model"):
    ctrl.set_eta(eta)
    ctrl.reset()
    obs = env.reset()
    mse_history = []
    oracle_history = []
    state_history = []
    z_history = []
    ent_history = []
    spec_history = []
    gain_history = []

    for t in range(steps):
        ctrl.maybe_split()

        action = get_action(action_mode)
        o_next, _, _ = env.step(action)

        target = torch.tensor(o_next, dtype=torch.float32)
        z, _ = ctrl.route(obs)
        K_cur = z.size(-1)
        weights = z.detach()

        preds = []
        for m in ctrl.models:
            with torch.no_grad():
                p = m(obs, action)
            preds.append(p)

        soft_pred = sum(float(weights[0, i]) * preds[i] for i in range(K_cur))

        ent = routing_entropy(weights)
        ent_history.append(ent)
        spec_history.append(specialization_index(preds, target))

        error = abs(soft_pred - float(o_next[0]))
        ctrl.record_usage(int(weights.argmax().item()))
        for i in range(K_cur):
            e_i = abs(float(preds[i].item()) - float(o_next[0]))
            ctrl.track_error(i, e_i)

        perr = torch.stack([
            ((preds[i].detach() - target) ** 2).mean()
            for i in range(K_cur)
        ])
        oracle_err = perr.min().item()

        if action_mode != "frozen_policy":
            ctrl.gating_optimizer.zero_grad()
            for m in ctrl.models:
                m.optimizer.zero_grad()

        weight_vals = [float(weights[0, i].item()) for i in range(K_cur)]

        if regime_visible and K_cur >= 2:
            sign = env.sign
            if sign > 0:
                weight_vals_biased = [1.0 if i == 0 else 0.0 for i in range(K_cur)]
            else:
                weight_vals_biased = [1.0 if i == 1 else 0.0 for i in range(K_cur)]
        else:
            weight_vals_biased = weight_vals

        task_loss = 0.0
        single_losses = []
        for i in range(K_cur):
            pred_i = ctrl.models[i](obs, action)
            single_loss = ((pred_i - target) ** 2).mean()
            single_losses.append(single_loss.detach())
            task_loss += weight_vals_biased[i] * single_loss

        single_best = torch.stack(single_losses).min()
        structure_gain = torch.relu(single_best - task_loss.detach())
        gain_history.append(float(structure_gain.item()))

        beta_eff = max(0.0, structure_beta)
        loss_val = task_loss - beta_eff * structure_gain

        if action_mode != "frozen_policy":
            loss_val.backward()
            ctrl.gating_optimizer.step()
            for m in ctrl.models:
                m.optimizer.step()

        drift = env.drift if env.t > 0 else 0.0
        advantage = float(oracle_err - float(task_loss.item()))
        ctrl.step_record(drift, advantage)

        mse_history.append(float(task_loss.item()))
        oracle_history.append(oracle_err)
        state_history.append(float(o_next[0]))
        z_history.append(weights.detach().numpy().copy())

        ctrl.maybe_merge()
        ctrl.maybe_prune()

        obs = o_next

    return mse_history, oracle_history, state_history, z_history, ent_history, spec_history, gain_history


def compute_S_adv(ctrl, env_single, seed, train_steps=1200, test_steps=300, eta=0.5,
                  drift_rate=0.02, flip_period=500, structure_beta=0.0,
                  coupling_mode="drift", coupling_beta=0.8, coupling_gamma=0.5,
                  regime_visible=False, kappa=4.0, action_mode="model"):
    obs_dim = 2 if regime_visible else 1
    env_kw = dict(kappa=kappa, drift_rate=drift_rate, flip_period=flip_period,
                  coupling_mode=coupling_mode, coupling_beta=coupling_beta,
                  coupling_gamma=coupling_gamma, regime_visible=regime_visible)
    reset_seed(seed)
    env = DriftingDoubleWell(**env_kw)
    env_s = DriftingDoubleWell(**env_kw)
    reset_seed(seed)

    single_model = Expert(obs_dim=obs_dim, hidden_dim=4)
    train_single(single_model, env_s, train_steps, action_mode=action_mode)

    reset_seed(seed + 10000)
    env_test_s = DriftingDoubleWell(**env_kw)
    mse_s, st_s = evaluate_single(single_model, env_test_s, test_steps, action_mode=action_mode)
    baseline_mse = float(np.mean(mse_s))

    train_mse, _, st_m, z_m, ent_train, spec_train, gain_train = train_multi(
        env, ctrl, train_steps, eta=eta, structure_beta=structure_beta,
        regime_visible=regime_visible, action_mode=action_mode)

    ctrl.reset()
    reset_seed(seed + 10000)
    env_test = DriftingDoubleWell(**env_kw)
    obs = env_test.reset()
    mse_test = []
    st_test = []
    ent_test = []
    spec_test = []
    single_best_errors = []
    for _ in range(test_steps):
        action = get_action(action_mode)
        o_next, _, _ = env_test.step(action)
        target = torch.tensor(o_next, dtype=torch.float32)
        z, _ = ctrl.route(obs)
        K_cur = z.size(-1)
        z_detached = z.detach()
        preds = [ctrl.models[i](obs, action) for i in range(K_cur)]
        soft_pred = sum(float(z_detached[0, i]) * preds[i] for i in range(K_cur))
        loss = ((soft_pred - target) ** 2).mean().item()
        mse_test.append(loss)
        st_test.append(float(o_next[0]))
        ent_test.append(routing_entropy(z_detached))
        spec_test.append(specialization_index(preds, target))

        perr = [((preds[i].detach() - target) ** 2).mean().item() for i in range(K_cur)]
        single_best_errors.append(min(perr))
        obs = o_next

    var_m = compute_prediction_variance(mse_test)
    dwell_m = compute_dwell_time(st_test)
    S_multi = compute_stability(var_m, 1.0, dwell_m)

    var_s = compute_prediction_variance(mse_s)
    dwell_s = compute_dwell_time(st_s)
    S_single = compute_stability(var_s, 1.0, dwell_s)

    posthoc_gain = max(0.0, baseline_mse - float(np.mean(mse_test)))

    S_adv = S_multi / (S_single + 1e-8)
    log = ctrl.log()

    structure_type = classify_structure_type(
        ctrl.n_models(), expert_utilization(z_m, ctrl.n_models()),
        float(np.mean(ent_test)))

    return {
        "S_adv": S_adv,
        "S_multi": S_multi,
        "S_single": S_single,
        "baseline_mse": baseline_mse,
        "multi_mse": float(np.mean(mse_test)),
        "eta_series": log["eta"],
        "eta_max": eta * 3.0,
        "stability": S_multi,
        "performance": 1.0 / max(0.001, float(np.mean(mse_test))),
        "log": log,
        "structure": {
            "routing_entropy_mean_train": float(np.mean(ent_train)),
            "routing_entropy_mean_test": float(np.mean(ent_test)),
            "routing_entropy_test": ent_test,
            "specialization_mean_train": float(np.mean(spec_train)),
            "specialization_mean_test": float(np.mean(spec_test)),
            "specialization_test": spec_test,
            "expert_utilization": expert_utilization(z_m, ctrl.n_models()),
            "n_experts": ctrl.n_models(),
            "structure_type": structure_type,
            "stable": check_structure_stable(ent_test, ctrl.n_models(), threshold=0.01),
            "structure_gain_mean": float(np.mean(gain_train)),
            "structure_gain_posthoc": posthoc_gain,
            "coverage": {
                "state_coverage": state_coverage(st_test),
                "transition_entropy": transition_entropy(st_test),
                "visitation_uniformity": visitation_uniformity(st_test),
            },
        },
    }


def run_lambda_scan(eta=0.5, drift_rate=0.02, seed=0, regime_visible=False, action_mode="model"):
    obs_dim = 2 if regime_visible else 1
    betas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    print(f"\n=== BETA SCAN (structure_gain reward) ===")
    print(f"{'beta':>8} {'S_adv':>8} {'entropy':>8} {'n_exp':>6} {'struct':>10} {'gain':>8} {'stable':>8}")
    print("-" * 68)
    results = []
    for beta in betas:
        reset_seed(seed)
        ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=obs_dim)
        ctrl.init()
        r = compute_S_adv(ctrl, None, seed, eta=eta, drift_rate=drift_rate,
                          structure_beta=beta, regime_visible=regime_visible, action_mode=action_mode)
        s = r["structure"]
        results.append(r)
        print(f"{beta:>8.3f} {r['S_adv']:>8.4f} {s['routing_entropy_mean_test']:>8.4f} "
              f"{s['n_experts']:>6} {s['structure_type']:>10} {s['structure_gain_mean']:>8.4f} "
              f"{str(s['stable']):>8}")
    return results


def run_band_scan(eta=0.5, g_min=0.0, g_max=3.0, n_points=31, seed=0, flip_period=200,
                  structure_beta=0.0, coupling_mode="drift", coupling_beta=0.8,
                  regime_visible=False, action_mode="model"):
    obs_dim = 2 if regime_visible else 1
    g_vals = np.linspace(g_min, g_max, n_points)
    results = []

    for g in g_vals:
        drift = g / eta if g > 1e-8 else 1e-4
        reset_seed(seed)
        ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=obs_dim)
        ctrl.init()
        r = compute_S_adv(ctrl, None, seed, eta=eta, drift_rate=drift, flip_period=flip_period,
                          structure_beta=structure_beta, coupling_mode=coupling_mode,
                          coupling_beta=coupling_beta, regime_visible=regime_visible,
                          action_mode=action_mode)
        s = r["structure"]
        results.append({
            "g": g,
            "drift": drift,
            "S_adv": r["S_adv"],
            "stability": r["stability"],
            "performance": r["performance"],
            "multi_mse": r["multi_mse"],
            "baseline_mse": r["baseline_mse"],
            "entropy": s["routing_entropy_mean_test"],
            "specialization": s["specialization_mean_test"],
            "n_experts": s["n_experts"],
            "expert_util": s["expert_utilization"],
            "structure_type": s["structure_type"],
            "stable": s["stable"],
            "structure_gain": s.get("structure_gain_posthoc", 0.0),
            "state_coverage": s.get("coverage", {}).get("state_coverage", 0.0),
            "transition_entropy": s.get("coverage", {}).get("transition_entropy", 0.0),
            "visitation_uniformity": s.get("coverage", {}).get("visitation_uniformity", 0.0),
        })

    S_advs = np.array([r["S_adv"] for r in results])
    above_one = S_advs > 1.0
    entropies = np.array([r["entropy"] for r in results])

    stable_start = None
    stable_end = None
    for i, a in enumerate(above_one):
        if a and stable_start is None:
            stable_start = g_vals[i]
        if a:
            stable_end = g_vals[i]

    print(f"\nBAND SCAN: g ∈ [{g_min}, {g_max}], eta={eta}, β={structure_beta}")
    print(f"{'g':>8} {'S_adv':>8} {'entropy':>8} {'coverage':>8} {'t_entropy':>10} {'v_unif':>8} {'window'}")
    print("-" * 60)
    for r in results:
        w = "YES" if r["S_adv"] > 1.0 else "---"
        print(f"{r['g']:>8.2f} {r['S_adv']:>8.4f} {r['entropy']:>8.4f} "
              f"{r['state_coverage']:>8.4f} {r['transition_entropy']:>10.4f} "
              f"{r['visitation_uniformity']:>8.4f} {w:>8}")

    if stable_start:
        print(f"\nStable interval (S_adv>1): g ∈ [{stable_start:.2f}, {stable_end:.2f}]")
    else:
        print("\nNo emergence window")

    return results, g_vals, S_advs, entropies


def run_geq(g_target=1.0, seed=0, structure_beta=0.0, coupling_mode="drift", coupling_beta=0.8,
            regime_visible=False, action_mode="model"):
    obs_dim = 2 if regime_visible else 1
    configs = [(0.1, 10.0), (0.2, 5.0), (0.5, 2.0)]
    results = []
    for eta, drift in configs:
        reset_seed(seed)
        ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=obs_dim)
        ctrl.init()
        r = compute_S_adv(ctrl, None, seed, eta=eta, drift_rate=drift,
                          structure_beta=structure_beta,
                          coupling_mode=coupling_mode, coupling_beta=coupling_beta,
                          regime_visible=regime_visible, action_mode=action_mode)
        s = r["structure"]
        results.append({
            "eta": eta, "drift": drift, "g": eta * drift,
            "S_adv": r["S_adv"], "stability": r["stability"],
            "performance": r["performance"], "multi_mse": r["multi_mse"],
            "baseline_mse": r["baseline_mse"],
            "entropy": s["routing_entropy_mean_test"], "n_experts": s["n_experts"],
            "structure_type": s["structure_type"],
            "structure_gain": s.get("structure_gain_posthoc", 0.0),
        })

    print(f"\nG-EQUIVALENCE (g={g_target}, β={structure_beta})")
    print(f"{'eta':>8} {'drift':>8} {'g':>8} {'S_adv':>8} {'entropy':>8} {'n_exp':>6} {'struct':>10}")
    print("-" * 64)
    for r in results:
        print(f"{r['eta']:>8.2f} {r['drift']:>8.2f} {r['g']:>8.2f} {r['S_adv']:>8.4f} "
              f"{r['entropy']:>8.4f} {r['n_experts']:>6} {r['structure_type']:>10}")
    return results


def run_mpv(seed=0, regime_visible=False, action_mode="model"):
    obs_dim = 2 if regime_visible else 1
    reset_seed(seed)
    ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=obs_dim)
    ctrl.init()
    r = compute_S_adv(ctrl, None, seed, regime_visible=regime_visible, action_mode=action_mode)
    s = compute_success(r)
    print(f"MVP: S_adv={r['S_adv']:.4f} entropy={r['structure']['routing_entropy_mean_test']:.4f} "
          f"struct={r['structure']['structure_type']} success={s['success']}")
    return r, s


def run_perturbation(g=0.68, eta=0.5, n_seeds=5, structure_beta=0.0,
                     coupling_mode="drift", coupling_beta=0.8, regime_visible=False,
                     action_mode="model"):
    obs_dim = 2 if regime_visible else 1
    drift = g / eta if g > 1e-8 else 1e-4
    print(f"\nPERTURBATION (g={g}, β={structure_beta})")
    print(f"{'condition':<24} {'S_adv':>8} {'entropy':>8} {'n_exp':>6} {'struct':>10} {'emerge'}")
    print("-" * 66)

    all_results = []
    for s in range(n_seeds):
        reset_seed(s)
        ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=obs_dim)
        ctrl.init()
        r = compute_S_adv(ctrl, None, s, eta=eta, drift_rate=drift,
                          structure_beta=structure_beta,
                          coupling_mode=coupling_mode, coupling_beta=coupling_beta,
                          regime_visible=regime_visible, action_mode=action_mode)
        st = r["structure"]
        tag = f"seed={s}"
        e = "YES" if r["S_adv"] > 1.0 else "---"
        print(f"{tag:<24} {r['S_adv']:>8.4f} {st['routing_entropy_mean_test']:>8.4f} "
              f"{st['n_experts']:>6} {st['structure_type']:>10} {e}")
        all_results.append((tag, r))

    for pert in ["shuffle_experts", "random_gating"]:
        reset_seed(0)
        ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=obs_dim)
        ctrl.init()
        if pert == "shuffle_experts":
            params_list = [[p.data.clone() for p in m.predictor.parameters()] for m in ctrl.models]
            indices = list(range(len(ctrl.models)))
            random.shuffle(indices)
            for i, m in enumerate(ctrl.models):
                src = params_list[indices[i]]
                for p, s in zip(m.predictor.parameters(), src):
                    p.data = s.clone()
        if pert == "random_gating":
            for p in ctrl.gating.parameters():
                p.data = torch.randn_like(p) * 0.1
        r = compute_S_adv(ctrl, None, 0, eta=eta, drift_rate=drift,
                          structure_beta=structure_beta,
                          coupling_mode=coupling_mode, coupling_beta=coupling_beta,
                          regime_visible=regime_visible, action_mode=action_mode)
        st = r["structure"]
        e = "YES" if r["S_adv"] > 1.0 else "---"
        print(f"{pert:<24} {r['S_adv']:>8.4f} {st['routing_entropy_mean_test']:>8.4f} "
              f"{st['n_experts']:>6} {st['structure_type']:>10} {e}")
        all_results.append((pert, r))

    S_advs = [r["S_adv"] for _, r in all_results]
    n_emerge = sum(1 for s in S_advs if s > 1.0)
    robust = n_emerge >= len(S_advs) * 0.6
    print(f"\nEmergence: {n_emerge}/{len(S_advs)} "
          f"S_adv={np.mean(S_advs):.4f}+/-{np.std(S_advs):.4f} "
          f"{'ROBUST' if robust else 'FRAGILE'}")
    return all_results


def run_multistability(g_values, eta=0.5, n_seeds=5, structure_beta=0.0,
                       coupling_mode="drift", coupling_beta=0.8, regime_visible=False,
                       action_mode="model"):
    obs_dim = 2 if regime_visible else 1
    print(f"\n=== MULTI-STABILITY SCAN ===")
    print(f"{'g':>8} {'seed':>6} {'S_adv':>8} {'entropy':>8} {'n_exp':>6} {'struct':>10}")
    print("-" * 56)
    stability_map = {}
    for g in g_values:
        drift = g / eta if g > 1e-8 else 1e-4
        g_results = []
        for s in range(n_seeds):
            reset_seed(s)
            ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=obs_dim)
            ctrl.init()
            r = compute_S_adv(ctrl, None, s, eta=eta, drift_rate=drift,
                              structure_beta=structure_beta,
                              coupling_mode=coupling_mode, coupling_beta=coupling_beta,
                              regime_visible=regime_visible, action_mode=action_mode)
            st = r["structure"]
            print(f"{g:>8.2f} {s:>6} {r['S_adv']:>8.4f} {st['routing_entropy_mean_test']:>8.4f} "
                  f"{st['n_experts']:>6} {st['structure_type']:>10}")
            g_results.append(st["structure_type"])
        stability_map[g] = {
            "types": g_results,
            "unique_types": list(set(g_results)),
            "n_unique": len(set(g_results)),
        }
    return stability_map


def reachable_set_size(g, eta=0.5, n_seeds=20, steps_per=200, n_bins=30):
    drift = g / eta if g > 1e-8 else 1e-4
    env = DriftingDoubleWell(kappa=4.0, drift_rate=drift, flip_period=200,
                             coupling_mode="drift", coupling_beta=0.8, regime_visible=True)
    all_states = []
    for s in range(n_seeds):
        reset_seed(s)
        env.reset()
        x0 = (s / n_seeds) * 6.0 - 3.0
        env.state = np.array([x0], dtype=np.float32)
        for _ in range(steps_per):
            action = random.randint(0, 1)
            o_next, _, _ = env.step(action)
            all_states.append(float(o_next[0]))
    return state_coverage(all_states, n_bins=n_bins)


def escape_rate(g, eta=0.5, n_seeds=10, steps_per=500):
    drift = g / eta if g > 1e-8 else 1e-4
    env = DriftingDoubleWell(kappa=4.0, drift_rate=drift, flip_period=200,
                             coupling_mode="drift", coupling_beta=0.8, regime_visible=True)
    all_crossings = []
    for s in range(n_seeds):
        reset_seed(s)
        env.reset()
        x0 = 1.5 if s < n_seeds / 2 else -1.5
        env.state = np.array([x0], dtype=np.float32)
        prev_side = 1.0 if x0 > 0 else -1.0
        crossings = 0
        for _ in range(steps_per):
            action = random.randint(0, 1)
            o_next, _, _ = env.step(action)
            current_side = 1.0 if o_next[0] > 0 else -1.0
            if current_side != prev_side:
                crossings += 1
                prev_side = current_side
        all_crossings.append(crossings / max(1, steps_per))
    return float(np.mean(all_crossings))


def run_explorability_scan(g_min=0.0, g_max=3.0, n_points=16, eta=0.5):
    g_vals = np.linspace(g_min, g_max, n_points)
    coverages = []
    reachables = []
    escape_rates = []

    print(f"\n=== EXPLORABILITY SCAN: g ∈ [{g_min}, {g_max}] ===")
    print(f"{'g':>8} {'coverage':>10} {'reachable':>10} {'escape_rate':>12}")
    print("-" * 44)

    for g in g_vals:
        drift = g / eta if g > 1e-8 else 1e-4
        reset_seed(0)
        ctrl = Controller(K=4, hidden_dim=4, gating_hidden=8, obs_dim=2)
        ctrl.init()
        r = compute_S_adv(ctrl, None, 0, eta=eta, drift_rate=drift, flip_period=200,
                          structure_beta=0.0, regime_visible=True, action_mode="model")
        cov = r["structure"]["coverage"]["state_coverage"]
        reach = reachable_set_size(g, eta=eta)
        esc = escape_rate(g, eta=eta)
        coverages.append(cov)
        reachables.append(reach)
        escape_rates.append(esc)
        print(f"{g:>8.2f} {cov:>10.4f} {reach:>10.4f} {esc:>12.4f}")

    return g_vals, coverages, reachables, escape_rates
