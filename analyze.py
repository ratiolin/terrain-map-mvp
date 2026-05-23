import numpy as np


def classify(history):
    h = history

    trend = np.polyfit(range(len(h)), h, 1)[0]
    var = np.var(h[-1000:])

    if var > 0.01:
        return "divergent"

    if trend < -1e-6 and var < 0.0001:
        return "convergent"

    if abs(trend) < 1e-6 and var < 0.001:
        return "wrong_convergence"

    if var > 0.0001:
        return "oscillation"

    return "convergent"


def get_metrics(history, label="", rewards=None):
    h = history
    trend = np.polyfit(range(len(h)), h, 1)[0]
    tail_mean = np.mean(h[-1000:])
    tail_var = np.var(h[-1000:])
    max_err = np.max(h)
    max_increase = max(0.0, float(np.diff(h).max()))

    print(f"--- {label} ---")
    print(f"Trend:                {trend:.6f}")
    print(f"Tail Mean Error:      {tail_mean:.6f}")
    print(f"Tail Variance:        {tail_var:.6f}")
    print(f"Max Error:            {max_err:.6f}")
    print(f"Max Step Increase:    {max_increase:.6f}")
    if rewards is not None and len(rewards) > 0:
        print(f"Mean Reward (last 1k):{np.mean(rewards[-1000:]):.4f}")
    print("")


def is_stable(history, var_th=1e-4):
    var = np.var(history[-1000:])
    trend = np.polyfit(range(len(history)), history, 1)[0]
    return var < var_th and abs(trend) < 1e-4


def rollout(agent, env, steps=2000):
    obs = env.reset()
    errors = []
    for _ in range(steps):
        a = agent.act(obs)
        o_next, r, done = env.step(a)
        pred = agent.predict(obs, a).detach().numpy()
        error = np.mean(np.abs(pred - o_next))
        errors.append(error)
        obs = o_next if not done else env.reset()
    return np.array(errors)


def rollout_mean_state(agent, env, steps=500):
    obs = env.reset()
    states = []
    for _ in range(steps):
        a = agent.act(obs)
        o_next, _, done = env.step(a)
        states.append(obs[0])
        obs = o_next if not done else env.reset()
    return np.mean(np.array(states))


def verify_stage3(P1, P2, env, T=2000, var_th=5e-4):
    h1 = rollout(P1, env, T)
    h2 = rollout(P2, env, T)

    s1 = is_stable(h1, var_th=var_th)
    s2 = is_stable(h2, var_th=var_th)

    m1 = rollout_mean_state(P1, env)
    m2 = rollout_mean_state(P2, env)

    test_states = np.linspace(-0.8, 0.8, 100)
    max_diff = 0.0
    for x in test_states:
        obs = np.array([x], dtype=np.float32)
        p1 = P1.predict(obs, 0).detach().numpy()[0]
        p2 = P2.predict(obs, 0).detach().numpy()[0]
        max_diff = max(max_diff, abs(p1 - p2))

    diverged = max_diff > 0.02
    separated = abs(m1 - m2) > 0.2 or max_diff > 0.05

    return s1 and s2 and diverged and separated


def model_distance(model_a, model_b):
    import torch
    vecs = []
    for p in model_a.predictor.parameters():
        vecs.append(p.data.flatten())
    va = torch.cat(vecs)
    vecs = []
    for p in model_b.predictor.parameters():
        vecs.append(p.data.flatten())
    vb = torch.cat(vecs)
    return torch.norm(va - vb).item()


def all_separated(models, min_dist=0.3):
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            if model_distance(models[i], models[j]) < min_dist:
                return False
    return True


def temporal_consistency(weight_history, smooth_thresh=0.001, window=1):
    if len(weight_history) < 2:
        return {"stability": 1.0, "mean_change": 0.0, "max_change": 0.0,
                "n_compared": 0, "n_skipped_k1": 0}

    changes = []
    matches = []
    skipped_k1 = 0
    is_onehot = False

    for t in range(1, len(weight_history)):
        w_prev = np.asarray(weight_history[t - 1])
        w_curr = np.asarray(weight_history[t])
        if len(w_prev) != len(w_curr):
            continue
        if len(w_prev) == 1:
            skipped_k1 += 1
            continue
        if np.max(w_prev) > 0.9 or np.max(w_curr) > 0.9:
            is_onehot = True
        elif len(w_prev) > 1 and np.max(w_prev) > 1.0 / len(w_prev) * 1.1:
            is_onehot = True
        diff = float(np.max(np.abs(w_curr - w_prev)))
        changes.append(diff)
        matches.append(1.0 if np.argmax(w_prev) == np.argmax(w_curr) else 0.0)

    if not changes:
        return {"stability": 1.0, "mean_change": 0.0, "max_change": 0.0,
                "n_compared": 0, "n_skipped_k1": skipped_k1, "is_onehot": False}

    changes = np.array(changes)
    matches = np.array(matches)

    if is_onehot:
        if window > 1 and len(matches) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(matches, kernel, mode='valid')
            stability = float(np.mean(smoothed > 0.5))
        else:
            stability = float(np.mean(matches))
    else:
        if window > 1 and len(changes) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(changes, kernel, mode='valid')
            stable_mask = smoothed < smooth_thresh
        else:
            stable_mask = changes < smooth_thresh
        stability = float(np.mean(stable_mask))

    return {
        "stability": stability,
        "mean_change": float(np.mean(changes)),
        "max_change": float(np.max(changes)),
        "n_compared": len(changes),
        "n_skipped_k1": skipped_k1,
        "is_onehot": is_onehot
    }


def z_separation(z_history, state_history, threshold=0.3):
    if len(z_history) != len(state_history):
        return 0.0

    z_arr = np.asarray(z_history)
    s_arr = np.asarray(state_history)

    pos_mask = s_arr[:, 0] > 0
    neg_mask = ~pos_mask

    if pos_mask.sum() < 10 or neg_mask.sum() < 10:
        return 0.0

    z_pos = z_arr[pos_mask]
    z_neg = z_arr[neg_mask]

    separation = float(np.linalg.norm(z_pos.mean(0) - z_neg.mean(0)))
    return separation


def specialization_score(per_model_errors, z_history):
    if not z_history:
        return 0.0
    final_k = len(z_history[-1])
    mask = [len(z) == final_k for z in z_history]
    z_filtered = [z_history[i] for i in range(len(z_history)) if mask[i]]
    if len(z_filtered) < 100:
        return 0.0

    z_arr = np.asarray(z_filtered)
    n = len(z_filtered)

    errs_filtered = []
    for k in range(final_k):
        if k < len(per_model_errors):
            raw = per_model_errors[k]
            e = [raw[i] for i in range(min(len(raw), len(mask))) if i < len(mask) and mask[i]]
            if len(e) < n:
                e = e + [e[-1]] * (n - len(e)) if e else [0.0] * n
            errs_filtered.append(np.asarray(e[:n]))
        else:
            errs_filtered.append(np.zeros(n))

    winner = z_arr.argmax(axis=1)
    specs = []

    for k in range(final_k):
        own = (winner == k)
        other = (winner != k)
        if own.sum() < 50 or other.sum() < 50:
            specs.append(0.0)
            continue
        errs = errs_filtered[k]
        err_own = errs[own].mean()
        err_other = errs[other].mean()
        spec_k = np.clip(
            np.log(err_other + 1e-8) - np.log(err_own + 1e-8),
            -1.0,
            np.log(20.0)
        )
        specs.append(spec_k)

    return float(np.mean(specs)) if specs else 0.0
