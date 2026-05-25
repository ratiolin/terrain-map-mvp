import numpy as np


def compute_stability(prediction_variance, routing_consistency, dwell_time):
    return (routing_consistency * dwell_time) / (1.0 + prediction_variance)


def compute_dwell_time(state_history):
    if len(state_history) < 2:
        return 1.0
    basins = [1.0 if s > 0 else -1.0 for s in state_history]
    dwells = []
    current = basins[0]
    count = 1
    for i in range(1, len(basins)):
        if basins[i] == current:
            count += 1
        else:
            dwells.append(count)
            current = basins[i]
            count = 1
    dwells.append(count)
    return float(np.mean(dwells))


def compute_prediction_variance(mse_history):
    return float(np.var(mse_history))


def compute_routing_consistency(switch_rate):
    return 1.0 - switch_rate


def exists_window(S_adv_values, baseline="single_expert", threshold=1.0):
    if isinstance(S_adv_values, dict):
        S_adv_values = list(S_adv_values.values())
    if not isinstance(S_adv_values, (list, tuple, np.ndarray)):
        S_adv_values = [S_adv_values]
    return any(s > threshold for s in S_adv_values)


def exists_collapse(eta_values, eta_max):
    if isinstance(eta_values, dict):
        eta_values = list(eta_values.values())
    if not isinstance(eta_values, (list, tuple, np.ndarray)):
        eta_values = [eta_values]
    return any(e > eta_max for e in eta_values)


def compute_success(metrics_dict):
    S_adv_curve = metrics_dict.get("S_adv", [])
    baseline_perf = metrics_dict.get("baseline_mse", 0.0)
    eta_series = metrics_dict.get("eta_series", [])
    eta_max = metrics_dict.get("eta_max", float("inf"))
    stability = metrics_dict.get("stability", 0.0)
    performance = metrics_dict.get("performance", 0.0)

    success = (
        exists_window(S_adv_curve, baseline="single_expert")
        and exists_collapse(eta_series, eta_max)
    )

    return {
        "success": success,
        "S_adv_curve": S_adv_curve,
        "baseline_perf": baseline_perf,
        "eta_series": eta_series,
        "stability": stability,
        "performance": performance,
    }


def check_structure_stable(entropy_history, n_experts_active, threshold=0.01):
    if len(entropy_history) < 100:
        return True
    tail_entropy = np.mean(entropy_history[-100:])
    if tail_entropy < threshold:
        return False
    return True


def classify_structure_type(n_experts, utilization, entropy):
    if n_experts == 1:
        return "single"
    if entropy < 0.01:
        return "single"
    active = sum(1 for u in utilization if u > 0.05)
    if active >= 3:
        return "multi"
    if active == 2:
        return "pattern"
    return "single"


def state_coverage(state_history, n_bins=30, state_range=(-3.0, 3.0)):
    if len(state_history) == 0:
        return 0.0
    bins = np.linspace(state_range[0], state_range[1], n_bins + 1)
    counts, _ = np.histogram(state_history, bins=bins)
    return float(np.sum(counts > 0)) / n_bins


def transition_entropy(state_history, n_bins=20, state_range=(-3.0, 3.0)):
    if len(state_history) < 2:
        return 0.0
    bins = np.linspace(state_range[0], state_range[1], n_bins + 1)
    edges = np.digitize(np.array(state_history[:-1]), bins) - 1
    edges_next = np.digitize(np.array(state_history[1:]), bins) - 1
    edges = np.clip(edges, 0, n_bins - 1)
    edges_next = np.clip(edges_next, 0, n_bins - 1)
    joint = np.zeros((n_bins, n_bins))
    for s, sn in zip(edges, edges_next):
        joint[s, sn] += 1
    joint = joint / max(1.0, joint.sum())
    joint = joint + 1e-10
    joint = joint / joint.sum()
    return float(-np.sum(joint * np.log(joint)))


def visitation_uniformity(state_history, n_bins=30, state_range=(-3.0, 3.0)):
    if len(state_history) == 0:
        return 0.0
    bins = np.linspace(state_range[0], state_range[1], n_bins + 1)
    counts, _ = np.histogram(state_history, bins=bins)
    p = counts / max(1.0, counts.sum())
    p = p + 1e-10
    p = p / p.sum()
    uniform = np.ones(n_bins) / n_bins
    kl = np.sum(p * np.log(p / uniform))
    return float(np.exp(-kl))
