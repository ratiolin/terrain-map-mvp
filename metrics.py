import numpy as np


class Metrics:
    def __init__(self):
        self.pred_errors = []
        self.rewards = []

    def update(self, reward, pred, actual):
        error = np.mean(np.abs(pred - actual))

        self.pred_errors.append(error)
        self.rewards.append(reward)

        return error

    def get_history(self):
        return np.array(self.pred_errors)

    def summary(self):
        return {
            "mean_reward": np.mean(self.rewards),
            "mean_pred_error": np.mean(self.pred_errors)
        }


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


def compute_routing_response_time(z_history):
    if len(z_history) < 5:
        return 1.0
    z_argmax = [int(np.argmax(z)) if len(z) > 0 else 0 for z in z_history]

    recovery_times = []
    for t in range(1, len(z_argmax)):
        if z_argmax[t] != z_argmax[t - 1]:
            for end in range(t + 3, min(t + 100, len(z_argmax))):
                if all(z_argmax[end - i] == z_argmax[end]
                       for i in range(3)):
                    recovery_times.append(end - t)
                    break

    if len(recovery_times) == 0:
        return 1.0
    return float(np.mean(recovery_times))


def compute_metrics_from_seed(seed_result):
    prediction_variance = compute_prediction_variance(seed_result["mse_test_history"])
    routing_consistency = compute_routing_consistency(seed_result["switch_rate"])
    dwell_time = compute_dwell_time(seed_result["state_history"])
    response_time = compute_routing_response_time(seed_result["z_history"])
    return {
        "variance": prediction_variance,
        "consistency": routing_consistency,
        "dwell_time": dwell_time,
        "response_time": response_time,
    }
