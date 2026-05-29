import json
import os
import numpy as np


class ExperimentLogger:
    """Unified logging interface for all experiments."""

    def __init__(self, save_dir, experiment_name, seed):
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.seed = seed
        self.logs = []

    def log(self, state, action, next_state,
            prediction_error_raw=0.0,
            control_magnitude_raw=0.0,
            panic_signal=0.0,
            mode_label=None):
        entry = {
            "state": state.tolist() if isinstance(state, np.ndarray) else state,
            "action": action.tolist() if isinstance(action, np.ndarray) else action,
            "next_state": next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            "prediction_error_raw": float(prediction_error_raw),
            "control_magnitude_raw": float(control_magnitude_raw),
            "panic_signal": float(panic_signal),
            "mode_label": str(mode_label) if mode_label is not None else None,
        }
        self.logs.append(entry)

    def log_dict(self, d):
        self.logs.append(d)

    def save(self, filename="trajectory.json"):
        path = os.path.join(self.save_dir, f"seed_{self.seed:03d}_{filename}")
        with open(path, "w") as f:
            json.dump(self.logs, f, indent=2)
        return path

    def save_summary(self, data, filename="summary.json"):
        path = os.path.join(self.save_dir, f"seed_{self.seed:03d}_{filename}")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def get_logs(self):
        return self.logs

    def clear(self):
        self.logs = []


class ResultAggregator:
    """Aggregate results across seeds and dimensions."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_layer_result(self, layer_name, data, filename=None):
        if filename is None:
            filename = f"{layer_name}.json"
        path = os.path.join(self.base_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def aggregate_across_seeds(self, results_list):
        """Given list of per-seed result dicts, compute mean ± std."""
        if not results_list:
            return {}
        aggregated = {}
        for key in results_list[0]:
            vals = [float(r[key]) for r in results_list if key in r]
            if vals:
                aggregated[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
        return aggregated
