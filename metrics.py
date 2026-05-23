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
