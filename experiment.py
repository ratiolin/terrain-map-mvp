from metrics import Metrics
from loop import run


def experiment(env, agent, controller, steps=10000, hook=None):
    metrics = Metrics()
    history = run(env, agent, metrics, controller, steps, hook=hook)
    return history, metrics
