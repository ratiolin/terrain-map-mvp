from env import Env
from agent import Agent
from controller import Controller, SlowController, FreezeController
from experiment import experiment
from analyze import classify, get_metrics

import random
import numpy as np
import torch

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    obs_dim = 4
    act_dim = 2

    # Experiment A: baseline (no perturbation)
    env_a = Env()
    agent_a = Agent(obs_dim, act_dim)
    controller_a = Controller()
    history_A, metrics_A = experiment(env_a, agent_a, controller_a)

    # Experiment B: slow updates (induce oscillation)
    env_b = Env()
    agent_b = Agent(obs_dim, act_dim)
    history_B, metrics_B = experiment(env_b, agent_b, SlowController())

    # Experiment C: extreme learning rate (induce divergence)
    env_c = Env()
    agent_c = Agent(obs_dim, act_dim, lr=1000.0)
    history_C, metrics_C = experiment(env_c, agent_c, Controller())

    # Experiment D: environmental noise (distribution shift)
    env_d = Env(noise=0.05)
    agent_d = Agent(obs_dim, act_dim)
    history_D, metrics_D = experiment(env_d, agent_d, Controller())

    # Experiment E: mid-run mutation (gravity change + freeze learning at t=5000)
    def gravity_hook(t, env):
        if t == 5000:
            env.env.gravity = 50.0

    env_e = Env()
    agent_e = Agent(obs_dim, act_dim)
    history_E, metrics_E = experiment(env_e, agent_e, FreezeController(freeze_at=4999), hook=gravity_hook)

    before = history_E[:5000]
    after = history_E[5000:]
    rewards_before = metrics_E.rewards[:5000]
    rewards_after = metrics_E.rewards[5000:]

    print("=== Classification ===")
    print("A:", classify(history_A))
    print("B:", classify(history_B))
    print("C:", classify(history_C))
    print("D:", classify(history_D))
    print("E:", classify(history_E))
    print("E before:", classify(before), "| after:", classify(after))

    print("")
    print("=== Detailed Metrics ===")
    get_metrics(history_A, "A — baseline", metrics_A.rewards)
    get_metrics(history_B, "B — slow updates", metrics_B.rewards)
    get_metrics(history_C, "C — extreme lr (divergent)", metrics_C.rewards)
    get_metrics(history_D, "D — environmental noise", metrics_D.rewards)
    get_metrics(history_E, "E — mid-run mutation", metrics_E.rewards)
    print("--- E Split ---")
    get_metrics(before, "E before (steps 0–4999)", rewards_before)
    get_metrics(after, "E after (steps 5000–9999)", rewards_after)
