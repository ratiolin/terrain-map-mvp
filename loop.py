import numpy as np


def run(env, agent, metrics, controller, steps=10000, hook=None):
    obs = env.reset()

    history = []
    loss = 0.0

    for t in range(steps):

        a = agent.act(obs)
        o_next, r, done = env.step(a)

        pred = agent.predict(obs, a).detach().numpy()
        error = metrics.update(r, pred, o_next)

        if controller.should_update(error):
            loss = agent.update(obs, a, o_next)

        if hook is not None:
            hook(t, env)

        history.append(error)

        obs = o_next if not done else env.reset()

    return np.array(history)
