import numpy as np


def route(state, models, prev_state=None, prev_action=None):
    if len(models) == 1:
        return 0

    if prev_state is None:
        deltas = []
        obs = np.asarray(state, dtype=np.float32)
        for m in models:
            pred = m.predict(obs, 0).detach().numpy()
            deltas.append(float(np.mean(np.abs(pred - obs))))
        return int(np.argmin(deltas))

    errors = []
    obs = np.asarray(prev_state, dtype=np.float32)
    act = prev_action if prev_action is not None else 0
    target = np.asarray(state, dtype=np.float32)

    for m in models:
        pred = m.predict(obs, act).detach().numpy()
        err = float(np.mean(np.abs(pred - target)))
        errors.append(err)

    return int(np.argmin(errors))
