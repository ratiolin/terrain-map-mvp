import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import Metrics


def run_soft(env, agent, controller, steps=10000, hook=None):
    controller.init_models(agent)
    gating = controller.gating
    gating_opt = controller.gating_optimizer

    obs = env.reset()
    controller.gating_reset()
    history = []

    z_history = []
    prev_z = None
    weight_history_all = []

    use_z = controller.use_z
    activate_z_loss = False
    running_error = 1.0

    for t in range(steps):
        k_before = len(controller.models)

        controller.maybe_split()
        gating = controller.gating
        gating_opt = controller.gating_optimizer

        if len(controller.models) != k_before:
            prev_z = None

        a = agent.act(obs)
        o_next, r, done = env.step(a)

        target_tensor = torch.tensor(o_next, dtype=torch.float32)
        weights = controller.gating_weights(obs)
        preds = [m.predict(obs, a) for m in controller.models]

        mode = "semi-hard" if (use_z and controller.freeze_structure) else "soft"

        if mode == "hard":
            idx = int(weights.argmax().item())
            soft_pred = preds[idx]
        elif mode == "semi-hard":
            soft_pred = sum(weights[i].detach() * preds[i] for i in range(len(preds)))
        else:
            soft_pred = sum(weights[i] * preds[i] for i in range(len(preds)))

        error = float(np.mean(np.abs(soft_pred.detach().numpy() - o_next)))

        controller.should_update(error, 0)
        controller.record_usage(int(weights.argmax().item()))
        for i in range(len(preds)):
            e_i = float(np.mean(np.abs(preds[i].detach().numpy() - o_next)))
            controller.track_error(i, e_i)

        loss_pred = ((soft_pred - target_tensor) ** 2).mean()
        entropy = -(weights * torch.log(weights + 1e-8)).sum()

        loss = loss_pred - 0.005 * entropy

        if use_z:
            K = len(preds)
            with torch.no_grad():
                perr = torch.stack([
                    ((preds[i].detach() - target_tensor) ** 2).mean()
                    for i in range(K)
                ])
                perr = perr - perr.min()
                z_target_raw = torch.softmax(-perr / 0.05, dim=-1)

            z_logits = controller._last_logits

            z_loss = F.kl_div(
                F.log_softmax(z_logits, dim=-1),
                z_target_raw.detach(),
                reduction='sum'
            )

            temporal_loss_z = torch.tensor(0.0)
            if prev_z is not None and len(weights) == len(prev_z):
                temporal_loss_z = ((weights - prev_z) ** 2).sum()

            running_error = 0.99 * running_error + 0.01 * error
            if not activate_z_loss and running_error < 0.05:
                activate_z_loss = True

            if activate_z_loss:
                if not controller.freeze_structure:
                    loss = loss + 0.5 * z_loss
                else:
                    balance_loss = -(weights * torch.log(weights + 1e-8)).sum()
                    loss = loss + 0.01 * z_loss + 0.005 * balance_loss
                if temporal_loss_z.item() > 0:
                    loss = loss + 0.02 * temporal_loss_z
        else:
            temporal_loss = torch.tensor(0.0)
            logits = getattr(controller, '_last_logits', None)
            if controller.use_temporal and prev_z is not None:
                if weights.shape == controller._last_logits.shape:
                    temporal_loss = ((weights - prev_z) ** 2).sum()
            loss = loss + 0.02 * temporal_loss

        gating_opt.zero_grad()
        for m in controller.models:
            m.optimizer.zero_grad()
        loss.backward()
        gating_opt.step()
        for m in controller.models:
            m.optimizer.step()

        if use_z or controller.use_temporal:
            prev_z = weights.detach().clone()

        z_history.append(weights.detach().numpy().copy())
        weight_history_all.append(weights.detach().numpy().copy())

        if hook is not None:
            hook(t, env)
        history.append(error)

        controller.maybe_merge()
        if len(controller.models) != k_before:
            prev_z = None
            k_before = len(controller.models)
        gating = controller.gating
        gating_opt = controller.gating_optimizer
        controller.maybe_prune()
        if len(controller.models) != k_before:
            prev_z = None
        gating = controller.gating
        gating_opt = controller.gating_optimizer

        if done:
            obs = env.reset()
            controller.gating_reset()
            prev_z = None
        else:
            obs = o_next

    return np.array(history), weight_history_all


def experiment_soft(env, agent, controller, steps=10000, hook=None):
    history, wh = run_soft(env, agent, controller, steps, hook=hook)
    controller._weight_history = wh
    metrics = Metrics()
    return history, metrics


def experiment_growth(env, agent, controller, steps=10000, hook=None):
    from router import route

    controller.init_models(agent)
    obs = env.reset()
    history = []
    prev_obs = None
    prev_action = None
    active_idx = 0

    for t in range(steps):
        controller.maybe_split()
        if t > 0:
            if len(controller.models) > 1 and random.random() < 0.05:
                active_idx = random.randint(0, len(controller.models) - 1)
            else:
                active_idx = route(obs, controller.models, prev_obs, prev_action)
            controller.record_usage(active_idx)
        active = controller.get_active_agent(active_idx)
        a = agent.act(obs)
        o_next, r, done = env.step(a)
        pred = active.predict(obs, a).detach().numpy()
        error = float(np.mean(np.abs(pred - o_next)))
        if controller.should_update(error, active_idx):
            active.update(obs, a, o_next)
        controller.track_error(active_idx, error)
        if hook is not None:
            hook(t, env)
        history.append(error)
        controller.maybe_merge()
        controller.maybe_prune()
        prev_obs = obs
        prev_action = a
        obs = o_next if not done else env.reset()
    return np.array(history)
