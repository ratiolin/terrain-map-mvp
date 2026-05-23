import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_policy(env, agent, controller, steps=10000, act_dim=None):
    if not hasattr(controller, 'models') or len(controller.models) == 0:
        controller.init_models(agent)
    gating = controller.gating
    gating_opt = controller.gating_optimizer

    obs = env.reset()
    controller.gating_reset()

    history = []
    reward_history = []
    entropy_history = []
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

        # ---- 1. 策略闭环: pi = z_soft, action = sample(pi) ----
        weights = controller.gating_weights(obs)
        z_soft = weights
        K = len(z_soft)

        if K > 1:
            a = torch.multinomial(z_soft, 1).item()
        else:
            a = 0

        if act_dim is not None:
            a = a % act_dim

        # ---- 2. 接入环境 ----
        o_next, r, done = env.step(a)

        reward_history.append(r)

        # ---- 3. 记录策略熵 ----
        policy_entropy = -(z_soft * torch.log(z_soft + 1e-8)).sum()
        entropy_history.append(float(policy_entropy.item()))

        # ---- 世界模型 (同 run_soft) ----
        target_tensor = torch.tensor(o_next, dtype=torch.float32)
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
            Kz = len(preds)
            with torch.no_grad():
                perr = torch.stack([
                    ((preds[i].detach() - target_tensor) ** 2).mean()
                    for i in range(Kz)
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
            if controller.use_temporal and prev_z is not None:
                if len(weights) == len(prev_z):
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

    # ---- 5. 构造 Phi ----
    tail = min(1000, steps)
    avg_reward = float(np.mean(reward_history[-tail:])) if reward_history else 0.0
    reward_var = float(np.var(reward_history[-tail:])) if reward_history else 0.0
    avg_pred_error = float(np.mean(history[-tail:])) if history else 0.0
    avg_entropy = float(np.mean(entropy_history[-tail:])) if entropy_history else 0.0

    Phi = {
        "avg_reward": avg_reward,
        "reward_variance": reward_var,
        "policy_entropy": avg_entropy,
        "pred_error": avg_pred_error
    }

    return np.array(history), weight_history_all, np.array(reward_history), np.array(entropy_history), Phi
