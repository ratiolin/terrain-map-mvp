import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


def train_closed_loop(model, env, num_episodes=10, episode_length=2000,
                      lr=1e-3, lambda_ctrl=0.1, seed=None, logger=None,
                      exploration_noise=0.03):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_logs = []

    env.set_seed(seed if seed is not None else 0)

    for ep in range(num_episodes):
        state = env.reset()
        for step in range(episode_length):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().numpy()

            noise = np.random.randn(*a_np.shape) * exploration_noise
            a_noisy = a_np + noise

            next_state, risk_actual, _, info = env.step(a_noisy)

            risk_t = torch.tensor([[risk_actual]], dtype=torch.float32)
            pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
            action_loss = torch.mean(action ** 2)
            loss = pred_loss + lambda_ctrl * action_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if logger is not None:
                logger.log(state, a_np, next_state,
                           float(pred_loss.item()), float(action_loss.item()),
                           0.0, getattr(info, 'get', lambda *x: None)('drift', None))

            state = next_state
            all_logs.append({"pred_loss": float(pred_loss.item()),
                             "action_loss": float(action_loss.item())})

    return all_logs


def rollout(env, model, steps=2000, exploration_noise=0.0, seed=None):
    """Generate rollout from a trained model."""
    if seed is not None:
        env.set_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    states, actions, next_states, risks, infos = [], [], [], [], []
    state = env.reset()
    for _ in range(steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            action, h, risk_pred = model(s_t)
        a_np = action.squeeze(0).numpy()

        noise = np.random.randn(*a_np.shape) * exploration_noise
        a_noisy = a_np + noise

        next_state, risk, _, info = env.step(a_noisy)

        states.append(state.copy())
        actions.append(a_np.copy())
        next_states.append(next_state.copy())
        risks.append(risk)
        infos.append(info)

        state = next_state

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "next_states": np.array(next_states),
        "risks": np.array(risks),
        "infos": infos,
    }


def train_offline(model, dataset, num_epochs=200, lr=1e-3, lambda_ctrl=0.1,
                  batch_size=256, seed=None):
    """Train model offline from fixed dataset."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    states = torch.from_numpy(dataset["states"].astype(np.float32))
    next_states = torch.from_numpy(dataset["next_states"].astype(np.float32))
    actions = torch.from_numpy(dataset["actions"].astype(np.float32))

    n = len(states)
    all_logs = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            s_batch = states[idx]
            ns_batch = next_states[idx]

            action, h, risk_pred = model(s_batch)
            risk_t = torch.norm(ns_batch[:, :model.action_dim], dim=-1, keepdim=True)
            pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
            action_loss = torch.mean(action ** 2)
            loss = pred_loss + lambda_ctrl * action_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        all_logs.append({"epoch_loss": epoch_loss / max(n_batches, 1)})

    return all_logs


def compute_cost(env, model, steps=500, n_rollouts=3, seed=None):
    """Compute average cost (risk) over multiple evaluation rollouts."""
    costs = []
    for r in range(n_rollouts):
        rollout_seed = (seed + r * 1000) if seed is not None else (r * 1000)
        data = rollout(env, model, steps=steps, exploration_noise=0.0, seed=rollout_seed)
        costs.append(np.mean(data["risks"]))
    return float(np.mean(costs)), float(np.std(costs))
