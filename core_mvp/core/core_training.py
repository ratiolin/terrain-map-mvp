import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import threading
import queue
from core_mvp.core.core_env import MultiModeEnv

torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_closed_loop(model, env, num_episodes=10, episode_length=2000,
                      lr=1e-3, lambda_ctrl=0.1, seed=None, logger=None,
                      exploration_noise=0.03):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_logs = []

    env.set_seed(seed if seed is not None else 0)

    for ep in range(num_episodes):
        state = env.reset()
        for step in range(episode_length):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
            action, h, risk_pred = model(s_t)
            a_np = action.squeeze(0).detach().cpu().numpy()

            noise = np.random.randn(*a_np.shape) * exploration_noise
            a_noisy = a_np + noise

            next_state, risk_actual, _, info = env.step(a_noisy)

            risk_t = torch.tensor([[risk_actual]], dtype=torch.float32, device=DEVICE)
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


def rollout(env, model, steps=2000, exploration_noise=0.0, seed=None,
            intervention=None):
    """Generate rollout from a trained model.

    intervention ∈ {None, 'grad_masked', 'grad_disabled'}
      None          — normal forward pass
      grad_masked   — action.detach() before predictor (cut gradient)
      grad_disabled — torch.no_grad() on entire forward pass
    """
    if seed is not None:
        env.set_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    states, actions, next_states, risks, infos = [], [], [], [], []
    state = env.reset()
    for _ in range(steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)

        if intervention is None:
            with torch.no_grad():
                action, h, risk_pred = model(s_t)
        elif intervention == 'grad_masked':
            with torch.no_grad():
                h = model.encoder(s_t)
                action = model.actor(h)
                action = action.detach()
        elif intervention == 'grad_disabled' or intervention == 'no_grad':
            with torch.no_grad():
                action, h, risk_pred = model(s_t)

        a_np = action.squeeze(0).cpu().numpy()

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


def train_closed_loop_discrete(env, model, num_episodes=50, max_steps_per_ep=200,
                               lr=1e-3, entropy_weight=0.01, seed=None):
    """Train discrete-action ClosedLoopModel with Gumbel-Softmax + entropy."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    env.set_seed(seed if seed is not None else 0)
    all_logs = []
    for ep in range(num_episodes):
        state = env.reset()
        ep_loss = 0.0; ep_steps = 0
        for _ in range(max_steps_per_ep):
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
            action_probs, risk_pred, logits = model(s_t)
            action_idx = int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())
            next_state, risk, done, _ = env.step(action_idx)
            risk_t = torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)
            pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
            entropy = -torch.mean(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1))
            loss = pred_loss + entropy_weight * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += float(loss.item()); ep_steps += 1
            if done:
                break
            state = next_state
        all_logs.append({"ep": ep, "loss": ep_loss / max(ep_steps, 1)})
    return all_logs


def train_parallel(env_fn, model, total_steps, lr=1e-3, lambda_ctrl=0.1,
                   seeds=None, exploration_noise=0.03, mini_batch_size=2048):
    """All-GPU training: model + env dynamics run entirely on GPU."""
    if seeds is None: seeds = [0]
    n_envs = len(seeds)
    torch.manual_seed(seeds[0]); np.random.seed(seeds[0])
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Create envs just for reset — everything else runs on GPU
    envs = [env_fn(seed) for seed in seeds]
    d = envs[0].state_dim; k = envs[0].action_dim
    states = torch.from_numpy(np.stack([e.reset() for e in envs]).astype(np.float32)).to(DEVICE)
    action_norms_log = []
    for _ in range(total_steps):
        action, h, risk_pred = model(states)
        an = action + torch.randn_like(action) * exploration_noise
        ns, rs = MultiModeEnv.step_batch_gpu(states, an, d, k, 'closed')
        loss = nn.functional.mse_loss(risk_pred.squeeze(-1), rs) + lambda_ctrl * (action**2).sum(-1).mean()
        anorms = torch.norm(action, dim=-1)
        small_mask = anorms < 0.01
        if small_mask.any():
            loss = loss + 0.1 * (0.01 - anorms[small_mask]).mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        action_norms_log.extend(float(torch.norm(a).item()) for a in action)
        states = ns
    return np.array(action_norms_log)


def train_parallel_by_d(env_fn, model, total_steps, lr=1e-3, lambda_ctrl=0.1,
                        seeds=None, exploration_noise=0.03):
    """Train one model shared across multiple (d, seed) env pairs.

    env_fn(d, seed) → env instance.
    seeds: list of (d, seed) tuples.
    """
    if seeds is None:
        seeds = [(2, 0)]
    n_envs = len(seeds)

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    envs = [env_fn(d, seed) for d, seed in seeds]
    d_max = max(d for d, _ in seeds)
    for env, (d, s) in zip(envs, seeds):
        env.set_seed(s)

    states = np.stack([e.reset() for e in envs]).astype(np.float32)
    if states.shape[1] < d_max:
        states = np.pad(states, ((0, 0), (0, d_max - states.shape[1])))
    action_norms_log = []

    for _ in range(total_steps):
        s_t = torch.from_numpy(states).to(DEVICE)
        action, h, risk_pred = model(s_t)
        a_np = action.detach().cpu().numpy()

        action_norms_log.extend(float(np.linalg.norm(a)) for a in a_np)

        next_states = np.zeros_like(states)
        risks = np.zeros(n_envs, dtype=np.float32)
        for i, env in enumerate(envs):
            d_i = seeds[i][0]
            an = a_np[i, :d_i] if d_i < a_np.shape[1] else a_np[i]
            an = an + np.random.randn(*an.shape) * exploration_noise
            ns, risk, _, _ = env.step(an)
            next_states[i, :d_i] = ns[:d_i]
            risks[i] = risk

        risk_t = torch.from_numpy(risks).unsqueeze(1).to(DEVICE)
        pred_loss = nn.functional.mse_loss(risk_pred, risk_t)
        action_loss = (action ** 2).sum(dim=-1).mean()
        loss = pred_loss + lambda_ctrl * action_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        states = next_states

    return np.array(action_norms_log)


def _rollout_worker(env_fn, seeds, model_class, model_init_kwargs, data_q, stop,
                    exploration_noise, device_str, rollout_steps=8000):
    """Thread worker: N parallel envs, collects rollout_steps per batch."""
    device = torch.device(device_str)
    torch.backends.cudnn.benchmark = True
    envs = [env_fn(s) for s in seeds]
    for e, s in zip(envs, seeds): e.set_seed(s)
    m = model_class(**model_init_kwargs)
    m.to(device); m.eval()
    n_envs = len(envs)
    st = np.stack([e.reset() for e in envs]).astype(np.float32)
    while not stop.is_set():
        buf_s, buf_a, buf_ns, buf_r = [], [], [], []
        for _ in range(rollout_steps):
            s_t = torch.from_numpy(st).to(device)
            with torch.no_grad():
                a, _, _ = m(s_t)
            an = a.cpu().numpy()
            ns = np.zeros_like(st)
            rs = np.zeros(n_envs, dtype=np.float32)
            for i, e in enumerate(envs):
                an_i = an[i] + np.random.randn(*an[i].shape) * exploration_noise
                nsi, ri, _, _ = e.step(an_i)
                ns[i] = nsi; rs[i] = ri
            buf_s.append(st.copy()); buf_a.append(an.copy())
            buf_ns.append(ns.copy()); buf_r.append(rs.copy())
            st = ns
        try:
            data_q.put_nowait((np.stack(buf_s), np.stack(buf_a), np.stack(buf_ns), np.stack(buf_r)))
        except queue.Full:
            pass
        except queue.Full:
            pass


def train_parallel_async(env_fn, model, total_iters, lr=1e-3, lambda_ctrl=0.1,
                         seeds=None, exploration_noise=0.03,
                         mini_batch_size=512, rollout_steps=8000):
    """Producer-consumer: bg thread runs N envs, batches 8000-step data; main does mini-SGD."""
    if seeds is None: seeds = [0]
    torch.manual_seed(seeds[0]); np.random.seed(seeds[0])
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    kwargs = {k: v for k, v in model.__dict__.items()
              if k in ('state_dim', 'hidden_dim', 'action_dim')}
    n_actions = getattr(model, 'n_actions', None)
    if n_actions is not None: kwargs['n_actions'] = n_actions

    data_q = queue.Queue(maxsize=4)
    stop = threading.Event()
    t = threading.Thread(target=_rollout_worker,
                         args=(env_fn, seeds, type(model), kwargs, data_q, stop,
                               exploration_noise, str(DEVICE), rollout_steps),
                         daemon=True)
    t.start()

    action_norms_log = []; step = 0
    while step < total_iters:
        s_b, a_b, ns_b, r_b = data_q.get()
        n_samples = s_b.shape[0]
        for start in range(0, n_samples, mini_batch_size):
            end = min(start + mini_batch_size, n_samples)
            s_t = torch.from_numpy(s_b[start:end]).to(DEVICE)
            r_t = torch.from_numpy(r_b[start:end]).to(DEVICE)
            h = model.encoder(s_t); action = model.actor(h)
            risk_pred = model.predictor(torch.cat([h, action], dim=-1))
            loss = nn.functional.mse_loss(risk_pred.squeeze(-1), r_t) + lambda_ctrl * (action**2).sum(dim=-1).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        action_norms_log.extend(float(np.linalg.norm(a)) for a in action.detach().cpu().numpy())
        step += 1
    stop.set()
    return np.array(action_norms_log)


def train_latent_shift(env, model, total_steps, lr=1e-3, lambda_action=0.1,
                       blocked=False):
    """Train LatentShiftModel: env returns grad_risk used as hint for next step."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    state = env.reset()
    grad_hint = None
    action_norms = []
    for _ in range(total_steps):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)
        if model.use_grad_hint:
            gh_val = grad_hint if grad_hint is not None else 0.0
            gh_t = torch.tensor([[gh_val]], dtype=torch.float32, device=DEVICE)
            s_in = torch.cat([s_t, gh_t], dim=-1)
        else:
            s_in = s_t
        h_out = model.encoder(s_in)
        action = model.actor(h_out)
        action_norms.append(float(torch.norm(action.detach()).item()))
        if blocked:
            risk_pred = model.predictor(torch.cat([h_out, action.detach()], dim=-1))
        else:
            risk_pred = model.predictor(torch.cat([h_out, action], dim=-1))
        an = action.detach().cpu().numpy().squeeze(0)
        ns, risk, grad_risk, done, _ = env.step(an)
        loss = nn.functional.mse_loss(risk_pred, torch.tensor([[risk]], dtype=torch.float32, device=DEVICE)) + lambda_action * (action**2).sum()
        anorm = torch.norm(action)
        if anorm < 0.01:
            loss = loss + 0.1 * (0.01 - anorm)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        state = ns; grad_hint = grad_risk
        if done: state = env.reset(); grad_hint = None
    return np.array(action_norms)
