import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env_double_well import DoubleWellEnv, TripleWellEnv
from agent import Agent
from controller import GatingGrowthController
from loop_multi import experiment_soft
from analyze import (
    all_separated, rollout_collect_balanced, state_to_label
)
from gating import TextZGatingNet


def reset_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def copy_zgating_to_text(new_text_gating, old_zgating):
    with torch.no_grad():
        new_text_gating.gru.load_state_dict(old_zgating.gru.state_dict())
        new_text_gating.output.load_state_dict(old_zgating.output.state_dict())
        new_text_gating.direct.load_state_dict(old_zgating.direct.state_dict())


def make_policy_head(state_dim, act_dim=2, hidden_dim=32):
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, act_dim),
    )


def train_policy_heads(env, ctrl, state_dim, T=5000, lr=0.01,
                       entropy_weight=0.01, baseline_decay=0.99):
    K = ctrl.n_models()
    policy_heads = [make_policy_head(state_dim, act_dim=2) for _ in range(K)]
    optimizers = [optim.Adam(ph.parameters(), lr=lr) for ph in policy_heads]

    for m in ctrl.models:
        for p in m.parameters():
            p.requires_grad = False
    for p in ctrl.gating.parameters():
        p.requires_grad = False
    ctrl.freeze_structure = True

    baseline = 0.0
    ctrl.gating_reset()
    obs = env.reset()
    hist_rewards = []

    for t in range(T):
        weights = ctrl.gating_weights(obs)
        expert_k = int(weights.argmax().item())

        s = torch.tensor(obs, dtype=torch.float32)
        logits = policy_heads[expert_k](s)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        o_next, r, done = env.step(int(action.item()))
        hist_rewards.append(r)

        log_prob = dist.log_prob(action)
        advantage = r - baseline
        baseline = baseline_decay * baseline + (1.0 - baseline_decay) * r

        entropy = dist.entropy()
        loss = -log_prob * advantage - entropy_weight * entropy

        optimizers[expert_k].zero_grad()
        loss.backward()
        optimizers[expert_k].step()

        if done:
            obs = env.reset()
            ctrl.gating_reset()
        else:
            obs = o_next

    tail_r = np.mean(hist_rewards[-1000:]) if len(hist_rewards) >= 1000 else np.mean(hist_rewards)
    print(f"  policy training: {T} steps, tail_r={tail_r:.4f}, baseline={baseline:.4f}")
    return policy_heads


def run_policy_rollout(env, ctrl, policy_heads, steps=1000, agent=None):
    ctrl.gating_reset()
    obs = env.reset()
    states = []
    rewards = []

    for _ in range(steps):
        weights = ctrl.gating_weights(obs)
        expert_k = int(weights.argmax().item())
        if policy_heads is not None:
            s = torch.tensor(obs, dtype=torch.float32)
            logits = policy_heads[expert_k](s)
            action = int(logits.argmax().item())
        elif agent is not None:
            action = agent.act(obs)
        else:
            action = 0
        o_next, r, done = env.step(action)
        states.append(float(obs[0]))
        rewards.append(r)
        if done:
            obs = env.reset()
            ctrl.gating_reset()
        else:
            obs = o_next

    labels = [state_to_label(np.array([s])) for s in states]
    n_pos = sum(1 for l in labels if l == "positive_basin")
    n_neg = sum(1 for l in labels if l == "negative_basin")
    n_bound = sum(1 for l in labels if l == "boundary")
    mean_x = float(np.mean(states))
    mean_r = float(np.mean(rewards))
    return mean_x, n_pos, n_neg, n_bound, mean_r


def train_proj_head(z_states, labels, K, epochs=2000, lr=0.01):
    label_to_idx = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}
    proj = nn.Linear(K, 3)
    opt = optim.Adam(proj.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    z_t = torch.tensor(np.array(z_states), dtype=torch.float32)
    y_t = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)
    for epoch in range(epochs):
        logits = proj(z_t)
        loss = ce(logits, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = proj(z_t).argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
    return proj, acc


def run_env_training_9bp(env, text_gating, ctrl, proj, label_to_idx,
                         T=5000, align_weight=0.1, scale_weight=0.0, scale_max=10.0):
    for m in ctrl.models:
        for p in m.parameters():
            p.requires_grad = False
    for p in text_gating.gru.parameters():
        p.requires_grad = False
    for p in text_gating.output.parameters():
        p.requires_grad = False
    for p in text_gating.direct.parameters():
        p.requires_grad = False
    for p in proj.parameters():
        p.requires_grad = False

    opt = optim.Adam([
        {'params': text_gating.text_emb.parameters(), 'lr': 0.01},
        {'params': text_gating.text_proj.parameters(), 'lr': 0.01},
        {'params': [text_gating.text_scale], 'lr': 0.01},
    ])

    text_gating.reset()
    obs = env.reset()
    label_idx = 0
    label_seq = [0, 1, 2]
    seq_ptr = 0

    for t in range(T):
        if t % 100 == 0:
            label_idx = label_seq[seq_ptr % 3]
            seq_ptr += 1

        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        lb = torch.tensor([label_idx])
        z_soft, z_logits, z, base_logits, text_offset = text_gating(s, label=lb)

        if t % 1000 == 0:
            bn = base_logits.norm().item()
            tn = text_offset.norm().item()
            print(f"  step {t:4d}: base_norm={bn:.3f} text_norm={tn:.3f} "
                  f"ratio={tn/(bn+1e-8):.3f} scale={text_gating.text_scale.item():.3f}")

        o_next, _, done = env.step(0)
        target_tensor = torch.tensor(o_next, dtype=torch.float32)

        preds = [m.predict(obs, 0) for m in ctrl.models]
        soft_pred = sum(z_soft[0, i] * preds[i] for i in range(len(preds)))

        loss_pred = ((soft_pred - target_tensor) ** 2).mean()
        entropy_term = -(z_soft * torch.log(z_soft + 1e-8)).sum()
        loss_env = loss_pred - 0.005 * entropy_term

        logits_align = proj(z_soft)
        loss_align = F.cross_entropy(logits_align, lb)
        loss_scale = scale_weight * (text_gating.text_scale ** 2)

        loss = loss_env + align_weight * loss_align + loss_scale

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            text_gating.text_scale.clamp_(0, scale_max)

        if done:
            obs = env.reset()
            text_gating.reset()
        else:
            obs = o_next

    print(f"  final scale={text_gating.text_scale.item():.4f}")


def run_text_policy_rollout(env, ctrl, text_gating, policy_heads, label, steps=1000):
    label_to_idx = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}
    label_idx = label_to_idx[label]

    text_gating.reset()
    ctrl.gating_reset()
    obs = env.reset()
    states = []

    for _ in range(steps):
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        z_soft, _, _, _, _ = text_gating(s, label=label_idx)
        expert_k = int(z_soft.argmax(dim=-1).item())
        s_in = torch.tensor(obs, dtype=torch.float32)
        logits = policy_heads[expert_k](s_in)
        action = int(logits.argmax().item())
        o_next, _, done = env.step(action)
        states.append(float(obs[0]))
        if done:
            obs = env.reset()
            text_gating.reset()
            ctrl.gating_reset()
        else:
            obs = o_next

    mean_x = float(np.mean(states))
    labels_arr = [state_to_label(np.array([s])) for s in states]
    n_pos = sum(1 for l in labels_arr if l == "positive_basin")
    n_neg = sum(1 for l in labels_arr if l == "negative_basin")
    n_bound = sum(1 for l in labels_arr if l == "boundary")
    return mean_x, n_pos, n_neg, n_bound


def train_controller(env, agent, ctrl_type, steps=5000):
    import copy as cp
    ctrl = GatingGrowthController(check_interval=200, env_type=ctrl_type,
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=8 if ctrl_type == "doublewell" else 10,
                                  use_z=True)
    experiment_soft(env, agent, ctrl, steps=steps)
    K = ctrl.n_models()
    if K < 2:
        ctrl.env_type = ctrl_type
        ctrl.models = []
        ctrl.init_models(agent)
        n_extra = 3 if ctrl_type == "doublewell" else 4
        for _ in range(n_extra):
            child = cp.deepcopy(ctrl.models[0])
            child.optimizer = optim.Adam(child.parameters(), lr=1e-3)
            with torch.no_grad():
                for p in child.predictor.parameters():
                    p.add_(torch.randn_like(p) * 0.3)
            ctrl.models.append(child)
            ctrl.gating.expand()
            ctrl.gating_optimizer = optim.Adam(ctrl.gating.parameters(), lr=1e-3)
        ctrl.usage = [0] * len(ctrl.models)
        ctrl.errors = [[] for _ in range(len(ctrl.models))]
        ctrl.birth_step = [0] * len(ctrl.models)
        ctrl.freeze_structure = True
        for _ in range(6):
            obs = env.reset()
            ctrl.gating_reset()
            for __ in range(500):
                w = ctrl.gating_weights(obs)
                a = agent.act(obs)
                o_next, _, done = env.step(a)
                preds = [m.predict(obs, a) for m in ctrl.models]
                sp = sum(w[i] * preds[i] for i in range(len(preds)))
                target = torch.tensor(o_next, dtype=torch.float32)
                loss_pred = ((sp - target) ** 2).mean()
                entropy = -(w * torch.log(w + 1e-8)).sum()
                perr = torch.stack([((preds[i].detach() - target) ** 2).mean() for i in range(len(preds))])
                perr = perr - perr.min()
                zt = torch.softmax(-perr / 0.1, dim=-1)
                zl = F.kl_div(F.log_softmax(ctrl._last_logits, dim=-1), zt, reduction='sum')
                loss = loss_pred - 0.005 * entropy + 0.5 * zl
                ctrl.gating_optimizer.zero_grad()
                for m in ctrl.models:
                    m.optimizer.zero_grad()
                loss.backward()
                ctrl.gating_optimizer.step()
                for m in ctrl.models:
                    m.optimizer.step()
                obs = o_next if not done else env.reset()
        K = ctrl.n_models()
    return ctrl, K


def test_doublewell_9d():
    print("=" * 60)
    print("TEST 1: DoubleWell — Phase 9D Multi-Head Policy")
    print("=" * 60)

    reset_seed(42)
    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl, K = train_controller(env, agent, "doublewell", steps=5000)
    print(f"K = {K}")

    print("\n--- Baseline (before policy training) ---")
    env_test0 = DoubleWellEnv(noise=0.0)
    mx0, p0, n0, b0, r0 = run_policy_rollout(env_test0, ctrl, None, steps=1000, agent=agent)
    print(f"  mean_x={mx0:+.4f}  pos={p0}  neg={n0}  bound={b0}  r={r0:.4f}")

    print("\n--- Step 1-4: Train multi-head policies ---")
    policy_heads = train_policy_heads(env, ctrl, state_dim=1, T=5000)

    print("\n--- After policy training ---")
    env_test1 = DoubleWellEnv(noise=0.0)
    mx1, p1, n1, b1, r1 = run_policy_rollout(env_test1, ctrl, policy_heads, steps=1000)
    print(f"  mean_x={mx1:+.4f}  pos={p1}  neg={n1}  bound={b1}  r={r1:.4f}")

    print("\n--- Step 5-6: Freeze world model, redo 9B' ---")
    env_balanced = DoubleWellEnv(noise=0.0, reset_pos=0.5)
    states, z_states, labels = rollout_collect_balanced(ctrl, env_balanced, n_per_class=200)
    proj, proj_acc = train_proj_head(z_states, labels, K, epochs=2000, lr=0.01)
    print(f"  proj acc = {proj_acc:.4f}")

    text_gating = TextZGatingNet(state_dim=1, K=K, hidden_dim=32,
                                 temperature=0.5, emb_dim=16, scale_init=10.0)
    copy_zgating_to_text(text_gating, ctrl.gating)
    label_to_idx = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}
    env_train = DoubleWellEnv(noise=0.02)
    run_env_training_9bp(env_train, text_gating, ctrl, proj, label_to_idx,
                         T=3000, align_weight=0.1, scale_weight=0.0)

    print("\n--- Step 7: 9C' Verification ---")
    env_v = DoubleWellEnv(noise=0.02)
    results = {}
    for label in ["negative_basin", "boundary", "positive_basin"]:
        mx, p, n, b = run_text_policy_rollout(env_v, ctrl, text_gating,
                                               policy_heads, label, steps=1000)
        results[label] = (mx, p, n, b)
        print(f"  {label:16s}: mean_x={mx:+.4f}  pos={p}  neg={n}  bound={b}")

    pos_mx = results["positive_basin"][0]
    neg_mx = results["negative_basin"][0]
    bound_mx = results["boundary"][0]
    bound_n = results["boundary"][3]

    c1 = pos_mx > 0
    c2 = neg_mx < 0
    c3 = abs(bound_mx) < 0.5 or bound_n > 50

    print(f"\n  [C1] positive → x>0:       {c1} ({pos_mx:+.4f})")
    print(f"  [C2] negative → x<0:       {c2} ({neg_mx:+.4f})")
    print(f"  [C3] boundary → |x|<0.5:    {c3} (|{bound_mx:.4f}|, bound_cnt={bound_n})")

    n_pass = sum([c1, c2, c3])
    print(f"  RESULT: {n_pass}/3 criteria met")

    print()
    reset_seed(42)


def test_triplewell_9d():
    print("=" * 60)
    print("TEST 2: TripleWell — Phase 9D Multi-Head Policy")
    print("=" * 60)

    reset_seed(42)
    env = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl, K = train_controller(env, agent, "triplewell", steps=8000)
    print(f"K = {K}")

    print("\n--- Baseline (before policy training) ---")
    env_test0 = TripleWellEnv(noise=0.0, reset_range=(-1.2, 1.2))
    mx0, p0, n0, b0, r0 = run_policy_rollout(env_test0, ctrl, None, steps=1000, agent=agent)
    print(f"  mean_x={mx0:+.4f}  pos={p0}  neg={n0}  bound={b0}  r={r0:.4f}")

    print("\n--- Step 1-4: Train multi-head policies ---")
    policy_heads = train_policy_heads(env, ctrl, state_dim=1, T=5000)

    print("\n--- After policy training ---")
    env_test1 = TripleWellEnv(noise=0.0, reset_range=(-1.2, 1.2))
    mx1, p1, n1, b1, r1 = run_policy_rollout(env_test1, ctrl, policy_heads, steps=1000)
    print(f"  mean_x={mx1:+.4f}  pos={p1}  neg={n1}  bound={b1}  r={r1:.4f}")

    print("\n--- Step 5-6: Freeze world model, redo 9B' ---")
    env_balanced = TripleWellEnv(noise=0.0, reset_range=(-1.2, 1.2))
    states, z_states, labels = rollout_collect_balanced(ctrl, env_balanced, n_per_class=200)
    proj, proj_acc = train_proj_head(z_states, labels, K, epochs=2000, lr=0.01)
    print(f"  proj acc = {proj_acc:.4f}")

    text_gating = TextZGatingNet(state_dim=1, K=K, hidden_dim=32,
                                 temperature=0.5, emb_dim=16, scale_init=10.0)
    copy_zgating_to_text(text_gating, ctrl.gating)
    label_to_idx = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}
    env_train = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    run_env_training_9bp(env_train, text_gating, ctrl, proj, label_to_idx,
                         T=5000, align_weight=0.1, scale_weight=0.0)

    print("\n--- Step 7: 9C' Verification ---")
    env_v = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    results = {}
    for label in ["negative_basin", "boundary", "positive_basin"]:
        mx, p, n, b = run_text_policy_rollout(env_v, ctrl, text_gating,
                                               policy_heads, label, steps=1000)
        results[label] = (mx, p, n, b)
        print(f"  {label:16s}: mean_x={mx:+.4f}  pos={p}  neg={n}  bound={b}")

    pos_mx = results["positive_basin"][0]
    neg_mx = results["negative_basin"][0]
    bound_mx = results["boundary"][0]
    bound_n = results["boundary"][3]

    c1 = pos_mx > 0
    c2 = neg_mx < 0
    c3 = abs(bound_mx) < 0.5 or bound_n > 50

    print(f"\n  [C1] positive → x>0:       {c1} ({pos_mx:+.4f})")
    print(f"  [C2] negative → x<0:       {c2} ({neg_mx:+.4f})")
    print(f"  [C3] boundary → |x|<0.5:    {c3} (|{bound_mx:.4f}|, bound_cnt={bound_n})")

    n_pass = sum([c1, c2, c3])
    print(f"  RESULT: {n_pass}/3 criteria met")

    print()
    reset_seed(42)


if __name__ == "__main__":
    test_doublewell_9d()
    test_triplewell_9d()

    print("=" * 60)
    print("STAGE 9D COMPLETE")
