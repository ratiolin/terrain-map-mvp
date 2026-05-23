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
from loop_multi import experiment_soft, run_soft_finetune
from analyze import (
    all_separated, rollout_collect_balanced, state_to_label, state_to_basin
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


def train_policy_heads_with_diversity(env, ctrl, state_dim, T=5000, lr=0.01,
                                       entropy_weight=0.01, baseline_decay=0.99,
                                       div_weight_start=0.0, div_weight_end=5.0,
                                       eps_start=0.2, eps_end=0.05,
                                       boundary_weight_start=0.1, boundary_weight_end=1.0):
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
    head_x_sums = [0.0] * K
    head_x_counts = [0] * K
    head_usage = [0] * K

    for t in range(T):
        epsilon = eps_start + (eps_end - eps_start) * (t / T)

        weights = ctrl.gating_weights(obs)

        if random.random() < epsilon:
            expert_k = random.randint(0, K - 1)
            exploring = True
        else:
            expert_k = int(weights.argmax().item())
            exploring = False

        head_usage[expert_k] += 1

        s = torch.tensor(obs, dtype=torch.float32)
        logits = policy_heads[expert_k](s)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        o_next, r, done = env.step(int(action.item()))
        hist_rewards.append(r)

        head_x_sums[expert_k] += float(obs[0])
        head_x_counts[expert_k] += 1

        log_prob = dist.log_prob(action)
        advantage = r - baseline
        baseline = baseline_decay * baseline + (1.0 - baseline_decay) * r

        entropy = dist.entropy()
        reward_loss = -log_prob * advantage

        bias_k = ctrl.region_bias[expert_k] if expert_k < len(ctrl.region_bias) else 0.0
        x_current = float(obs[0])
        intrinsic_bonus = bias_k * x_current
        div_weight = div_weight_start + (div_weight_end - div_weight_start) * (t / T)
        bw = boundary_weight_start + (boundary_weight_end - boundary_weight_start) * (t / T)

        loss = (-log_prob * (advantage + div_weight * intrinsic_bonus - bw * (x_current ** 2))
                - entropy_weight * entropy)

        optimizers[expert_k].zero_grad()
        loss.backward()
        optimizers[expert_k].step()

        if done:
            obs = env.reset()
            ctrl.gating_reset()
        else:
            obs = o_next

    tail_r = np.mean(hist_rewards[-1000:]) if len(hist_rewards) >= 1000 else np.mean(hist_rewards)
    print(f"  policy training: {T} steps, tail_r={tail_r:.4f}")

    min_usage = min(head_usage) if head_usage else 0
    all_trained = min_usage > 50
    print(f"  Head usage: min={min_usage}  all_trained={all_trained}")
    for k in range(K):
        mx = head_x_sums[k] / max(1, head_x_counts[k])
        bias = ctrl.region_bias[k] if k < len(ctrl.region_bias) else 0.0
        n = head_x_counts[k]
        print(f"    head[{k}]: bias={bias:+.3f}  mean_x={mx:+.4f}  count={n}  usage={head_usage[k]}")

    return policy_heads


def run_policy_rollout(env, ctrl, policy_heads, steps=1000):
    ctrl.gating_reset()
    obs = env.reset()
    states = []

    for _ in range(steps):
        weights = ctrl.gating_weights(obs)
        expert_k = int(weights.argmax().item())
        s = torch.tensor(obs, dtype=torch.float32)
        logits = policy_heads[expert_k](s)
        action = int(logits.argmax().item())
        o_next, _, done = env.step(action)
        states.append(float(obs[0]))
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
    return mean_x, n_pos, n_neg, n_bound


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
                         T=3000, align_weight=0.1, scale_weight=0.0, scale_max=10.0):
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


def train_proj_head_3class(z_states, labels, K, epochs=2000, lr=0.01):
    label_to_idx = {"left_basin": 0, "center_basin": 1, "right_basin": 2}
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
    return proj, acc, label_to_idx


def run_text_policy_rollout_3class(env, ctrl, text_gating, policy_heads, label, steps=1000):
    label_to_idx = {"left_basin": 0, "center_basin": 1, "right_basin": 2}
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
    basins = [state_to_basin(np.array([s])) for s in states]
    n_left = sum(1 for b in basins if b == "left_basin")
    n_center = sum(1 for b in basins if b == "center_basin")
    n_right = sum(1 for b in basins if b == "right_basin")
    return mean_x, n_left, n_center, n_right


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
        ctrl.region_bias = [0.0] * len(ctrl.models)
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
    ctrl.init_region_bias()
    run_soft_finetune(env, agent, ctrl, steps=2000, region_bias=ctrl.region_bias)
    print(f"  finetune done, z aligned with region_bias")
    return ctrl, K


def test_doublewell_9e():
    print("=" * 60)
    print("TEST 1: DoubleWell — Phase 9E Region-Biased Policies")
    print("=" * 60)

    reset_seed(42)
    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl, K = train_controller(env, agent, "doublewell", steps=5000)
    print(f"K = {K}")

    print(f"region_bias: {[f'{b:+.3f}' for b in ctrl.region_bias]}")

    print("\n--- Step 1-8: Train multi-head policies with diversity loss ---")
    policy_heads = train_policy_heads_with_diversity(
        env, ctrl, state_dim=1, T=5000,
        div_weight_start=0.0, div_weight_end=5.0)

    print("\n--- Verify head distributions ---")
    env_test = DoubleWellEnv(noise=0.0)
    head_stats = {}
    for k in range(K):
        ctrl.gating_reset()
        obs = env_test.reset()
        xs = []
        for _ in range(500):
            s = torch.tensor(obs, dtype=torch.float32)
            logits = policy_heads[k](s)
            action = int(logits.argmax().item())
            o_next, _, done = env_test.step(action)
            xs.append(float(obs[0]))
            obs = o_next if not done else env_test.reset()
        mean_x = float(np.mean(xs))
        head_stats[k] = mean_x
        print(f"  head[{k}]: bias={ctrl.region_bias[k]:+.3f}  solo_mean_x={mean_x:+.4f}")

    sorted_biases = sorted(range(K), key=lambda k: ctrl.region_bias[k])
    sorted_means = [head_stats[k] for k in sorted_biases]
    coverage_ok = sorted_means[-1] - sorted_means[0] > 0.3
    has_neg = any(v < -0.1 for v in sorted_means)
    has_pos = any(v > 0.1 for v in sorted_means)
    print(f"  coverage spread={sorted_means[-1] - sorted_means[0]:.4f}")
    print(f"  has_neg={has_neg} has_pos={has_pos} coverage_ok={coverage_ok}")

    print("\n--- Step 5-6: Freeze world model, 9B' training ---")
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

    print("\n--- Step 9: 9C' Verification ---")
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


def test_triplewell_9e():
    print("=" * 60)
    print("TEST 2: TripleWell — Phase 9E 3-Class Natural Differentiation")
    print("=" * 60)

    reset_seed(42)
    env = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)

    import copy as cp
    ctrl = GatingGrowthController(check_interval=200, env_type="triplewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=10, use_z=True)
    experiment_soft(env, agent, ctrl, steps=8000)
    K = ctrl.n_models()
    if K < 2:
        ctrl.env_type = "triplewell"
        ctrl.models = []
        ctrl.init_models(agent)
        for _ in range(4):
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
        ctrl.region_bias = [0.0] * len(ctrl.models)
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
    ctrl.init_region_bias()
    print(f"K = {K}")
    print(f"region_bias: {[f'{b:+.3f}' for b in ctrl.region_bias]}")
    print("(region-aligned gating: OFF)")

    print("\n--- Train multi-head policies (ε-greedy + intrinsic bonus) ---")
    policy_heads = train_policy_heads_with_diversity(
        env, ctrl, state_dim=1, T=5000,
        div_weight_start=0.0, div_weight_end=5.0,
        boundary_weight_start=0.1, boundary_weight_end=1.0)

    print("\n--- Solo head distributions ---")
    env_test = TripleWellEnv(noise=0.0, reset_range=(-1.2, 1.2))
    head_stats = {}
    for k in range(K):
        ctrl.gating_reset()
        obs = env_test.reset()
        xs = []
        for _ in range(500):
            s = torch.tensor(obs, dtype=torch.float32)
            logits = policy_heads[k](s)
            action = int(logits.argmax().item())
            o_next, _, done = env_test.step(action)
            xs.append(float(obs[0]))
            obs = o_next if not done else env_test.reset()
        mean_x = float(np.mean(xs))
        head_stats[k] = mean_x
        basin = state_to_basin(np.array([mean_x]))
        print(f"  head[{k}]: bias={ctrl.region_bias[k]:+.3f}  solo_mean_x={mean_x:+.4f}  -> {basin}")

    sorted_biases = sorted(range(K), key=lambda k: ctrl.region_bias[k])
    sorted_means = [head_stats[k] for k in sorted_biases]
    spread = sorted_means[-1] - sorted_means[0] if sorted_means else 0
    has_left = any(v < -0.5 for v in sorted_means)
    has_center = any(-0.5 <= v <= 0.5 for v in sorted_means)
    has_right = any(v > 0.5 for v in sorted_means)
    print(f"  spread={spread:.4f}  left={has_left} center={has_center} right={has_right}")

    print("\n--- 9B' training (3-class labels) ---")
    from analyze import rollout_collect_balanced
    states, z_states, labels_old = rollout_collect_balanced(ctrl, env_test, n_per_class=200)
    labels_tri = [state_to_basin(s) for s in states]
    proj, proj_acc, label_to_idx = train_proj_head_3class(
        [z for z in z_states], labels_tri, K, epochs=2000, lr=0.01)
    print(f"  proj acc = {proj_acc:.4f}")

    text_gating = TextZGatingNet(state_dim=1, K=K, hidden_dim=32,
                                 temperature=0.5, emb_dim=16, scale_init=10.0)
    copy_zgating_to_text(text_gating, ctrl.gating)
    env_train = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    run_env_training_9bp(env_train, text_gating, ctrl, proj, label_to_idx,
                         T=3000, align_weight=0.1, scale_weight=0.0)

    print("\n--- 3-Text Closed-Loop Verification ---")
    env_v = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    results = {}
    for label in ["left_basin", "center_basin", "right_basin"]:
        mx, nl, nc, nr = run_text_policy_rollout_3class(
            env_v, ctrl, text_gating, policy_heads, label, steps=1000)
        results[label] = (mx, nl, nc, nr)
        print(f"  {label:16s}: mean_x={mx:+.4f}  left={nl} center={nc} right={nr}")

    left_mx = results["left_basin"][0]
    center_mx = results["center_basin"][0]
    right_mx = results["right_basin"][0]
    center_cnt = results["center_basin"][2]

    c1 = left_mx < -0.3
    c2 = right_mx > 0.3
    c3 = abs(center_mx) < 0.3 or center_cnt > 100

    print(f"\n  [C1] left_text   → x<-0.3:    {c1} ({left_mx:+.4f})")
    print(f"  [C2] right_text  → x>+0.3:    {c2} ({right_mx:+.4f})")
    print(f"  [C3] center_text → |x|<0.3:    {c3} (|{center_mx:.4f}|, center_cnt={center_cnt})")

    n_pass = sum([c1, c2, c3])
    print(f"  RESULT: {n_pass}/3 criteria met")
    print()
    reset_seed(42)


if __name__ == "__main__":
    test_doublewell_9e()
    test_triplewell_9e()

    print("=" * 60)
    print("STAGE 9E COMPLETE")
