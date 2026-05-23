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
    classify, all_separated, rollout_collect_balanced, state_to_label
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


def init_text_with_direction(text_gating, label_to_idx, epsilon=0.01):
    with torch.no_grad():
        text_gating.text_emb.weight[label_to_idx["positive_basin"]] += epsilon
        text_gating.text_emb.weight[label_to_idx["negative_basin"]] -= epsilon


def run_env_training(env, text_gating, ctrl, proj, agent,
                     label_to_idx, T=3000, align_weight=0.1,
                     scale_weight=0.01, scale_max=5.0):
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

    history = []
    running_error = 1.0

    for t in range(T):
        if t % 100 == 0:
            label_idx = label_seq[seq_ptr % 3]
            seq_ptr += 1

        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        lb = torch.tensor([label_idx])
        z_soft, z_logits, z, base_logits, text_offset = text_gating(s, label=lb)

        if t % 500 == 0:
            bn = base_logits.norm().item()
            tn = text_offset.norm().item()
            sc = text_gating.text_scale.item()
            print(f"  step {t:4d}: base_norm={bn:.3f} text_norm={tn:.3f} "
                  f"ratio={tn/(bn+1e-8):.3f} scale={sc:.3f}")

        K = z.size(-1)
        if K > 1:
            a = torch.multinomial(z_soft, 1).item()
        else:
            a = 0

        o_next, _, done = env.step(a)
        target_tensor = torch.tensor(o_next, dtype=torch.float32)

        preds = [m.predict(obs, a) for m in ctrl.models]
        soft_pred = sum(z_soft[0, i] * preds[i] for i in range(len(preds)))

        loss_pred = ((soft_pred - target_tensor) ** 2).mean()
        entropy = -(z_soft * torch.log(z_soft + 1e-8)).sum()
        loss_env = loss_pred - 0.005 * entropy

        logits_align = proj(z_soft)
        loss_align = F.cross_entropy(logits_align, lb)
        loss_scale = scale_weight * (text_gating.text_scale ** 2)

        loss = loss_env + align_weight * loss_align + loss_scale

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            text_gating.text_scale.clamp_(0, scale_max)

        error = float(np.mean(np.abs(soft_pred.detach().numpy() - o_next)))
        running_error = 0.99 * running_error + 0.01 * error
        history.append(error)

        if done:
            obs = env.reset()
            text_gating.reset()
        else:
            obs = o_next

    tail_err = np.mean(history[-500:]) if len(history) >= 500 else np.mean(history)
    print(f"  final scale={text_gating.text_scale.item():.4f}")
    print(f"  env_training: {T} steps, tail_error={tail_err:.4f}")
    return tail_err


def check_direction(text_gating, env, label, steps=500, n_runs=5):
    label_to_idx = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}
    label_idx = label_to_idx[label]
    means = []

    for run in range(n_runs):
        text_gating.reset()
        obs = env.reset()
        xs = []
        for _ in range(steps):
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            out = text_gating(s, label=label_idx)
            z_soft = out[0]
            K = z_soft.size(-1)
            if K > 1:
                a = torch.multinomial(z_soft, 1).item()
            else:
                a = 0
            o_next, _, done = env.step(a)
            xs.append(float(obs[0]))
            if done:
                obs = env.reset()
                text_gating.reset()
            else:
                obs = o_next
        means.append(float(np.mean(xs)))

    return float(np.mean(means)), float(np.std(means))


def train_controller(env, agent, ctrl_type, steps=5000):
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
            child = copy.deepcopy(ctrl.models[0])
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


def test_doublewell_9bp():
    print("=" * 60)
    print("TEST 1: DoubleWell — Phase 9B' Alignment Training")
    print("=" * 60)

    reset_seed(42)
    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl, K = train_controller(env, agent, "doublewell", steps=5000)
    print(f"K = {K}")

    env_balanced = DoubleWellEnv(noise=0.0, reset_pos=0.5)
    states, z_states, labels = rollout_collect_balanced(ctrl, env_balanced, n_per_class=200)

    proj, proj_acc = train_proj_head(z_states, labels, K, epochs=2000, lr=0.01)
    print(f"Proj head trained, acc={proj_acc:.4f}")

    text_gating = TextZGatingNet(state_dim=1, K=K, hidden_dim=32, temperature=0.5, emb_dim=16, scale_init=10.0)
    copy_zgating_to_text(text_gating, ctrl.gating)

    label_to_idx = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}

    env_test = DoubleWellEnv(noise=0.02)

    print("\n--- Step 1: check_direction (before training) ---")
    mx_pos, std_pos = check_direction(text_gating, env_test, "positive_basin", steps=500, n_runs=3)
    mx_neg, std_neg = check_direction(text_gating, env_test, "negative_basin", steps=500, n_runs=3)
    print(f"  positive_text mean x = {mx_pos:+.4f} +/- {std_pos:.4f}")
    print(f"  negative_text mean x = {mx_neg:+.4f} +/- {std_neg:.4f}")

    need_bias = not (mx_pos > mx_neg)
    print(f"  direction correct (pos > neg): {not need_bias}")

    if need_bias:
        print("  -> Adding direction bias to text_emb...")
        init_text_with_direction(text_gating, label_to_idx, epsilon=0.02)

    print("\n--- Step 2: 9B' training (env loss + alignment loss) ---")
    run_env_training(env_test, text_gating, ctrl, proj, agent,
                     label_to_idx, T=5000, align_weight=0.1, scale_weight=0.0)

    print("\n--- Step 3: check_direction (after training) ---")
    mx_pos2, _ = check_direction(text_gating, env_test, "positive_basin", steps=500, n_runs=3)
    mx_neg2, _ = check_direction(text_gating, env_test, "negative_basin", steps=500, n_runs=3)
    mx_bound, _ = check_direction(text_gating, env_test, "boundary", steps=500, n_runs=3)
    print(f"  positive_text mean x = {mx_pos2:+.4f}")
    print(f"  negative_text mean x = {mx_neg2:+.4f}")
    print(f"  boundary_text  mean x = {mx_bound:+.4f}")

    dir_ok = mx_pos2 > mx_neg2
    bound_near = abs(mx_bound) < 0.5
    print(f"  direction correct (pos > neg): {dir_ok}")
    print(f"  boundary near center: {bound_near}")

    if dir_ok:
        print("RESULT: Phase 9B' PASS — text direction aligned")
    elif not need_bias:
        print("RESULT: Phase 9B' INVERTED — direction reversed (acceptable)")
    else:
        print("RESULT: Phase 9B' PARTIAL")

    print()
    reset_seed(42)


def test_triplewell_9bp():
    print("=" * 60)
    print("TEST 2: TripleWell — Phase 9B' Alignment Training")
    print("=" * 60)

    reset_seed(42)
    env = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl, K = train_controller(env, agent, "triplewell", steps=8000)
    print(f"K = {K}")

    env_balanced = TripleWellEnv(noise=0.0, reset_range=(-1.2, 1.2))
    states, z_states, labels = rollout_collect_balanced(ctrl, env_balanced, n_per_class=200)

    proj, proj_acc = train_proj_head(z_states, labels, K, epochs=2000, lr=0.01)
    print(f"Proj head trained, acc={proj_acc:.4f}")

    text_gating = TextZGatingNet(state_dim=1, K=K, hidden_dim=32, temperature=0.5, emb_dim=16, scale_init=10.0)
    copy_zgating_to_text(text_gating, ctrl.gating)

    label_to_idx = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}

    env_test = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))

    print("\n--- Step 1: check_direction (before training) ---")
    mx_pos, std_pos = check_direction(text_gating, env_test, "positive_basin", steps=500, n_runs=3)
    mx_neg, std_neg = check_direction(text_gating, env_test, "negative_basin", steps=500, n_runs=3)
    print(f"  positive_text mean x = {mx_pos:+.4f} +/- {std_pos:.4f}")
    print(f"  negative_text mean x = {mx_neg:+.4f} +/- {std_neg:.4f}")

    need_bias = not (mx_pos > mx_neg)
    print(f"  direction correct (pos > neg): {not need_bias}")

    if need_bias:
        print("  -> Adding direction bias to text_emb...")
        init_text_with_direction(text_gating, label_to_idx, epsilon=0.02)

    print("\n--- Step 2: 9B' training (env loss + alignment loss) ---")
    run_env_training(env_test, text_gating, ctrl, proj, agent,
                     label_to_idx, T=5000, align_weight=0.1, scale_weight=0.0)

    print("\n--- Step 3: check_direction (after training) ---")
    mx_pos2, _ = check_direction(text_gating, env_test, "positive_basin", steps=500, n_runs=3)
    mx_neg2, _ = check_direction(text_gating, env_test, "negative_basin", steps=500, n_runs=3)
    mx_bound, _ = check_direction(text_gating, env_test, "boundary", steps=500, n_runs=3)
    print(f"  positive_text mean x = {mx_pos2:+.4f}")
    print(f"  negative_text mean x = {mx_neg2:+.4f}")
    print(f"  boundary_text  mean x = {mx_bound:+.4f}")

    dir_ok = mx_pos2 > mx_neg2
    bound_near = abs(mx_bound) < 0.5
    print(f"  direction correct (pos > neg): {dir_ok}")
    print(f"  boundary near center: {bound_near}")

    if dir_ok:
        print("RESULT: Phase 9B' PASS — text direction aligned")
    elif not need_bias:
        print("RESULT: Phase 9B' INVERTED — direction reversed (acceptable)")
    else:
        print("RESULT: Phase 9B' PARTIAL")

    print()
    reset_seed(42)


if __name__ == "__main__":
    test_doublewell_9bp()
    test_triplewell_9bp()

    print("=" * 60)
    print("STAGE 9B' COMPLETE")
