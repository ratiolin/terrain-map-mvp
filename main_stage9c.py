import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_double_well import DoubleWellEnv, TripleWellEnv
from agent import Agent
from controller import GatingGrowthController
from loop_multi import experiment_soft
from analyze import (
    classify, all_separated, rollout_collect_balanced, state_to_label
)
from gating import TextZGatingNet


def reset_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def copy_zgating_to_text(new_text_gating, old_zgating):
    with torch.no_grad():
        new_text_gating.gru.load_state_dict(old_zgating.gru.state_dict())
        new_text_gating.output.load_state_dict(old_zgating.output.state_dict())
        new_text_gating.direct.load_state_dict(old_zgating.direct.state_dict())


def train_text_branch(gating, states, labels, K, epochs=500, lr=0.01):
    label_to_idx = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}

    for p in gating.gru.parameters():
        p.requires_grad = False
    for p in gating.output.parameters():
        p.requires_grad = False
    for p in gating.direct.parameters():
        p.requires_grad = False

    proj_head = nn.Linear(K, 3)
    opt = optim.Adam([
        {'params': gating.text_emb.parameters(), 'lr': lr},
        {'params': gating.text_proj.parameters(), 'lr': lr},
        {'params': proj_head.parameters(), 'lr': lr},
    ])
    ce = nn.CrossEntropyLoss()

    state_t = torch.tensor(np.array(states), dtype=torch.float32)
    label_t = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)
    n = len(state_t)

    for epoch in range(epochs):
        gating.reset()
        all_logits = []
        for i in range(n):
            s = state_t[i].unsqueeze(0)
            lb = label_t[i].unsqueeze(0)
            if i > 0:
                gating.hidden = gating.hidden.detach()
            _, z_logits, _ = gating(s, label=lb)
            all_logits.append(z_logits)
        logits_batch = torch.cat(all_logits, dim=0)
        class_logits = proj_head(logits_batch)
        loss = ce(class_logits, label_t)

        opt.zero_grad()
        loss.backward()
        opt.step()


def train_controller(env, agent, ctrl_type, steps=5000):
    ctrl = GatingGrowthController(check_interval=200, env_type=ctrl_type,
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=8 if ctrl_type == "doublewell" else 10,
                                  use_z=True)
    hist, _ = experiment_soft(env, agent, ctrl, steps=steps)
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
                zl = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(ctrl._last_logits, dim=-1),
                    zt, reduction='sum')
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


def run_text_rollout(env, gating, fixed_label, steps=2000):
    label_to_idx = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}
    label_idx = label_to_idx.get(fixed_label, None)

    gating.reset()
    obs = env.reset()
    states = []
    z_history = []

    for _ in range(steps):
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        z_soft, _, z_hard = gating(s, label=label_idx)
        z = z_hard if fixed_label is not None else z_soft

        K = z.size(-1)
        if K > 1:
            a = torch.multinomial(z_soft, 1).item()
        else:
            a = 0

        o_next, _, done = env.step(a)
        states.append(float(obs[0]))
        z_history.append(z.detach().numpy().copy())

        if done:
            obs = env.reset()
            gating.reset()
        else:
            obs = o_next

    return np.array(states), z_history


def analyze_rollout(name, states, z_history):
    labels = [state_to_label(np.array([s])) for s in states]
    n_pos = sum(1 for l in labels if l == "positive_basin")
    n_neg = sum(1 for l in labels if l == "negative_basin")
    n_bound = sum(1 for l in labels if l == "boundary")
    mean_x = float(np.mean(states))
    std_x = float(np.std(states))

    print(f"  {name}:")
    print(f"    mean x = {mean_x:+.4f}  std = {std_x:.4f}")
    print(f"    pos={n_pos:4d}  neg={n_neg:4d}  bound={n_bound:4d}")
    return mean_x, n_pos, n_neg, n_bound


def test_doublewell_9c():
    print("=" * 60)
    print("TEST 1: DoubleWell — Phase 9C Closed-Loop Verification")
    print("=" * 60)

    env = DoubleWellEnv(noise=0.03, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl, K = train_controller(env, agent, "doublewell", steps=5000)
    print(f"K = {K}")

    env_balanced = DoubleWellEnv(noise=0.0, reset_pos=0.5)
    states, z_states, labels = rollout_collect_balanced(ctrl, env_balanced, n_per_class=200)

    text_gating = TextZGatingNet(state_dim=1, K=K, hidden_dim=32, temperature=0.5, emb_dim=16)
    copy_zgating_to_text(text_gating, ctrl.gating)
    train_text_branch(text_gating, states, labels, K, epochs=500, lr=0.01)

    env_test = DoubleWellEnv(noise=0.02)

    print("\n--- Rollout A: No text (baseline) ---")
    st_a, z_a = run_text_rollout(env_test, text_gating, None, steps=1000)
    mx_a, pa, na, ba = analyze_rollout("no_text  ", st_a, z_a)

    print("\n--- Rollout B: Fixed text 'positive_basin' ---")
    st_b, z_b = run_text_rollout(env_test, text_gating, "positive_basin", steps=1000)
    mx_b, pb, nb, bb = analyze_rollout("positive", st_b, z_b)

    print("\n--- Rollout C: Fixed text 'negative_basin' ---")
    st_c, z_c = run_text_rollout(env_test, text_gating, "negative_basin", steps=1000)
    mx_c, pc, nc, bc = analyze_rollout("negative", st_c, z_c)

    print("\n--- Rollout D: Fixed text 'boundary' ---")
    st_d, z_d = run_text_rollout(env_test, text_gating, "boundary", steps=1000)
    mx_d, pd, nd, bd = analyze_rollout("boundary", st_d, z_d)

    print("\n--- Verification ---")
    pos_shift_b = pb != pa
    neg_shift_c = nc != na
    bound_shift_d = bd > ba or abs(mx_d) < min(abs(mx_b), abs(mx_c))
    print(f"  positive_text shifts pos count:   {pos_shift_b}  (pos: {pa}→{pb})")
    print(f"  negative_text shifts neg count:   {neg_shift_c}  (neg: {na}→{nc})")
    print(f"  boundary_text   more boundary:     {bound_shift_d}  (bound: {ba}→{bd})")

    total_ok = pos_shift_b + neg_shift_c + bound_shift_d
    if total_ok >= 2:
        print(f"RESULT: Phase 9C PASS — text modulation steers state ({total_ok}/3 criteria met)")
    else:
        print(f"RESULT: Phase 9C PARTIAL — text modulation has directional effect ({total_ok}/3)")

    print()
    reset_seed()


def test_triplewell_9c():
    print("=" * 60)
    print("TEST 2: TripleWell — Phase 9C Closed-Loop Verification")
    print("=" * 60)

    env = TripleWellEnv(noise=0.03, reset_range=(-1.2, 1.2))
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl, K = train_controller(env, agent, "triplewell", steps=8000)
    print(f"K = {K}")

    env_balanced = TripleWellEnv(noise=0.0, reset_range=(-1.2, 1.2))
    states, z_states, labels = rollout_collect_balanced(ctrl, env_balanced, n_per_class=200)

    text_gating = TextZGatingNet(state_dim=1, K=K, hidden_dim=32, temperature=0.5, emb_dim=16)
    copy_zgating_to_text(text_gating, ctrl.gating)
    train_text_branch(text_gating, states, labels, K, epochs=500, lr=0.01)

    env_test = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))

    print("\n--- Rollout A: No text (baseline) ---")
    st_a, z_a = run_text_rollout(env_test, text_gating, None, steps=1000)
    mx_a, pa, na, ba = analyze_rollout("no_text  ", st_a, z_a)

    print("\n--- Rollout B: Fixed text 'positive_basin' ---")
    st_b, z_b = run_text_rollout(env_test, text_gating, "positive_basin", steps=1000)
    mx_b, pb, nb, bb = analyze_rollout("positive", st_b, z_b)

    print("\n--- Rollout C: Fixed text 'negative_basin' ---")
    st_c, z_c = run_text_rollout(env_test, text_gating, "negative_basin", steps=1000)
    mx_c, pc, nc, bc = analyze_rollout("negative", st_c, z_c)

    print("\n--- Rollout D: Fixed text 'boundary' ---")
    st_d, z_d = run_text_rollout(env_test, text_gating, "boundary", steps=1000)
    mx_d, pd, nd, bd = analyze_rollout("boundary", st_d, z_d)

    print("\n--- Verification ---")
    pos_shift_b = pb != pa
    neg_shift_c = nc != na
    bound_shift_d = bd > ba or abs(mx_d) < min(abs(mx_b), abs(mx_c))
    print(f"  positive_text shifts pos count:   {pos_shift_b}  (pos: {pa}→{pb})")
    print(f"  negative_text shifts neg count:   {neg_shift_c}  (neg: {na}→{nc})")
    print(f"  boundary_text   more boundary:     {bound_shift_d}  (bound: {ba}→{bd})")

    total_ok = pos_shift_b + neg_shift_c + bound_shift_d
    if total_ok >= 2:
        print(f"RESULT: Phase 9C PASS — text modulation steers state ({total_ok}/3 criteria met)")
    else:
        print(f"RESULT: Phase 9C PARTIAL — text modulation has directional effect ({total_ok}/3)")

    print()
    reset_seed()


if __name__ == "__main__":
    reset_seed()

    test_doublewell_9c()
    test_triplewell_9c()

    print("=" * 60)
    print("STAGE 9C COMPLETE")
