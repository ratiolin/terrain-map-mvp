import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from env_double_well import DoubleWellEnv, TripleWellEnv
from agent import Agent
from controller import GatingGrowthController
from loop_multi import experiment_soft
from analyze import (
    classify, all_separated, rollout_collect_balanced, state_to_label
)


def reset_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def copy_zgating_to_text(new_text_gating, old_zgating):
    with torch.no_grad():
        new_text_gating.gru.load_state_dict(old_zgating.gru.state_dict())
        new_text_gating.output.load_state_dict(old_zgating.output.state_dict())
        new_text_gating.direct.load_state_dict(old_zgating.direct.state_dict())


def train_text_branch(gating, states, labels, K, epochs=2000, lr=0.01):
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

        if epoch % 200 == 0:
            with torch.no_grad():
                preds = class_logits.argmax(dim=1)
                acc = (preds == label_t).float().mean().item()
                print(f"  epoch {epoch:4d}: loss={loss.item():.4f}, acc={acc:.4f}")

    with torch.no_grad():
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
        preds = class_logits.argmax(dim=1)
        final_acc = (preds == label_t).float().mean().item()
    print(f"  final acc: {final_acc:.4f}")

    return proj_head, final_acc


def _compare_silent(gating, pos, label_idx):
    gating.reset()
    state = np.array([pos], dtype=np.float32)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    _, _, z_no = gating(state_t, label=None)
    _, _, z_with = gating(state_t, label=label_idx)
    return z_no.detach().numpy(), z_with.detach().numpy()


def compare_text_vs_notext(gating, pos, label_idx):
    z_no, z_with = _compare_silent(gating, pos, label_idx)
    print(f"  pos={pos:+.2f}:  no_text={z_no[0].round(3)}  with_text={z_with[0].round(3)}")
    return z_no, z_with


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


def test_doublewell_9b():
    print("=" * 60)
    print("TEST 1: DoubleWell — Phase 9B Text-Modulated Gating")
    print("=" * 60)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl, K = train_controller(env, agent, "doublewell", steps=5000)
    print(f"K = {K}")

    from gating import TextZGatingNet
    text_gating = TextZGatingNet(state_dim=1, K=K, hidden_dim=32, temperature=0.5, emb_dim=16)
    copy_zgating_to_text(text_gating, ctrl.gating)

    env_clean = DoubleWellEnv(noise=0.0, reset_pos=0.5)
    states, z_states, labels = rollout_collect_balanced(ctrl, env_clean, n_per_class=200)

    print("Training text branch...")
    proj_head, text_acc = train_text_branch(text_gating, states, labels, K, epochs=500, lr=0.01)

    label_map = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}

    print("\n--- Comparison at key positions ---")
    for pos in [-0.8, -0.5, -0.1, 0.0, 0.1, 0.5, 0.8]:
        lb = state_to_label(np.array([pos]))
        compare_text_vs_notext(text_gating, pos, label_map[lb])

    print("\n--- Text modulation strength (mean ||z_with - z_no||) ---")
    z_diffs = {}
    for pos_class in ["negative_basin", "boundary", "positive_basin"]:
        idx = label_map[pos_class]
        diffs = []
        for pos in np.linspace(-1.5, 1.5, 30):
            z_no, z_with = _compare_silent(text_gating, float(pos), idx)
            diffs.append(float(np.linalg.norm(z_with[0] - z_no[0])))
        z_diffs[pos_class] = float(np.mean(diffs))
        print(f"  {pos_class:16s}: {z_diffs[pos_class]:.4f}")

    has_modulation = any(v > 0.01 for v in z_diffs.values())
    print(f"\nText modulation active: {has_modulation}")

    if has_modulation:
        print("RESULT: Phase 9B PASS")
    else:
        print("RESULT: Phase 9B WEAK")

    print()
    reset_seed()


def test_triplewell_9b():
    print("=" * 60)
    print("TEST 2: TripleWell — Phase 9B Text-Modulated Gating")
    print("=" * 60)

    env = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl, K = train_controller(env, agent, "triplewell", steps=8000)
    print(f"K = {K}")

    from gating import TextZGatingNet
    text_gating = TextZGatingNet(state_dim=1, K=K, hidden_dim=32, temperature=0.5, emb_dim=16)
    copy_zgating_to_text(text_gating, ctrl.gating)

    env_clean = TripleWellEnv(noise=0.0, reset_range=(-1.2, 1.2))
    states, z_states, labels = rollout_collect_balanced(ctrl, env_clean, n_per_class=200)

    print("Training text branch...")
    proj_head, text_acc = train_text_branch(text_gating, states, labels, K, epochs=500, lr=0.01)

    label_map = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}

    print("\n--- Text modulation strength (mean ||z_with - z_no||) ---")
    z_diffs = {}
    for pos_class in ["negative_basin", "boundary", "positive_basin"]:
        idx = label_map[pos_class]
        diffs = []
        for pos in np.linspace(-1.5, 1.5, 30):
            z_no, z_with = _compare_silent(text_gating, float(pos), idx)
            diffs.append(float(np.linalg.norm(z_with[0] - z_no[0])))
        z_diffs[pos_class] = float(np.mean(diffs))
        print(f"  {pos_class:16s}: {z_diffs[pos_class]:.4f}")

    has_modulation = any(v > 0.01 for v in z_diffs.values())
    print(f"\nText modulation active: {has_modulation}")

    if has_modulation:
        print("RESULT: Phase 9B PASS")
    else:
        print("RESULT: Phase 9B WEAK")

    print()
    reset_seed()


if __name__ == "__main__":
    reset_seed()

    test_doublewell_9b()
    test_triplewell_9b()

    print("=" * 60)
    print("STAGE 9B COMPLETE")
