import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_double_well import DoubleWellEnv, TripleWellEnv
from agent import Agent
from controller import GatingGrowthController
from loop_multi import experiment_soft
from analyze import (
    classify, all_separated, rollout_collect_balanced,
    train_linear_probe, evaluate_clf, confusion, state_to_label
)


def reset_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def test_doublewell_9a():
    print("=" * 60)
    print("TEST 1: DoubleWell — Phase 9A Constrained Alignment")
    print("=" * 60)

    env = DoubleWellEnv(noise=0.02, reset_pos=0.5)
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="doublewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=8, use_z=True)

    hist, _ = experiment_soft(env, agent, ctrl, steps=12000)

    K = ctrl.n_models()
    n_sep = all_separated(ctrl.models, min_dist=0.3)
    print(f"K = {K}")
    print(f"All separated: {n_sep}")

    if K < 2:
        print("K < 2 — forcing expansion for projection head training")
        ctrl.env_type = "doublewell"
        ctrl.models = []
        ctrl.init_models(agent)
        import copy
        for _ in range(3):
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
        for _ in range(3000):
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

    # Phase 9A: freeze all, train projection head
    for m in ctrl.models:
        for p in m.parameters():
            p.requires_grad = False
    for p in ctrl.gating.parameters():
        p.requires_grad = False
    ctrl.freeze_structure = True

    env_clean = DoubleWellEnv(noise=0.0, reset_pos=0.5)
    states, z_states, labels = rollout_collect_balanced(ctrl, env_clean, n_per_class=200)

    n_pos = sum(1 for l in labels if l == "positive_basin")
    n_neg = sum(1 for l in labels if l == "negative_basin")
    n_bound = sum(1 for l in labels if l == "boundary")
    print(f"Balanced labels: positive_basin={n_pos}, negative_basin={n_neg}, boundary={n_bound}")

    label_to_idx = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}
    z_arr = np.asarray(z_states, dtype=np.float32)
    y_arr = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    proj = nn.Linear(K, 3)
    opt = optim.Adam(proj.parameters(), lr=0.01)
    ce = nn.CrossEntropyLoss()

    z_t = torch.tensor(z_arr)
    y_t = torch.tensor(y_arr)

    for epoch in range(2000):
        logits = proj(z_t)
        loss = ce(logits, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = proj(z_t)
        preds = logits.argmax(dim=1).numpy()

    acc = float((preds == y_arr).mean())
    idx_to_label = {0: "negative_basin", 1: "boundary", 2: "positive_basin"}

    print(f"\nOverall accuracy: {acc:.4f}")
    print("Per-class accuracy:")
    per_class_ok = True
    for idx, cls_name in idx_to_label.items():
        mask = y_arr == idx
        if mask.sum() > 0:
            cls_acc = float((preds[mask] == idx).mean())
            status = "OK" if cls_acc >= 0.8 else "FAIL"
            if cls_acc < 0.8:
                per_class_ok = False
            print(f"  {cls_name:16s}: {cls_acc:.4f}  [{status}]")
        else:
            print(f"  {cls_name:16s}: no samples")

    if acc >= 0.9 and per_class_ok:
        print("\nRESULT: Phase 9A PASS — aligned projection learned, PROCEED TO 9B")
    else:
        print("\nRESULT: Phase 9A FAIL — alignment insufficient")

    print()
    reset_seed()


def test_triplewell_9a():
    print("=" * 60)
    print("TEST 2: TripleWell — Phase 9A Constrained Alignment")
    print("=" * 60)

    env = TripleWellEnv(noise=0.02, reset_range=(-1.2, 1.2))
    agent = Agent(obs_dim=1, act_dim=1, hidden_dim=2)
    ctrl = GatingGrowthController(check_interval=200, env_type="triplewell",
                                  merge_thresh=0.2, prune_thresh=0.03,
                                  max_models=10, use_z=True)

    hist, _ = experiment_soft(env, agent, ctrl, steps=20000)

    K = ctrl.n_models()
    n_sep = all_separated(ctrl.models, min_dist=0.3)
    print(f"K = {K}")
    print(f"All separated: {n_sep}")

    if K < 2:
        print("K < 2 — forcing expansion")
        ctrl.env_type = "triplewell"
        ctrl.models = []
        ctrl.init_models(agent)
        import copy
        for _ in range(4):
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
        for _ in range(3000):
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

    for m in ctrl.models:
        for p in m.parameters():
            p.requires_grad = False
    for p in ctrl.gating.parameters():
        p.requires_grad = False
    ctrl.freeze_structure = True

    env_clean = TripleWellEnv(noise=0.0, reset_range=(-1.2, 1.2))
    states, z_states, labels = rollout_collect_balanced(ctrl, env_clean, n_per_class=200)

    n_pos = sum(1 for l in labels if l == "positive_basin")
    n_neg = sum(1 for l in labels if l == "negative_basin")
    n_bound = sum(1 for l in labels if l == "boundary")
    print(f"Balanced labels: positive_basin={n_pos}, negative_basin={n_neg}, boundary={n_bound}")

    label_to_idx = {"negative_basin": 0, "boundary": 1, "positive_basin": 2}
    z_arr = np.asarray(z_states, dtype=np.float32)
    y_arr = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    proj = nn.Linear(K, 3)
    opt = optim.Adam(proj.parameters(), lr=0.01)
    ce = nn.CrossEntropyLoss()

    z_t = torch.tensor(z_arr)
    y_t = torch.tensor(y_arr)

    for epoch in range(2000):
        logits = proj(z_t)
        loss = ce(logits, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = proj(z_t)
        preds = logits.argmax(dim=1).numpy()

    acc = float((preds == y_arr).mean())
    idx_to_label = {0: "negative_basin", 1: "boundary", 2: "positive_basin"}

    print(f"\nOverall accuracy: {acc:.4f}")
    print("Per-class accuracy:")
    per_class_ok = True
    for idx, cls_name in idx_to_label.items():
        mask = y_arr == idx
        if mask.sum() > 0:
            cls_acc = float((preds[mask] == idx).mean())
            status = "OK" if cls_acc >= 0.8 else "FAIL"
            if cls_acc < 0.8:
                per_class_ok = False
            print(f"  {cls_name:16s}: {cls_acc:.4f}  [{status}]")
        else:
            print(f"  {cls_name:16s}: no samples")

    if acc >= 0.9 and per_class_ok:
        print("\nRESULT: Phase 9A PASS — aligned projection learned, PROCEED TO 9B")
    else:
        print("\nRESULT: Phase 9A FAIL — alignment insufficient")

    print()
    reset_seed()


if __name__ == "__main__":
    reset_seed()

    test_doublewell_9a()
    test_triplewell_9a()

    print("=" * 60)
    print("STAGE 9A COMPLETE")
