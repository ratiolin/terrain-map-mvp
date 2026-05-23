import random
import numpy as np
import torch

from env_double_well import DoubleWellEnv, TripleWellEnv
from agent import Agent
from controller import GatingGrowthController
from loop_multi import experiment_soft
from analyze import classify, all_separated, rollout_collect, train_linear_probe, evaluate_clf, confusion


def reset_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def test_doublewell_label_separability():
    print("=" * 60)
    print("TEST 1: DoubleWell — Representational Separability")
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
        print("K < 2 — forcing expansion for linear probe test")
        ctrl.env_type = "doublewell"
        ctrl.models = []
        ctrl.init_models(agent)
        import copy
        import torch.optim as optim
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

    env_clean = DoubleWellEnv(noise=0.0, reset_pos=0.5)
    states, z_states, labels = rollout_collect(ctrl, env_clean, agent=agent, steps=2000)

    n_pos = sum(1 for l in labels if l == "positive_basin")
    n_neg = sum(1 for l in labels if l == "negative_basin")
    n_bound = sum(1 for l in labels if l == "boundary")
    print(f"Labels: positive_basin={n_pos}, negative_basin={n_neg}, boundary={n_bound}")

    W, b, label_to_idx = train_linear_probe(z_states, labels, lr=0.01, epochs=2000)
    acc = evaluate_clf(W, b, label_to_idx, z_states, labels)
    print(f"Linear probe accuracy: {acc:.4f}")
    confusion(W, b, label_to_idx, z_states, labels)

    if acc >= 0.9:
        print("RESULT: Representations already separable -- SKIP PHASE 9A")
    else:
        print("RESULT: Representations NOT separable -- ENTER PHASE 9A")

    print("PASS: DoubleWell label separability checked")
    print()
    reset_seed()


def test_triplewell_label_separability():
    print("=" * 60)
    print("TEST 2: TripleWell — Representational Separability")
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
        print("K < 2 — forcing expansion for probe test")
        ctrl.env_type = "triplewell"
        ctrl.models = []
        ctrl.init_models(agent)
        import copy
        import torch.optim as optim
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

    env_clean = TripleWellEnv(noise=0.0, reset_range=(-1.2, 1.2))
    states, z_states, labels = rollout_collect(ctrl, env_clean, agent=agent, steps=2000)

    n_pos = sum(1 for l in labels if l == "positive_basin")
    n_neg = sum(1 for l in labels if l == "negative_basin")
    n_bound = sum(1 for l in labels if l == "boundary")
    print(f"Labels: positive_basin={n_pos}, negative_basin={n_neg}, boundary={n_bound}")

    W, b, label_to_idx = train_linear_probe(z_states, labels, lr=0.01, epochs=2000)
    acc = evaluate_clf(W, b, label_to_idx, z_states, labels)
    print(f"Linear probe accuracy: {acc:.4f}")
    confusion(W, b, label_to_idx, z_states, labels)

    if acc >= 0.9:
        print("RESULT: Representations already separable -- SKIP PHASE 9A")
    else:
        print("RESULT: Representations NOT separable -- ENTER PHASE 9A")

    print("PASS: TripleWell label separability checked")
    print()
    reset_seed()


if __name__ == "__main__":
    reset_seed()

    test_doublewell_label_separability()
    test_triplewell_label_separability()

    print("=" * 60)
    print("STAGE 9 COMPLETE")
