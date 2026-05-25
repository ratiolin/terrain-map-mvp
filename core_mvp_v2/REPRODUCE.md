# REPRODUCE.md — Core MVP V2 (Entropy-Regularized)

## Key Change

Added entropy regularization to prevent structural collapse:
```
loss = task_loss - λ * routing_entropy(z)
```

λ=0.05 keeps routing soft (entropy ~1.38 ≈ log(4)), maintaining multi-expert structure.

## Reproduction

```bash
uv sync
uv run python -c "from core_mvp_v2.produce_results import produce_all; produce_all()"
```

## Lambda Scan
| λ | S_adv | entropy | n_exp | struct |
|----|-------|---------|-------|--------|
| 0.00 | 1.17 | 0.67 | 2 | pattern |
| 0.01 | 1.20 | 1.38 | 4 | multi |
| 0.05 | 1.20 | 1.38 | 4 | multi |
| 0.10 | 1.20 | 1.38 | 4 | multi |
| 0.20 | 1.20 | 1.38 | 4 | multi |

λ≥0.01 prevents collapse entirely.

## Band Scan (λ=0.05)
- Entropy: constant ~1.38 across all g ∈ [0,3]
- Structure: always "multi" (4 experts), always "stable"
- S_adv: non-monotonic, emergence window g∈[0.30, 3.00]
- Peak S_adv=2.52 at g=0.30

## g-Equivalence (λ=0.05, g=1.0)
| η | drift | S_adv | entropy | n_exp |
|----|-------|-------|---------|-------|
| 0.1 | 10 | 1.08 | 1.10 | 3 |
| 0.2 | 5 | 1.18 | 1.39 | 4 |
| 0.5 | 2 | 0.75 | 1.34 | 4 |

## Perturbation (g=0.5, λ=0.05)
- Emergence: 1/7 → FRAGILE
- Structure always survives perturbation (entropy ~1.37-1.39)
- Emergence itself is seed-sensitive despite stable structure

## Multi-Stability
- All seeds converge to "multi" at all g — deterministic structure
- S_adv varies across seeds → emergence is not deterministic

## Parameters
```
seed=0, eta=0.5, coupling=drift, beta=0.8
flip_period=200, kappa=1.0, K=4, hidden_dim=4, gating_hidden=8
train_steps=1200, test_steps=300, entropy_lambda=0.05
```

## Files
```
core_mvp_v2/
├── env.py, agent.py, gating.py, controller.py, metrics.py
├── run_mvp.py, produce_results.py
└── results/
    ├── phase_diagram.json
    └── structure_logs.pkl
```
