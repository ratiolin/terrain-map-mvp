# Terrain Map MVP

**Structure–Complexity Co-Evolution System**

A self-organizing learning system in which model structure adapts to environment structure. Effective capacity is not fixed, but depends on the separability of the experienced distribution within a closed-loop interaction.

---

## Status

```
STAGE 1–10 COMPLETE
```

Validated capabilities:

- dynamic structural adaptation (grow / merge / prune / freeze)
- state-dependent routing with temporal coherence
- expert specialization under separable regimes
- stable behavior under perturbation and recovery
- closed-loop interaction between model and environment
- multi-attractor behavior selection via internal modulation
- phase-dependent performance across environment drift regimes

---

## System Overview

```
state → GRU(hidden) + Linear(state) → z_logits → softmax → z_soft
                                                        ↓
models: {P₀, ..., P_K₋₁}  (independent predictors)
                                                        ↓
prediction:
  soft:       Σ z[k] · pred_k
  semi-hard:  Σ z[k].detach() · pred_k
  hard:       pred[argmax(z)]

loss = MSE + α·z_loss + λ·temporal_loss − β·entropy
```

Optional policy coupling:

```
action ~ z_soft
env.step(action) → next_state
```

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full Stage 10 experiment (emergence phase diagram)
uv run python main_stage10.py

# Optional: diagnose high-drift routing failure
uv run python main_stage10_diag.py
```

This will output the judgment matrix, emergence window, and core metrics.

---

## Project Structure

```
Core Components
├── agent.py               # Expert predictors (MLP)
├── gating.py              # GRU routing with instantaneous state path
├── controller.py          # Structural adaptation (split/merge/prune)
├── loop_multi.py          # Multi-expert training & routing coordination
├── loop_policy.py         # Closed-loop policy coupling
├── metrics.py             # System diagnostics
└── analyze.py             # Behavioral & stability analysis

Environments
├── env_double_well.py              # Static multi-attractor environment
└── env_drifting_double_well.py     # Non-stationary environment with drift & phase flips

Stage 10 Experiment
├── experiment10.py        # Experiment grid, training loops, judgment logic
├── baseline_single.py     # Single-expert baselines (Small/Large)
├── analysis_stage10.py    # Judgment function & result matrix output
├── main_stage10.py        # Entry point for the full emergence phase diagram
├── main_stage10_diag.py   # Drift=0.08 diagnostic run
└── main_stage10_final.py  # Final validation & ablation scripts
```

---

## Core Components

- **Predictors (`agent.py`)**  
  MLP models learning forward dynamics.

- **Gating (`gating.py`)**  
  GRU-based routing with instantaneous state path.

- **Controller (`controller.py`)**  
  Structural adaptation (split, merge, prune, freeze).

- **Routing / Training (`loop_multi.py`, `loop_policy.py`)**  
  Multi-expert coordination, credit assignment, and policy coupling.

- **Analysis (`metrics.py`, `analyze.py`)**  
  System-level diagnostics and stability metrics.

---

## Key Mechanisms

### Adaptive Structure

- split when error stagnates
- merge when expert representations converge
- prune when usage falls below threshold
- freeze after prolonged stability

### Routing Evolution

- state and history jointly determine expert allocation
- routing stabilizes before specialization emerges
- specialization depends on input separability

### Credit Assignment

| mode       | effect                  |
|------------|-------------------------|
| soft       | shared gradient flow    |
| semi-hard  | expert-specific learning|
| hard       | discrete expert selection|

### Closed-Loop Behavior

- routing determines both prediction and action
- actions influence future state distributions
- training data is not fixed — it co-evolves with the policy
- the system operates as a feedback process with the environment

### Multi-Attractor Control

- experts specialize to distinct regions of state space
- routing selects attractors based on internal state and modulation
- behavior emerges from these selection dynamics

---

## Metrics (Operationalized Stability)

- `prediction_variance` – consistency of model output under perturbation
- `routing_consistency` – temporal stability of expert allocation
- `specialization_gap` – performance contrast across experts
- `attractor_dwell_time` – persistence within a behavioural regime
- `performance_gap` – linear ensemble vs. single large model

---

## Phase Behavior (Core Experimental Findings)

```
Low drift:
    distribution separable
    strong specialization
    ensemble dominates

Mid drift:
    partial separability
    emergence window
    probabilistic success

High drift:
    distribution mixing
    specialization collapse
    single model dominates
```

---

## Key Empirical Law

> **Structural advantage depends on the match between the rate of environmental change and the separability of representations.**

or equivalently:

> **Effective capacity is determined by whether the input distribution can be structurally decomposed, not by parameter count alone.**

---

## Design Properties

- internally driven structural evolution (no external supervision)
- deterministic execution (fixed seeds)
- measurable phase transitions across drift regimes
- minimal stage-wise modifications — each stage builds on the previous

---

## Limitations

- specialization collapses under severe distribution mixing
- effective capacity decreases as environmental drift increases
- closed-loop influence is bounded by control strength
- validated only on low-dimensional continuous environments
- attractor-based control cannot override environment instability

---

## Summary

The system evolves from a single predictor into a structured ensemble that:

- adapts capacity to environment structure
- forms experts only when separability allows
- loses structure when separability collapses
- exhibits phase-dependent performance along the drift axis

**Core result:**

> **Structure is not a universal advantage; it is an emergent property under conditions of environmental separability.**