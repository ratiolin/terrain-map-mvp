# Terrain Map MVP

**Structure–Complexity Co-Evolution System**

A self-organizing learning system in which model structure adapts to environment structure. Effective capacity is not fixed, but depends on when structural decomposition is both **possible and stable** within a closed-loop interaction.

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
- banded emergence windows with inertia-limited stability

---

## Quick Start

```bash
# Install dependencies
uv sync

# 1. Baseline drift phase diagram (drift single axis)
uv run python main_stage10.py

# 2. Inertia dual-axis experiment (drift × inertia phase diagram)
uv run python main_stage10_inertia.py

# 3. Emergence window blind test (pre-hoc prediction)
uv run python main_stage10_blindtest.py

# View key outputs
# - Phase diagram: phase_diagram.png
# - Stability curve: stability_curve.png
# - Raw data: phase_diagram_raw.json, stability_curve.json
```

---

## Project Structure

```
├── README.md
├── pyproject.toml
├── uv.lock
│
├── Core Components
│   ├── agent.py                   # Expert predictor (MLP)
│   ├── gating.py                  # GRU routing
│   ├── gating_multi_scale.py      # Multi-scale gating with inertia control
│   ├── controller.py              # Structural adaptation (split/merge/prune)
│   ├── router.py                  # Routing logic
│   ├── loop.py                    # Training loop (single expert)
│   ├── loop_multi.py              # Multi-expert training & routing coordination
│   ├── loop_policy.py             # Closed-loop policy coupling
│   └── metrics.py / analyze.py    # System diagnostics & stability metrics
│
├── Environments
│   ├── env.py
│   ├── env_double_well.py         # Static multi-attractor environment
│   └── env_drifting_double_well.py # Non-stationary environment with drift & phase flips
│
├── Experiments & Validation (Stages 1–10)
│   ├── experiment.py / experiment10.py   # Experiment setup & grid definitions
│   ├── baseline_single.py               # Single-expert baselines (Small/Large)
│   ├── analysis_stage10.py              # Judgment function & result matrix
│   ├── main_stage3.py … main_stage9e.py # Historical stage scripts
│   ├── main_stage10.py                  # Baseline drift phase diagram
│   ├── main_stage10_phase.py            # Phase-dependent behavior analysis
│   ├── main_stage10_inertia.py          # Inertia dual-axis experiment
│   ├── main_stage10_ablation.py         # Ablation study
│   └── main_stage10_diag.py             # Diagnostic tools for high-drift failure
│
├── Blind Tests & Pre-hoc Prediction (New)
│   ├── main_stage10_blindtest.py
│   ├── main_stage10_blindtest2.py
│   └── main_stage10_blind3.py
│
└── Results Output
    ├── phase_diagram.png / phase_diagram_raw.json
    ├── stability_curve.png / stability_curve.json
    └── ...
```

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

## Key Mechanisms

### Adaptive Structure

- split when error stagnates
- merge when expert representations converge
- prune when usage falls below threshold
- freeze after prolonged stability

### Routing Evolution

- routing depends on state and history
- temporal coherence emerges before specialization
- inertia controls switching selectivity

### Credit Assignment

| mode       | effect                  |
|------------|-------------------------|
| soft       | shared gradient flow    |
| semi-hard  | expert-specific learning|
| hard       | discrete expert selection|

### Closed-Loop Behavior

- routing determines both prediction and action
- actions reshape future input distribution
- training data co-evolves with policy
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
- `attractor_dwell_time` – persistence within a behavioral regime
- `performance_gap` – linear ensemble vs. single large model
- `separability` – distance between expert error distributions (structural decomposability)
- `inertia` – effective switching resistance derived from routing temporal coherence

---

## Phase Behavior (Core Experimental Findings)

```
Structure advantage condition:

S_adv > 1  ⇔  drift ∈ W(kappa)   AND   η < η_max(drift)
```

where `W(kappa)` denotes discrete emergence windows and `η` is the effective routing inertia.

### Observed structure

```
drift axis:
    banded emergence windows (discrete)

inertia axis:
    right-side collapse boundary
```

### Regimes

```
Outside drift window:
    structure exists but is not useful

Inside window + low inertia:
    strong specialization, high advantage

Inside window + high inertia:
    collapse due to over-commitment

High drift:
    η_max shrinks → system cannot track → collapse
```

---

## Key Empirical Law

> **Structural advantage is determined by discrete environment–model resonance windows and bounded by inertia-limited stability.**

---

## Design Properties

- structure emerges only when decomposition is meaningful
- advantage is phase-dependent, not monotonic
- inertia introduces asymmetric stability constraint
- system behavior governed by coupled drift–inertia dynamics

---

## Limitations

- emergence windows currently observed as discrete (not yet predicted analytically)
- η_max(drift) functional form not fully derived
- validated on low-dimensional environments
- no global optimality guarantee

---

## Summary

The system evolves into a structured ensemble that:

- only benefits from structure within specific drift regimes
- loses advantage outside these regimes
- collapses under excessive inertia
- exhibits banded phase structure rather than smooth transitions

---

**Core result:**

> **Structure is useful only when environment dynamics fall within discrete compatibility windows and the system maintains sufficient responsiveness.**