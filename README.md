# Terrain Map MVP

**Structure–Complexity Co-Evolution System**

A self-organizing learning system in which model structure adapts to environment structure. Effective capacity is not fixed, but depends on when structural decomposition is both **possible and stable** within a closed-loop interaction. The resulting controller is not reducible to a low-dimensional explicit law, but operates as a hybrid dynamical system with internal state, memory, and discrete decision logic.

---

## Status

```
STAGE 1–10 COMPLETE
ADAPTIVE CONTROL LAYER INTEGRATED
```

Validated capabilities:

* dynamic structural adaptation (grow / merge / prune / freeze)

* state-dependent routing with temporal coherence

* expert specialization under separable regimes

* stable behavior under perturbation and recovery

* closed-loop interaction between model and environment

* multi-attractor behavior selection via internal modulation

* phase-dependent performance across environment drift regimes

* banded emergence windows with inertia-limited stability

* frequency-selective transfer behavior induced by finite memory

* learned inertia control without explicit prior

* performance-aware adaptive control (advantage-driven)

* probe-based stability boundary detection (online η_max estimation)

* hybrid meta-controller (aggressive–conservative blending)

* irreducible control behavior not representable by low-dimensional surrogate

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

# 4. Online adaptive control (full closed-loop system)
uv run python online_controller.py

# View key outputs
# - Phase diagram: phase_diagram.png
# - Stability curve: stability_curve.png
# - Stability analysis: stability_analysis.png
# - Raw data: *.json
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
├── Controllers (New)
│   ├── strategy.py                # Rule-based stability controller
│   ├── learned_controller.py      # Learned η(d) controller
│   ├── optimal_controller.py      # Oracle / optimal controller
│   ├── generalizable_controller.py # Generalizable meta-controller (v1)
│   ├── generalizable_v2.py        # Probe-aware controller (v2)
│   └── online_controller.py       # Final hybrid controller (v1+v2 + advantage)
│
├── Environments
│   ├── env.py
│   ├── env_double_well.py
│   └── env_drifting_double_well.py
│
├── Experiments & Validation (Stages 1–10)
│   ├── experiment.py / experiment10.py
│   ├── baseline_single.py
│   ├── analysis_stage10.py
│   ├── main_stage3.py … main_stage9e.py
│   ├── main_stage10.py
│   ├── main_stage10_phase.py
│   ├── main_stage10_inertia.py
│   ├── main_stage10_ablation.py
│   └── main_stage10_diag.py
│
├── Stability & Scaling Analysis
│   ├── scaling_law_experiment.py
│   ├── scaling_law.py
│   ├── scaling_law_censored.py
│   ├── band_structure.py
│   └── stability_pipeline.py
│
├── Results Output
│   ├── phase_diagram.png / phase_diagram_raw.json
│   ├── stability_curve.png / stability_curve.json
│   ├── band_structure.json
│   ├── eta_max_curve.json
│   ├── stability_regions.json
│   └── stability_function.json
```

---

## System Overview

```
state → GRU(hidden) + Linear(state) → z_logits → softmax → z_soft
                                                        ↓
models: {P₀, ..., P_K₋₁}
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

## Adaptive Control Layer (New)

### Meta-Control Law

```
η = w · η_aggressive + (1 - w) · η_conservative
```

### Gating Mechanism

```
w = sigmoid(advantage)
```

where:

```
advantage ≈ -Δloss
```

---

### Probe-Based Stability Estimation

```
η_max ≈ discovered via probe survival
η ← min(η, 0.95 · η_probe_bound)
```

---

### Behavior Modes

| mode         | condition            | behavior          |
| ------------ | -------------------- | ----------------- |
| aggressive   | high advantage       | exploit structure |
| conservative | low advantage        | stabilize         |
| probe        | near boundary        | explore η_max     |
| fallback     | instability detected | reduce η          |

---

## Key Mechanisms

### Adaptive Structure

* split when error stagnates

* merge when expert representations converge

* prune when usage falls below threshold

* freeze after prolonged stability

### Routing Evolution

* routing depends on state and history

* temporal coherence emerges before specialization

* inertia controls switching selectivity

* finite memory induces frequency-selective response

### Closed-Loop Behavior

* routing determines prediction and action

* actions reshape future input distribution

* data distribution co-evolves with policy

### Meta-Control (New)

* η is not fixed, but learned online

* control adapts to drift regime

* system balances:

  * stability

  * responsiveness

  * performance

* control behavior emerges from internal state, not reducible to explicit low-dimensional laws

---

## Metrics (Operationalized Stability)

* `prediction_variance`

* `routing_consistency`

* `specialization_gap`

* `attractor_dwell_time`

* `performance_gap`

* `separability`

* `inertia`

* `transfer_function(ω)`

* `advantage` (performance signal)

* `probe_survival_rate`

* `η_usage`

---

## Phase Behavior (Core Experimental Findings)

```
S_adv > 1  ⇔  drift ∈ W(kappa)   AND   η < η_max(drift)
```

---

### Observed structure

```
drift axis:
    banded emergence windows (discrete)

inertia axis:
    right-side collapse boundary

control axis:
    adaptive η(d) (non-monotonic, state-dependent)

frequency axis:
    band-pass-like transfer
```

---

### Regimes

```
Outside drift window:
    structure exists but not useful

Inside window + adaptive η:
    strong specialization and performance

High drift:
    η_max shrinks → tracking failure

Near boundary:
    probe-driven exploration

Frequency response:
    low ω: coherent tracking
    mid ω: suppression
    high ω: partial recovery
```

---

## Key Empirical Law

> **Structural advantage is determined by discrete environment–model resonance windows, modulated by inertia-limited stability, and optimized via performance-aware adaptive control.**

---

## Design Properties

* structure emerges only when decomposition is meaningful

* advantage is phase-dependent

* η is learned, not fixed

* control is performance-driven, not rule-driven

* stability boundary is discovered, not predefined

* system operates as a self-optimizing closed-loop

* control policy is history-dependent and internally stateful

* effective behavior cannot be compressed into simple analytic forms without loss of stability

---

## Limitations

* emergence windows not yet analytically predicted

* η_max(drift) only empirically learned

* frequency response not fully derived

* validated on low-dimensional environments

* no guarantee of global optimality

* controller behavior not reducible to interpretable closed-form expression

* post-hoc distillation into low-complexity models fails under closed-loop evaluation

---

## Summary

The system evolves into a structured, controlled ensemble that:

* activates structure only within valid drift bands

* adapts inertia dynamically based on performance

* discovers stability boundaries via probing

* avoids collapse through constraint enforcement

* optimizes behavior using direct performance signals

* exhibits banded phase structure and non-monotonic control

* operates as a hybrid dynamical system combining continuous adaptation and discrete decision logic

---

**Core result:**

> **Structure becomes useful only when environment dynamics fall within discrete compatibility windows, while control adapts in real time to maximize performance under stability constraints, forming an irreducible closed-loop dynamical system.**
