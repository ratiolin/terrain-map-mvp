# Terrain Map MVP

**Closed-Loop Stability Through Adaptation and Shaping**

A closed-loop learning system in which model structure and control policy co-evolve through interaction with the environment. The system maintains behavioral consistency and structural integrity — through adaptation when the environment is uncontrollable, through shaping when it is controllable. These are not two separate strategies but the same survival logic expressed under different boundary conditions.

---

## Status

```
STAGE 1–10 COMPLETE
CLOSED-LOOP VALIDATION COMPLETE
BOUNDARY CONDITION RECOGNITION: OPEN PROBLEM
```

### Validated

- Intrinsically closed-loop interaction (action → env → data distribution)
- Dynamic structural adaptation (grow / merge / prune / freeze)
- Banded emergence windows under adaptation objective (discrete drift intervals)
- Phase transition from collapsed policy to bang-bang control at λ/c ≈ 1
- Stability objective strictly dominates prediction objective across all tested target positions
- Adaptation and shaping separate via state distribution competition, not gradient opposition
- Capacity-competition gating produces discrete switching (jump magnitude 0.688)
- Gating floor ≈ 0.200 is structural, not a regularization artifact (β-invariant)

### Rejected

- **Unified stability metric U = alignment × feasibility**: Spearman correlation negative, performs worse than alignment alone
- **Intrinsic switching criterion in static-structure environments**: not found
- **λ* as an intrinsic critical point**: scale test confirms λ* ≈ c, switching point is a loss-scale balance, not a dynamical critical point

### Open

- **Intrinsic boundary condition recognition**: system cannot currently identify whether its environment is controllable without external specification via loss scale

---

## Quick Start

```bash
uv sync
uv run python run.py
```

---

## Project Structure

```
├── README.md
├── pyproject.toml
├── run.py
├── config.yaml
│
├── core/                          # Core engine
│   ├── agent.py
│   ├── controller.py
│   ├── env.py
│   ├── gating.py
│   ├── metrics.py
│   └── router.py
│
├── core_mvp_v2/                   # Current experiment system
│   ├── agent.py
│   ├── comparison_experiment.py
│   ├── control_env.py
│   ├── control_env_torch.py
│   ├── control_experiment.py
│   ├── control_experiment_v2.py
│   ├── controller.py
│   ├── env.py
│   ├── gating.py
│   ├── metrics.py
│   ├── produce_results.py
│   ├── run_mvp.py
│   ├── REPRODUCE.md
│   └── results/
│
└── results_final/
    ├── band_intervals.json
    ├── closed_loop_control_test.json
    ├── closed_loop_decomposition.json
    ├── closed_loop_verdict.json
    ├── comparison_A_vs_B.json
    ├── control_phase_diagram.json
    ├── explorability_phase.json
    ├── lambda_phase_transition.json
    ├── phase_diagram.json
    ├── phase_diagram_raw.json
    ├── run_output.json
    └── stability_curve.json
```

---

## Closed-Loop Stability Under Different Boundary Conditions

The system maintains closed-loop stability through two expressions of the same survival logic:

**Adaptation** (uncontrollable environment)
- Mechanism: minimize prediction error
- Condition: drift exceeds control authority
- Result: banded emergence windows; structure becomes useful within discrete drift intervals

**Shaping** (controllable environment)
- Mechanism: minimize control cost `(state - target)²`
- Condition: control authority exceeds environmental forcing
- Result: system maintains target state through active opposition; bands disappear under drift

### Experimental Evidence

| Condition | Adaptation (prediction loss) | Shaping (stability loss) |
|---|---|---|
| g = 0 | S_adv ≈ 1.0 | cost ≈ 0.005, in-zone 94% |
| g > 0.3 | S_adv = 1.5–2.1, bands present | cost = 9.0, system fails |
| target = 0 | failure rate 100% | failure rate 0% |
| target = ±1 | failure rate 100% | failure rate 0% |

### Separation Mechanism

Gradient alignment remains positive across all 132 tested (g, λ) points (range: 0.17–0.88). The two expressions do not oppose each other in gradient space. Separation occurs through **state distribution incompatibility**: when shaping succeeds, the system stays near target, removing the structural variation signal that adaptation requires; when adaptation succeeds, the system explores broadly, removing the directional signal that shaping requires.

---

## Phase Transition

```
loss = prediction_loss + λ · control_loss
```

| λ/c | behavior |
|---|---|
| < 1 | prediction dominant, policy collapsed or chaotic |
| ≈ 1 | **DISCRETE TRANSITION** (jump 0.688, robust across g and k) |
| > 1 | control dominant, bang-bang policy |

Transition is discrete (gating jump 0.688 vs soft baseline 0.220). Gating floor ≈ 0.200 persists regardless of regularization — the system always retains a minimum shaping signal for stability.

**Scale invariance test**: λ* ≈ c across c ∈ [0.5, 2.0, 5.0]. The transition point is a loss-scale balance, not an intrinsic dynamical critical point.

---

## Boundary Condition Recognition: Open Problem

The system cannot currently identify whether its environment is controllable without external specification. Switching between adaptation and shaping is triggered by the relative scale of the two loss terms (λ/c ≈ 1), not by any internal state signal.

**What is required for intrinsic recognition to emerge:**

1. Structure exists and changes over time (adaptation must keep up)
2. A target state exists (shaping is meaningful)
3. Tracking structural change and maintaining target state **compete for the same resource**

Condition 3 is absent in the current environment — adaptation consumes routing capacity while shaping consumes action magnitude. When resource competition exists, boundary recognition may emerge as an internal resource allocation problem rather than requiring external specification.

---

## Key Empirical Laws

1. **Structural advantage emerges within discrete drift windows** (under adaptation objective)
2. **Shaping dominates adaptation** when a target exists and control authority is sufficient
3. **Discrete phase transition at λ/c ≈ 1**, robust across drift and delay conditions
4. **Separation via state distribution competition**, not gradient opposition
5. **No intrinsic switching criterion** in static-structure or externally-scaled environments

---

## Limitations

- Validated on low-dimensional environments only
- Emergence windows not analytically predicted
- No intrinsic boundary condition recognition
- Switching point is loss-scale-dependent, not a universal constant
- No guarantee of global optimality

---

## Open Problems

1. **Intrinsic boundary condition recognition**: how can a system identify environment controllability without external loss specification?
2. **Resource competition environment**: design requires time-varying structure with shared capacity between adaptation and shaping
3. **Mixed regime**: does a closed-loop structure exist that handles both simultaneously?
4. **Real-time adaptation** under resource constraints
5. **Long-term behavior** in non-stationary environments
6. **Multi-agent compatibility**

---

## Core Result

> **Closed-loop stability is maintained through adaptation in uncontrollable environments and through shaping in controllable ones — the same survival logic under different boundary conditions. The transition between them is discrete and capacity-driven, but its trigger point is determined by external loss scale rather than internal system dynamics. Intrinsic boundary condition recognition remains an open problem.**