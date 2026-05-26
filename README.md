# Terrain Map MVP

**Closed-Loop Stability Through Adaptation and Shaping**

A closed-loop learning system where model structure and control policy co-evolve through interaction with the environment. Stability is maintained either by tracking environmental dynamics or by constraining them, depending on boundary conditions.

---

## Status

```
STAGE 1–10 COMPLETE
CLOSED-LOOP VALIDATION COMPLETE
BOUNDARY CONDITION RECOGNITION: PRINCIPLE VALIDATED, GENERALIZATION OPEN
```

### Validated

* Closed-loop interaction (action → environment → data distribution)
* Structural adaptation dynamics (grow / merge / prune / freeze)
* Discrete emergence bands under adaptation
* Phase transition at λ/c ≈ 1
* Stability objective dominates prediction objective
* Separation via state distribution, not gradients
* Capacity-based gating produces discrete switching
* Non-zero shaping floor is structural
* Intrinsic boundary condition recognition via rollout comparison (multiple seeds)
* Controllability subspace emergent in hidden states (R²=0.51, 10× improvement; causal necessity confirmed by ablation; causal sufficiency confirmed by injection)

---

### Rejected

* Unified stability metric
* Intrinsic switching in static environments
* λ* as intrinsic critical point

---

### Open

* Generalization of boundary recognition to high-dimensional, continuously-varying environments
* Continuous controllability tracking beyond binary drift intervals

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run Stage 1–10 baseline (drift phase diagram, etc.)
uv run python run.py

# Run intrinsic boundary recognition experiment (Phase 0)
uv run python core_mvp_v3/run.py
```

Key outputs are written to `results_final/` and `core_mvp_v2/results/`.

---

## Project Structure

```
├── README.md
├── pyproject.toml
├── uv.lock
├── config.yaml
├── run.py
│
├── core/
│   ├── agent.py
│   ├── controller.py
│   ├── env.py
│   ├── gating.py
│   ├── metrics.py
│   └── router.py
│
├── core_mvp_v2/
│   ├── agent.py
│   ├── controller.py
│   ├── env.py
│   ├── gating.py
│   ├── metrics.py
│   ├── run_mvp.py
│   ├── comparison_experiment.py
│   ├── gradient_alignment_experiment.py
│   ├── gated_objective.py
│   ├── delay_alignment.py
│   └── results/
│
├── core_mvp_v3/
│   ├── env.py
│   ├── models.py
│   ├── experiment.py
│   └── run.py
│
└── results_final/
```

---

## Closed-Loop Stability Under Different Boundary Conditions

**Adaptation**

* Mechanism: prediction minimization
* Condition: environment dominates
* Result: banded structure utility

**Shaping**

* Mechanism: control minimization
* Condition: control dominates
* Result: target stabilization

---

## Separation Mechanism

Separation arises from **state distribution incompatibility**, not gradient conflict.

---

## Phase Transition

```
loss = prediction_loss + λ · control_loss
```

| λ/c | behavior            |
| --- | ------------------- |
| < 1 | prediction-dominant |
| ≈ 1 | discrete transition |
| > 1 | control-dominant    |

Earlier experiments showed transition depended on loss scale. Later experiments removed external λ entirely and achieved intrinsic mode switching via rollout comparison and panic signals.

---

## Boundary Condition Recognition: Principle Validated

The system can endogenously recognize environmental controllability in low-dimensional settings. Through real-environment rollout comparison, the system perceives controllability and drives mode switching via multi-scale panic signals. Multiple seeds confirm the effect: ADAPT dominant (74–85%) in high-drift regimes, SHAPE dominant (up to 92%) in low-drift regimes.

**Emergent Controllability Subspace:**
The hidden state spontaneously extracts a low-dimensional controllability subspace. Controllability, nearly linearly undetectable from raw state (R²=0.05), becomes readable in the hidden space (R²=0.51, 10× increase) and localized to 5 specific dimensions. Ablation of this subspace collapses the probe (R² drop 83%), confirming causal necessity. Injection of ±5 units along the controllability direction causes 2.6–3.6× action magnitude swings, confirming causal sufficiency.

**Remaining Challenge:** Generalization to higher-dimensional, continuously-varying environments has not been tested.

---

## Key Empirical Laws

1. Structural advantage appears in discrete drift windows
2. Shaping dominates when control authority is sufficient
3. Transition is discrete; its trigger can be either external (loss scale) or internal (rollout comparison)
4. Separation via state distribution, not gradients
5. Hidden states spontaneously encode controllability in low-dimensional subspaces causally necessary for adaptive behavior
6. Representation rank is jointly determined by task intrinsic dimensionality and coupling-induced contraction: coupling provides the upper bound through spectral suppression, while task structure provides the lower bound through the minimal dimensionality required for optimal control. In low-dimensional tasks, rank-1 is overdetermined; in high-dimensional tasks, the final rank reflects the balance between task expansion and coupling contraction.

---

## Limitations

* Low-dimensional validation
* No analytic prediction of emergence bands
* Intrinsic boundary recognition confirmed only in binary drift interval environment
* Representation rank is currently evaluated only in a 1D double-well task; the interaction between task intrinsic dimensionality and coupling-induced spectral constraints has not been quantitatively mapped in higher-dimensional environments
* No guarantee of global optimality

---

## Open Problems

1. Generalization of intrinsic controllability detection to high-dimensional, continuously-varying environments
2. Real-time adaptation under resource constraints
3. Mixed regime stability and coexistence conditions
4. Long-term behavior in non-stationary settings
5. Multi-agent coordination
6. Minimum viable closed-loop system
7. Interaction between task intrinsic dimensionality and coupling-induced spectral constraints

---

## Core Result

> **Closed-loop stability is maintained by tracking in uncontrollable regimes and by constraining in controllable regimes. A system trained solely to predict can endogenously learn to perceive environmental controllability through real-environment interaction, extract it into a low-dimensional hidden subspace causally necessary for behavior, and switch between behavioral modes without external rewards or loss weights. Representation structure is jointly determined by task intrinsic dimensionality and coupling-induced spectral constraints. The principle of intrinsic boundary condition recognition has been validated in low-dimensional settings; generalization remains open.**