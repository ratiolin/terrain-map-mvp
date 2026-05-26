# Terrain Map MVP

**Closed-Loop Stability Through Adaptation and Shaping**

A closed-loop learning system where model structure and control policy co-evolve through interaction with the environment. Stability is maintained either by tracking environmental dynamics or by constraining them, depending on boundary conditions. The system further exhibits a low-dimensional, geometrically structured controllability subspace whose existence, stability, and functional role are empirically validated.

---

## Status

```
STAGE 1–10 COMPLETE
CLOSED-LOOP VALIDATION COMPLETE
BOUNDARY CONDITION RECOGNITION: PRINCIPLE VALIDATED, GEOMETRIC STRUCTURE IDENTIFIED, LOW-RANK SUBSPACE CONFIRMED, GENERALIZATION OPEN
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
* Controllability subspace emergent in hidden states (R²≈0.5, >10× improvement over raw state; causal necessity confirmed by ablation; causal sufficiency confirmed by injection)
* Geometric controllability subspace identified via Jacobian SVD with coordinate-invariant structure
* Strong low-rank structure of Jacobian (k80≈2–3), confirming existence of a compact controllability subspace
* Cross-trajectory stability of subspace (alignment ≈0.88 ± 0.05 across segments)
* Discrete latent dynamics confirmed in geometric subspace (high cluster separation; low intra-cluster angle)
* Partial intra-cluster continuous modulation (significant in controllable regime, weaker near uncontrollable boundary)
* Learned origin of geometric structure confirmed via weight randomization (alignment collapses to random while rank persists)

---

### Rejected

* Unified stability metric
* Intrinsic switching in static environments
* λ* as intrinsic critical point

---

### Open

* Generalization of boundary recognition to high-dimensional, continuously-varying environments
* Continuous controllability tracking beyond binary drift intervals
* Scaling laws of controllability subspace (rank, spectrum shape, task dependence)
* Cross-agent alignment laws of geometric controllability subspaces
* Failure regimes of subspace structure (e.g., spectrum flattening, loss of alignment)

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
The hidden state spontaneously extracts a low-dimensional controllability subspace. Controllability, nearly linearly undetectable from raw state, becomes linearly readable in hidden space and concentrates along a small number of dominant directions. This subspace is defined by the leading right singular vectors of the Jacobian (∂h/∂a), is invariant under coordinate transformations, and exhibits strong low-rank structure (k80≈2–3). Ablation collapses controllability, confirming causal necessity. Perturbation along these directions induces structured action changes, confirming causal sufficiency.

**Geometric Stability and Reproducibility:**
The controllability subspace is stable across independently sampled trajectory segments (alignment ≈0.88), indicating it is not a single-trajectory artifact. The structure persists across seeds, batches, and probe variations, demonstrating measurement robustness.

**Discrete and Continuous Structure:**
Within the geometric subspace, hidden-state dynamics exhibit discrete clustering, indicating state-type switching. Within clusters, partial continuous modulation exists: strong in controllable regimes and weakened near uncontrollable boundaries. This supports a hybrid structure of discrete modes with continuous internal regulation.

**Origin of Structure:**
Weight randomization destroys subspace alignment while preserving low-rank spectrum, showing that rank is architecture-constrained but geometric organization is learned. Thus, controllability is encoded as a learned geometric structure rather than a trivial byproduct of dimensionality.

**Cross-Agent Structure:**
Geometric controllability subspaces are not identical across agents but exhibit statistically significant overlap, indicating convergence toward task-constrained functional subspaces without coordinate alignment.

**Capacity Threshold:**
A minimal hidden dimension is sufficient for stable emergence of the controllability subspace. Increasing capacity strengthens structure but does not create it.

**Failure Conditions:**
If the Jacobian spectrum becomes flat, the low-rank subspace ceases to exist. If alignment drops to random levels, shared structure is lost. If controllability decoding collapses (R²→0), controllability is not encoded.

**Remaining Challenge:** Generalization to higher-dimensional, continuously-varying environments has not been tested.

---

## Key Empirical Laws

1. Structural advantage appears in discrete drift windows
2. Shaping dominates when control authority is sufficient
3. Transition is discrete; its trigger can be either external (loss scale) or internal (rollout comparison)
4. Separation via state distribution, not gradients
5. Hidden states encode controllability in a low-dimensional, geometrically defined, low-rank subspace that is causally necessary for adaptive behavior
6. Representation structure is jointly determined by task intrinsic dimensionality and coupling-induced spectral constraints
7. Geometric subspaces are coordinate-invariant, partially shared across agents, and statistically stable across trajectories
8. Discrete latent dynamics and continuous modulation coexist within the same geometric structure

---

## Limitations

* Low-dimensional validation
* No analytic prediction of emergence bands
* Intrinsic boundary recognition confirmed only in simplified environments
* Scaling laws of subspace rank and spectrum not yet characterized
* Cross-agent geometric alignment not fully understood
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
8. Scaling laws and failure boundaries of geometric controllability subspaces

---

## Core Result

Closed-loop stability is maintained by tracking in uncontrollable regimes and by constraining in controllable regimes.

A system trained solely to predict endogenously extracts controllability into a low-dimensional, coordinate-invariant, low-rank geometric subspace, organizing discrete mode switching with continuous internal modulation—without external rewards or loss weights.

This structure is stable, reproducible, and learned; its scaling and generalization properties remain open.