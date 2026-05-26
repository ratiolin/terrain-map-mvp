# Terrain Map MVP

**Closed-Loop Stability Through Adaptation and Shaping**

A closed-loop learning system where model structure and control policy co-evolve through interaction with the environment. Stability is maintained either by tracking environmental dynamics or by constraining them, depending on boundary conditions. The system exhibits a low-dimensional, geometrically structured controllability subspace; however, this structure is not a globally consistent coordinate object, but a task-constrained, locally expressed geometric organization whose functional role is empirically validated.

---

## Status

```
STAGE 1–10 COMPLETE
CLOSED-LOOP VALIDATION COMPLETE
BOUNDARY CONDITION RECOGNITION: PRINCIPLE VALIDATED, FUNCTIONAL STRUCTURE CONFIRMED, GLOBAL GEOMETRIC CONSISTENCY REJECTED, GENERALIZATION OPEN
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
* Controllability is encoded in hidden states (R²≈0.5, >10× improvement over raw state; causal necessity confirmed by ablation; causal sufficiency confirmed by intervention)
* Strong low-rank structure in Jacobian spectrum (k80≈2–3), indicating compact effective controllability directions
* Cross-trajectory statistical stability of controllability structure (alignment ≈0.88 ± 0.05 across segments)
* Discrete latent dynamics observable in controllability-relevant structure (high cluster separation; low intra-cluster angle)
* Partial intra-cluster continuous modulation (stronger in controllable regimes, weaker near uncontrollable boundary)
* Learned origin of functional structure confirmed via weight randomization (decoding collapses while rank persists)

---

### Rejected

* Unified stability metric
* Intrinsic switching in static environments
* λ* as intrinsic critical point
* Existence of a globally consistent, coordinate-invariant controllability subspace

---

### Open

* Generalization of boundary recognition to high-dimensional, continuously-varying environments
* Continuous controllability tracking beyond binary drift intervals
* Scaling laws of controllability structure (rank, spectrum shape, task dependence)
* Cross-agent alignment laws at the functional (not geometric) level
* Failure regimes of controllability encoding (e.g., spectrum flattening, loss of decoding, instability under distribution shift)

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

Key outputs are written to `results_final/` and `core_mvp_v2/results/`. Analysis scripts for geometric structure, gradient field, and multi-agent coordination are located in `core_mvp_v4/analysis/`, with results in `core_mvp_v4/results/`.

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
├── core_mvp_v4/
│   ├── __init__.py
│   ├── envs/
│   ├── analysis/
│   ├── experiments/
│   └── results/
│
└── results_final/
```

---

## Closed-Loop Stability Under Different Boundary Conditions

**Adaptation**

* Mechanism: prediction minimization
* Condition: environment dominates
* Result: tracking-based stabilization

**Shaping**

* Mechanism: control minimization
* Condition: control authority dominates
* Result: constraint-based stabilization

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

The system can endogenously recognize environmental controllability in low-dimensional settings. Through real-environment rollout comparison, the system evaluates controllability and drives mode switching via multi-scale panic signals. Multiple seeds confirm the effect: ADAPT dominant (74–85%) in high-drift regimes, SHAPE dominant (up to 92%) in low-drift regimes.

**Controllability Encoding:**
The hidden state encodes controllability in a low-dimensional manner. While raw state contains weak linear signal, hidden representations make controllability linearly decodable. This information concentrates along a small number of dominant directions, reflected in the low-rank structure of the Jacobian spectrum (k80≈2–3). Ablation removes controllability, confirming causal necessity. Targeted perturbations recover structured behavioral changes, confirming causal sufficiency.

**Geometric Structure (Revised Interpretation):**
Although dominant directions can be extracted locally (e.g., via Jacobian SVD), these directions do not form a globally consistent, coordinate-invariant subspace. Instead:

* Effective control directions are **state-dependent**
* Local linearizations are valid only within neighborhoods
* Global alignment is approximate and statistical, not exact

Thus, controllability is not encoded as a fixed geometric object, but as a **distributed, locally linear but globally non-coherent structure**.

**Statistical Stability:**
Across trajectory segments, extracted dominant directions exhibit high but imperfect alignment (≈0.88), indicating **statistical regularity rather than strict invariance**.

**Discrete and Continuous Structure:**
Hidden dynamics show:

* Discrete clustering (mode switching)
* Local continuous modulation within clusters

This supports a hybrid structure: **discrete regimes with local continuous control**, rather than a globally smooth manifold.

**Origin of Structure:**
Weight randomization preserves low-rank spectra but destroys controllability decoding. Therefore:

> Rank is architectural; controllability structure is learned and functional

**Cross-Agent Structure:**
Different agents do not share aligned geometric subspaces. However, they converge to **functionally equivalent solutions** with similar behavioral outcomes.

**Capacity Threshold:**
Minimal capacity is sufficient for controllability encoding. Increased capacity improves robustness but does not fundamentally change structure.

**Failure Conditions:**

* Spectrum flattening → loss of dominant directions
* Decoding collapse (R² → 0) → loss of controllability encoding
* Alignment degradation → loss of statistical structure

**Remaining Challenge:** Generalization to higher-dimensional, continuously-varying environments remains untested.

---

## Key Empirical Laws

1. Structural advantage appears in discrete drift windows
2. Shaping dominates when control authority is sufficient
3. Transition is discrete; trigger can be external (loss scale) or internal (rollout comparison)
4. Separation occurs via state distribution, not gradients
5. Controllability is encoded in a low-dimensional, low-rank, functionally defined structure, not a globally fixed subspace
6. Representation is determined by task constraints and optimization dynamics, not uniquely by function
7. Cross-agent consistency exists at the functional level, not the geometric level
8. Discrete dynamics and continuous modulation coexist locally, not globally

---

## Limitations

* Low-dimensional validation
* No analytic prediction of emergence bands
* Intrinsic boundary recognition confirmed only in simplified environments
* No global geometric model of controllability
* Scaling laws of controllability structure not yet characterized
* Cross-agent structure only understood statistically
* No guarantee of global optimality

---

## Open Problems

1. Generalization of intrinsic controllability detection to high-dimensional, continuously-varying environments
2. Real-time adaptation under resource constraints
3. Mixed regime stability and coexistence conditions
4. Long-term behavior in non-stationary settings
5. Multi-agent coordination
6. Minimum viable closed-loop system
7. Interaction between task intrinsic dimensionality and optimization-induced constraints
8. Scaling laws and breakdown conditions of controllability encoding

---

## Core Result

Closed-loop stability is maintained by tracking in uncontrollable regimes and by constraining in controllable regimes.

A system trained solely for prediction can encode controllability into a low-dimensional, low-rank structure in hidden states. This structure enables discrete mode switching with local continuous modulation, without requiring external rewards or explicit control objectives.

This encoding is functional, distributed, and state-dependent. It is stable at the statistical level, reproducible across runs, and sufficient for behavior—but not a globally consistent geometric object.

Its scaling behavior and generalization properties remain open.