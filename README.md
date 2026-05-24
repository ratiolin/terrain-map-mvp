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

---

## Key Mechanisms

### Adaptive Structure

- split when error stagnates
    
- merge when expert representations converge
    
- prune when usage falls below threshold
    
- freeze after prolonged stability
    

---

### Routing Evolution

- routing depends on state and history
    
- temporal coherence emerges before specialization
    
- inertia controls switching selectivity
    

---

### Closed-Loop Behavior

- routing determines both prediction and action
    
- actions reshape future input distribution
    
- training data co-evolves with policy
    

---

## Phase Behavior (Core Experimental Findings)

```
Structure advantage condition:

S_adv > 1  ⇔  drift ∈ W(kappa)   AND   η < η_max(drift)
```

---

### Observed structure

```
drift axis:
    banded emergence windows (discrete)

inertia axis:
    right-side collapse boundary
```

---

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