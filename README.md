# **Terrain Map MVP**

**Structure–Complexity Co-Evolution System**

A self-organizing system in which model structure and environment structure co-evolve. Effective capacity is not fixed, but depends on the separability of the experienced distribution within a closed-loop interaction.

---

## **Status**

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

## **System Overview**

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

## **Core Components**

- **Predictors (`agent.py`)**  
    MLP models learning forward dynamics
- **Gating (`gating.py`)**  
    GRU-based routing with instantaneous state path
- **Controller (`controller.py`)**  
    Structural adaptation
- **Routing / Training (`loop_multi.py`, `loop_policy.py`)**  
    Coordination and learning
- **Analysis (`metrics.py`, `analyze.py`)**  
    System-level diagnostics

---

## **Key Mechanisms**

### **Adaptive Structure**

- split under persistent error
- merge under functional convergence
- prune under low usage
- freeze after stabilization

### **Routing Evolution**

- state + history determine allocation
- routing stabilizes before specialization
- specialization depends on input separability

### **Credit Assignment**

|mode|effect|
|---|---|
|soft|shared learning|
|semi-hard|expert separation|
|hard|discrete specialization|

### **Closed-Loop Behavior**

- model influences action
- action shapes future data distribution
- training distribution is not fixed
- system operates as a feedback process

### **Multi-Attractor Control**

- experts correspond to regions of state space
- routing selects attractors
- behavior emerges from selection dynamics

---

## **Metrics (Operationalized Stability)**

- prediction_variance
- routing_consistency
- specialization_gap
- attractor_dwell_time
- performance_gap (linear vs single)

---

## **Phase Behavior (Core Experimental Findings)**

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

## **Key Empirical Law**

> **The structural advantage depends on the match between the rate of environmental change and the separability of representations.**

or equivalently:

> **Effective capacity depends on whether the input distribution can be structurally decomposed, not on parameter count alone.**

---

## **Design Properties**

- internally driven structural evolution
- deterministic execution
- measurable phase transitions
- no external structural supervision

---

## **Limitations**

- specialization fails under distribution mixing
- effective capacity decreases as drift increases
- closed-loop influence limited by control strength
- not validated in high-dimensional domains
- attractor-based control cannot override environment instability

---

## **Summary**

The system evolves into a structured ensemble that:

- adapts capacity to environment structure
- forms experts only when separability allows
- loses structure when separability collapses
- and exhibits phase-dependent performance

**Core result:**

> **Structure is not a universal advantage; it is an emergent property under conditions of environmental separability.**