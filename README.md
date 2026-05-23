# **Terrain Map MVP**

**Structure–Complexity Co-evolution System**

A self-organizing learning system where model capacity adapts automatically to environment structure. The number of experts grows, merges, prunes, and stabilizes based on internal signals, with learnable routing, temporal consistency, and adaptive credit assignment.

---

## **Status**

```
STAGE 1–8 COMPLETE
```

Validated capabilities:

- dynamic structure adaptation (grow / merge / prune / freeze)
    
- state-dependent routing with temporal coherence
    
- expert specialization via credit-aware training
    
- stable behavior under perturbation and recovery
    
- closed-loop interaction between model and environment
    
- verified across CartPole, DoubleWell, TripleWell
    

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
env.step(action) → next_state, reward
```

---

## **Core Components**

- **Predictors (`agent.py`)**  
    MLP models learning forward dynamics `(state, action) → next_state`
    
- **Gating (`gating.py`)**  
    GRU-based routing with direct state path for instantaneous response
    
- **Controller (`controller.py`)**  
    Handles structural transitions: split, merge, prune, freeze
    
- **Training loops (`loop_multi.py`, `loop_policy.py`)**  
    Multi-model routing and optional policy interaction
    

---

## **Quick Start**

```bash
uv sync

# baseline + perturbation mapping
uv run python main.py

# structural adaptation
uv run python main_stage4.py

# learnable routing
uv run python main_stage5.py

# temporal + explicit latent routing
uv run python main_stage6b.py

# stabilization + generalization
uv run python main_stage6d.py

# routing modes (credit assignment)
uv run python main_stage7.py

# policy coupling (closed loop)
uv run python main_stage8.py
```

---

## **Project Structure**

```
agent.py
gating.py
controller.py
loop.py
loop_multi.py
loop_policy.py
metrics.py
analyze.py

env.py
env_double_well.py

main.py
main_stage3.py
main_stage4.py
main_stage5.py
main_stage6a.py
main_stage6b.py
main_stage6c.py
main_stage6d.py
main_stage7.py
main_stage8.py
```

---

## **Key Mechanisms**

### **Adaptive Structure**

- split when error stagnates
    
- merge when models converge
    
- prune when usage is low
    
- freeze after prolonged stability
    

---

### **Routing Evolution**

- static → temporal → explicit latent
    
- state + history determine expert allocation
    
- routing becomes stable before specialization
    

---

### **Credit Assignment**

|mode|gradient flow|
|---|---|
|soft|full coupling|
|semi-hard|model-specific learning|
|hard|discrete assignment|

Semi-hard mode activates after structural stabilization.

---

### **Closed-Loop Behavior**

- routing defines both prediction weights and action probabilities
    
- actions affect future states
    
- data distribution shifts with learning
    
- model and environment form a feedback system
    

---

## **Metrics**

- **stability** — consistency of dominant expert
    
- **separation** — difference in routing across regions
    
- **specialization** — relative performance of experts
    
- **max_fraction** — usage concentration
    
- **policy_entropy** — action distribution diversity
    
- **reward_variance** — outcome variability
    

---

## **Design Properties**

- internally driven structural changes
    
- minimal incremental modifications per stage
    
- deterministic execution (fixed seeds)
    
- measurable and testable progression
    
- no external supervision for structure
    

---

## **Limitations**

- continuous routing reduces specialization without gradient isolation
    
- entropy is insufficient alone for boundary detection
    
- specialization depends on training distribution coverage
    
- policy is minimal (derived from routing, not optimized for reward)
    

---

## **Summary**

The system evolves from a single predictor into a structured ensemble that:

- discovers latent regions of the environment
    
- allocates capacity where needed
    
- stabilizes its own structure
    
- forms specialized experts
    
- and interacts with the environment through its own internal representation
    