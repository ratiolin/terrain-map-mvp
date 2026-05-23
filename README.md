# **Terrain Map MVP**

**Structure–Complexity Co-evolution System**

A self-organizing learning system in which model structure adapts to environment complexity. The system dynamically allocates capacity, discovers latent structure, and stabilizes specialized behaviors through internal signals and closed-loop interaction.

---

## **Status**

```
STAGE 1–9 COMPLETE
```

Validated capabilities:

- dynamic structure adaptation (grow / merge / prune / freeze)
    
- state-dependent routing with temporal coherence
    
- expert specialization via credit-aware training
    
- stable behavior under perturbation and recovery
    
- closed-loop interaction between model and environment
    
- multi-attractor behavior control via language modulation
    
- verified across CartPole, DoubleWell, and TripleWell environments
    

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
    Structural adaptation: split, merge, prune, freeze
    
- **Training loops (`loop_multi.py`, `loop_policy.py`)**  
    Multi-model routing and optional policy interaction
    

---

## **Quick Start**

```bash
uv sync

uv run python main.py
uv run python main_stage4.py
uv run python main_stage5.py
uv run python main_stage6b.py
uv run python main_stage6d.py
uv run python main_stage7.py
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
env_triple_well.py

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
main_stage9.py
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

- static → temporal → explicit latent routing
    
- state + history determine expert allocation
    
- routing stabilizes before specialization emerges
    

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
    
- actions influence future states
    
- data distribution shifts during learning
    
- model and environment form a feedback system
    

---

### **Language-Modulated Control**

- text input perturbs gating logits (`z_logits`)
    
- routing shifts expert selection
    
- selected experts induce distinct behavioral regimes
    

Validated behavior:

- **DoubleWell (2 attractors)**  
    language reliably switches between attractors (2/3 criteria)
    
- **TripleWell (3 attractors)**  
    language selects among three basins with stable separation (3/3 criteria)
    

Control operates at the level of **attractor selection**, not precise state targeting.

---

## **Metrics**

- **stability** — consistency of dominant expert
    
- **separation** — routing divergence across regions
    
- **specialization** — expert performance differences
    
- **max_fraction** — usage concentration
    
- **policy_entropy** — action diversity
    
- **reward_variance** — outcome variability
    

---

## **Design Properties**

- internally driven structural evolution
    
- minimal stage-wise modifications
    
- deterministic execution (fixed seeds)
    
- measurable progression at each stage
    
- no external supervision for structure
    

---

## **Limitations**

- continuous routing reduces specialization without gradient isolation
    
- entropy alone does not identify structural boundaries
    
- specialization depends on sufficient state-space coverage
    
- policy remains minimal and is not reward-optimized
    
- control is limited to attractor selection, not precise state stabilization
    
- validated on low-dimensional continuous environments with clear spatial structure; scaling to high-dimensional or weakly structured domains remains untested
    

---

## **Summary**

The system evolves from a single predictor into a structured ensemble that:

- discovers latent regions of the environment
    
- allocates capacity adaptively
    
- stabilizes its internal structure
    
- forms specialized experts
    
- and controls behavior by selecting among emergent dynamical regimes
    

Language modulation demonstrates that high-level signals can steer system behavior by influencing internal routing, enabling consistent selection among multiple attractors within a learned dynamical landscape.