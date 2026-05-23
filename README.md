# **Terrain Map MVP**

**Structure–Complexity Co-evolution System**

A self-organizing learning system in which model structure adapts to environment structure. Capacity emerges through interaction: experts grow, merge, prune, and stabilize based on internal signals, while routing, specialization, and behavior co-evolve in a closed loop.

---

## **Status**

```
STAGE 1–9 COMPLETE
```

Validated capabilities:

- dynamic structural adaptation (grow / merge / prune / freeze)
    
- state-dependent routing with temporal coherence
    
- expert specialization via credit-aware learning
    
- stable behavior under perturbation and recovery
    
- closed-loop interaction between model and environment
    
- multi-attractor behavior selection via language modulation
    
- verified on CartPole, DoubleWell, and TripleWell
    

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
action ~ z_soft (or policy_heads[z])
env.step(action) → next_state, reward
```

---

## **Core Components**

- **Predictors (`agent.py`)**  
    MLP models learning forward dynamics `(state, action) → next_state`
    
- **Gating (`gating.py`)**  
    GRU-based routing with direct state path for instantaneous response
    
- **Controller (`controller.py`)**  
    Structural adaptation: split, merge, prune, freeze, bias initialization
    
- **Routing / Training (`loop_multi.py`, `loop_policy.py`)**  
    Multi-expert coordination, credit assignment, and policy interaction
    
- **Analysis (`analyze.py`, `metrics.py`)**  
    Behavioral evaluation, routing statistics, and system diagnostics
    

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

# temporal routing
uv run python main_stage6b.py

# stabilization
uv run python main_stage6d.py

# routing modes
uv run python main_stage7.py

# closed-loop policy
uv run python main_stage8.py

# language-modulated control
uv run python main_stage9e.py
```

---

## **Project Structure**

```
agent.py
gating.py
controller.py
router.py

loop.py
loop_multi.py
loop_policy.py

metrics.py
analyze.py
experiment.py

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
main_stage9.py
main_stage9a.py
main_stage9b.py
main_stage9b_prime.py
main_stage9c.py
main_stage9c_prime.py
main_stage9d.py
main_stage9e.py
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

- static → temporal → latent-conditioned
    
- state + history determine expert allocation
    
- routing stabilizes before specialization
    

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

- routing determines prediction and action selection
    
- actions influence future state distribution
    
- training data evolves with policy
    
- system forms a feedback loop with environment
    

---

### **Multi-Attractor Control**

- experts specialize to different regions of state space
    
- routing selects experts based on state and modulation
    
- behavior emerges as attractor selection
    

Validated behavior:

- DoubleWell: bidirectional attractor control
    
- TripleWell: multi-attractor selection (3-way separation)
    

---

## **Metrics**

- **stability** — consistency of dominant expert
    
- **separation** — routing differentiation across regions
    
- **specialization** — expert performance contrast
    
- **max_fraction** — routing concentration
    
- **policy_entropy** — action diversity
    
- **reward_variance** — behavioral variability
    

---

## **Design Properties**

- internally driven structural evolution
    
- minimal stage-wise modifications
    
- deterministic execution (fixed seeds)
    
- measurable progression across stages
    
- no external supervision for structure
    

---

## **Limitations**

- continuous routing can reduce specialization without gradient isolation
    
- entropy alone is insufficient for boundary detection
    
- specialization depends on coverage of training distribution
    
- policy is indirect (emerges from routing, not globally optimized)
    
- control operates at attractor-selection level, not precise state targeting
    
- validated on low-dimensional continuous environments with clear spatial structure; scaling to high-dimensional or weakly structured domains remains untested
    

---

## **Summary**

The system evolves from a single predictor into a structured ensemble that:

- discovers latent structure in the environment
    
- allocates capacity adaptively
    
- stabilizes its own architecture
    
- forms specialized experts
    
- and controls behavior through internal routing dynamics
    

**Core result:**  
Behavior can be modulated by selecting among dynamically learned experts, enabling consistent control over system-level outcomes through internal structure rather than explicit supervision.