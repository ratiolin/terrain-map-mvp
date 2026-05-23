# **Terrain Map MVP**

**Structure-Complexity Co-evolution**: A system where model architecture (number of experts) grows, shrinks, merges, and freezes automatically in response to environment complexity — with learnable gating, temporal consistency, and switchable routing modes.

---

## Status

```
STAGE 1–7 COMPLETE

System validated with:
- adaptive structure growth
- state-dependent routing
- temporal consistency
- credit-aware training (semi-hard)
- tested on CartPole, DoubleWell, TripleWell
```

---

## Architecture Overview

```
State → [GRU hidden] + direct(state) → Z logits → softmax → z_soft (K weights)
                                                           ↓
Models: [P₀, P₁, ..., P_K₋₁] — each predicts next state independently
                                                           ↓
soft_pred =
{
  soft:      Σ z_soft[k] · pred_k
  semi-hard: Σ z_soft[k].detach() · pred_k
  hard:      pred[argmax(z)]
}
                                                           ↓
Loss = MSE(pred, target) + α·z_loss + λ·temporal_loss − β·entropy
```

**Key components:**

- **Predictor models** (`agent.py`): Small MLPs that learn forward dynamics `(state, action) → next_state`
    
- **Gating network** (`gating.py`): GRU + direct Linear(state) → K-way softmax routing weights
    
- **Growth controller** (`controller.py`): Manages model lifecycle — split, merge, prune, freeze
    
- **Routing** (`loop_multi.py`): Soft / semi-hard / hard routing modes with joint training
    

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run stage 1 — Single Model Learning (baseline dynamics fitting)
# Run stage 2 — Stability Mapping (5 perturbation experiments)
uv run python main.py

# Run stage 3 — Bifurcation (structural conservativeness + double-well)
uv run python main_stage3.py

# Run stage 4 — Self-Growing Architecture (K variable, merge/prune)
uv run python main_stage4.py

# Run stage 5 — Learnable Gating (soft routing)
uv run python main_stage5.py

# Run stage 6A — Temporal Gating (GRU-based history dependence)
uv run python main_stage6a.py

# Run stage 6B — Explicit Z (discrete latent + z_loss)
uv run python main_stage6b.py

# Run stage 6C — Expert Stabilization (auto-freeze + specialization)
uv run python main_stage6c.py

# Run stage 6D — Generalization & Robustness (OOD, boundaries)
uv run python main_stage6d.py

# Run stage 7 — Routing Modes (soft/semi-hard/hard, credit assignment)
uv run python main_stage7.py

# Run full pipeline (recommended)
uv run python main_stage7.py
```

---

## File Map

```
├── agent.py              # MLP agent: shared trunk, policy head, forward predictor
├── analyze.py            # Metrics: classify, temporal_consistency, z_separation,
│                         #   specialization_score, is_stable, rollout, verify_stage3
├── controller.py         # Controller hierarchy:
│                         #   Controller → SlowController → FreezeController →
│                         #   BifurcationController → GrowthController →
│                         #   GatingGrowthController
├── env.py                # CartPole-v1 wrapper with noise injection
├── env_double_well.py    # DoubleWellEnv + TripleWellEnv (smooth potentials)
├── experiment.py         # Unified experiment interface (stages 1-3)
├── gating.py             # GatingNet → TemporalGatingNet(GRU) → ZGatingNet(GRU+skip)
├── loop.py               # Basic training loop (stages 1-3)
├── loop_multi.py         # Advanced loops: run_growth, run_soft (routing modes)
├── metrics.py            # Error/reward tracking with history
├── router.py             # Hard routing by prediction error / argmin
├── main_stage1.py        # Stage 1: Single model baseline (no structure, no gating)
├── main.py               # Stage 2: 5 perturbation experiments (A-E)
├── main_stage3.py        # Stage 3: Bifurcation — CartPole + DoubleWell
├── main_stage4.py        # Stage 4: Self-Growing — CartPole + DoubleWell + TripleWell
├── main_stage5.py        # Stage 5: Learnable Gating — soft routing + collapse detection
├── main_stage6a.py       # Stage 6A: Temporal Gating — GRU-based
├── main_stage6b.py       # Stage 6B: Explicit Z — skip connection, z_loss
├── main_stage6c.py       # Stage 6C: Expert Stabilization — auto-freeze
├── main_stage6d.py       # Stage 6D: OOD Generalization — boundary uncertainty
└── main_stage7.py        # Stage 7: Routing Modes — soft/semi-hard/hard
```

---

## Core Concepts

### Stage 1 — Single Model Learning

A single predictor model learns environment dynamics without structure or routing:

- Objective: Fit `(state, action) → next_state` using MSE
    
- Behavior: Converges to a single global approximation
    
- Limitation: Cannot represent multi-modal or region-dependent dynamics
    

---

### Stability Mapping (Stage 2)

Five controlled perturbation experiments classify system response:

- **A**: Baseline (convergent)
    
- **B**: Slow updates → oscillation
    
- **C**: Extreme learning rate → divergence
    
- **D**: Environmental noise → oscillation
    
- **E**: Mid-run mutation → wrong convergence
    

---

### Structural Self-Growth (Stages 3-4)

Models split when prediction error stagnates ("stuck detection"), merge when parameters converge, and prune when usage drops below threshold. The number of models K grows organically from 1 upward.

---

### Learnable Gating (Stages 5-6)

A gating network replaces rule-based routing:

- **Stage 5**: Static MLP gating — `softmax(Linear(state))`
    
- **Stage 6A**: Temporal gating — `softmax(Linear(GRU(state, h_prev)))` for history-dependent routing
    
- **Stage 6B**: Explicit Z with skip connection — `z_logits = Linear(GRU(h)) + direct(state)` enabling state-immediate Z response
    
- **Stage 6C**: Auto-freeze — structure locks after 500 steps without changes; expert phase begins
    
- **Stage 6D**: OOD generalization — boundary regions show higher routing uncertainty (measured by entropy or winner diversity ν)
    

---

### Routing Modes (Stage 7)

Three switchable routing modes control gradient flow:

|Mode|Prediction|Gradient to gating|
|---|---|---|
|**soft**|`Σ z[k] · pred_k`|Full gradient through z|
|**semi-hard**|`Σ z[k].detach() · pred_k`|Only through z_loss|
|**hard**|`pred[argmax(z)]`|Only through z_loss|

Semi-hard auto-activates after structure freeze (credit assignment phase), enabling model specialization.

---

### Key Metrics

- **stability**: Fraction of steps where gating argmax is consistent (≥ 0.7 target)
    
- **Z separation**: L2 distance between mean Z in positive vs negative state regions (≥ 0.3 target)
    
- **specialization**: Mean log-ratio of model error on "other" vs "own" regions (≥ 0.5 target; semi-hard mode only — soft mode may be ≈ 0 due to shared gradients)
    
- **max_fraction**: Maximum usage share of any single model (≤ 0.85 target — prevents collapse)
    

---

## Dependencies

- Python ≥ 3.11
    
- `gymnasium[classic-control]`
    
- `torch`
    
- `numpy`
    

---

## Design Principles

1. **Minimal perturbation**: Each stage adds the smallest possible change to achieve the next capability level
    
2. **Endogenous triggers**: Structure changes (split, merge, prune, freeze) are triggered by internal metrics
    
3. **Compositional**: Controller hierarchy builds up capabilities through inheritance
    
4. **Reproducible**: Fixed seeds ensure repeatability
    
5. **Falsifiable**: Each stage has explicit pass/fail criteria derived from data
    

---

## Known Limitations

- Soft routing leads to low specialization due to shared gradients; semi-hard routing is required for expert differentiation
    
- Entropy alone is not a reliable boundary metric under continuous routing; winner diversity (ν) is more robust
    
- Specialization score is only meaningful in semi-hard mode; do not compare across routing modes
    