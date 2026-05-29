# Terrain Map MVP

**Closed-Loop Control as a Necessary Causal Mechanism**

A minimal framework proving that closed-loop control (action → state → risk → gradient → action) emerges spontaneously from survival pressure and is causally necessary for stability. The loop is not injected; it is discovered by the model as the only way to survive.

---

## Status

```
LAYER 1: CLOSED-LOOP DOMINANCE     — CONFIRMED (action-aware predictor, 87% pass)
LAYER 2: CAUSAL NECESSITY         — CONFIRMED (blocked-model collapse, 100% in DriftLevelEnv)
LAYER 3: EMERGENCE MECHANISM      — CONFIRMED (6/8, existence-driven model, 100% convergence)
LAYER 4: ROBUSTNESS & COMPRESSION — CONFIRMED (complexity sweep, information deprivation, dual-path noise suppression, minimal single-parameter controller)
```

### Proven

* **Closed-loop dominance**: action-aware predictor creates gradient pathway from risk back to policy — actions are non-zero and causally effective (Layer 1, `action_scale=0.2`, pass=87%)
* **Causal necessity via blocked retraining**: training with `action.detach()` collapses action norms and skyrockets risk (Layer 2, DriftLevelEnv 100% causal)
* **Online gradient disruption**: injecting gradient block or noise during training causes visible risk degradation (Layer 3, grad_blocked 136× risk increase)
* **Existence-driven learning**: loss = risk + λ||a||² — model learns control directly from survival pressure, no predictor needed (Layer 3, 100% emergence)
* **Complexity robustness**: closed-loop maintains risk <0.01 across drift ∈ {0.02,0.05,0.1,0.2} and noise ∈ {0,0.05,0.1} while blocked models diverge to 7M+ (Layer 4)
* **Information deprivation**: gradient noise σ=1.0 causes 240× risk increase; nullification at 60% rate causes 16× increase (Layer 4)
* **Bias adaptation**: closed-loop self-adapts to biased observations in true-vs-false world test (Layer 4)
* **Minimal compression**: single learned parameter `k = -11` maintains risk at 0.80 vs open-loop 1.67 (Layer 4)
* **Dual-path noise suppression**: model suppresses uncorrelated-noise channel to |k2/k1|=0.019 while exploiting any correlated signal (Layer 4)

### Architecture

| Model | Equation | Key |
|-------|----------|-----|
| `ClosedLoopModel` | `action = actor(h)`, `risk_pred = predictor([h, action])` | Gradient pathway: ∂loss/∂action ≠ 0 |
| `ExistenceDrivenModel` | `action = actor(encoder(s))` | loss = risk + λ||a||², gradient injected via `torch.autograd.backward` |
| `MinimalDirectModel` | `a = k * grad_hint` | Single-parameter controller |

### Environments

| Env | State | Action | Risk | Key Property |
|-----|-------|--------|------|-------------|
| `MultiModeEnv` | ℝᵈ (d=2-100) | ℝ² | ||x_ctrl|| | Double-well + OU noise, closed/open/pseudo modes |
| `DriftLevelEnv` | [0,1] | {0,1,2,3} | |level-0.5| | Discrete, drift forces action |
| `QuadraticDriftEnv` | ℝ | ℝ | (level-0.5)² | Continuous, risk never saturates |
| `LatentShiftDoubleWellEnv` | [x_obs] | ℝ | (x_true²-1)² | Hidden shift, grad_risk returned |

### Core Principle

```
The model does not learn to "control" — it learns to survive.
Survival requires non-zero actions when drift is present.
The gradient of risk with respect to action provides the only direction signal.
Cutting this gradient collapses the policy and the system fails.
Ergo: the closed loop is not an optimization choice; it is a causal necessity.
```

---

## Project Structure

```
core_mvp/
├── run.py                     # Single entry point: uv run python core_mvp/run.py --layer {1,2,all}
├── core/
│   ├── core_env.py            # All environments (MultiModeEnv, QuadraticDriftEnv, etc.)
│   ├── core_models.py         # All models (ClosedLoopModel, ExistenceDrivenModel, etc.)
│   ├── core_training.py       # Training loops (train_parallel, train_latent_shift, GPU-vectorized)
│   ├── core_metrics.py        # Metrics (W2, k80, effective_rank, CKA, GMM)
│   ├── core_panic.py          # PanicController (mode-switching for non-stationary envs)
│   └── core_logger.py         # ExperimentLogger, ResultAggregator
├── layers/
│   ├── layer1_final.py         # Layer 1: Closed-loop dominance (Phase A: existence, Phase B: directional)
│   ├── layer2_final.py         # Layer 2: Causal necessity across 3 environments
│   ├── layer3_final.py         # Layer 3: Existence-driven emergence with online intervention
│   └── layer4_final.py         # Layer 4: 6 experiments (delay, minimal, complexity, deprivation, true/false, dual-path)
├── legacy_v3/                 # Historical reference (v3 experiment/model/env)
└── results/
    ├── layer1/                # Layer 1 outputs
    ├── layer2/                # Layer 2 outputs
    ├── layer3/                # Layer 3 outputs
    └── layer4/                # Layer 4 outputs
```

---

## Quick Start

```bash
# Install
uv sync

# Quick test (all 4 layers, minimal seeds/steps)
uv run python core_mvp/run.py --quick

# Layer 1: Closed-loop dominance
uv run python core_mvp/run.py --layer 1

# Layer 2: Causal necessity
uv run python core_mvp/run.py --layer 2

# Layer 3: Emergence mechanism
uv run python core_mvp/run.py --layer 3

# Layer 4: Robustness, compression, deprivation, dual-path
uv run python core_mvp/run.py --layer 4

# Full experiment (all layers, 8 seeds)
uv run python core_mvp/run.py --layer all --seeds 8
```

---

## Layer Summary

### Layer 1 — Closed-Loop Dominance

**Question**: Does the model produce genuine control, or just statistical correlation?

**Answer**: Genuine control. The `ClosedLoopModel` with action-conditioned predictor (`risk_pred = predictor([h, action])`) creates a gradient pathway from prediction loss back to the actor. This produces non-zero actions (|a|≈0.01-0.05) that causally shift the state distribution (W2>0, gain>0, pseudo_ratio≈0). The old V4 model without this pathway collapsed to zero actions.

**Key metrics**: W2(closed,open) > 0, pseudo_ratio < 0.3, grad_norm > 0, |a| > 0.  
**Pass rate**: 87% (26/30 seeds) across d ∈ {2,5,10,20,50,100}.

### Layer 2 — Causal Necessity

**Question**: Is the closed loop causally necessary, or just correlated with stability?

**Answer**: Causally necessary. Training with `action.detach()` (blocking the risk gradient from reaching the actor) causes action norms to collapse and risk to skyrocket. The blocked model's failure proves the gradient pathway is essential.

**Key finding**: DriftLevelEnv achieves 100% causal confirmation. The blocked model cannot learn — its actions are driven only by the L2 penalty, which pushes them to zero.

### Layer 3 — Emergence Mechanism

**Question**: How and when does the closed loop form?

**Answer**: It forms during training at a detectable time t* (emergence point) when |action| first sustains above threshold and risk drops below early-training baseline. The `ExistenceDrivenModel` (loss = risk + λ||a||²) converges with 100% emergence rate.

**Key interventions**:  
- Gradient block during training → risk increases 136×  
- Gradient noise injection → risk increases 9×  
- Blocked model from scratch → risk = 24M vs normal = 0.0001  
- Gain variance σ = 0.0014 (stable control gain)

### Layer 4 — Robustness, Compression & Information

**Question**: How robust is the loop? Can it be compressed? Does it suppress noise?

**Answer**:  
- **Complexity robustness**: closed-loop survives all drift/noise combinations while blocked models diverge  
- **Information deprivation**: gradient noise (σ=1.0) → 240× risk; nullification (60%) → 16× risk  
- **Bias adaptation**: model self-adapts to biased observations (false world risk ≈ true world risk)  
- **Minimal compression**: single-parameter controller `a = k * grad_hint` maintains stable control (k = -11, risk = 0.80 < open-loop 1.67)  
- **Dual-path noise suppression**: model suppresses uncorrelated-noise channel to |k2/k1| = 0.019 while exploiting any correlated signal — the system is an "information integrator," not a gate

---

## Key Empirical Laws

1. **Gradient pathway is the architecture**: `predictor([h, action])` is the single change that transforms a passive predictor into an active controller.
2. **Survival pressure is the loss**: `loss = risk + λ||a||²` is sufficient for closed-loop emergence — no predictor needed.
3. **Drift forces control**: environments where inaction leads to increasing risk (drift, decay) naturally produce closed-loop behavior.
4. **Blocking proves causality**: `action.detach()` during training collapses the policy — the gradient pathway is causally necessary.
5. **The loop compresses to one parameter**: `a = k * grad_hint` is learnable and effective (k converges in all seeds).
6. **The system exploits any signal with non-zero correlation to risk reduction, regardless of its semantic validity**: only pure noise is ignored; any gradient direction that statistically reduces risk is utilized, whether it is real, delayed, or inverted. This makes the loop an information integrator, not a signal discriminator.

---

## Limitations

* Environment dimensionality currently limited to 1D-100D for continuous control
* Discrete action spaces (DriftLevelEnv) use separate model architecture
* QuadraticDriftEnv requires manual gradient injection via `torch.autograd.backward`; env dynamics are not natively differentiable
* Minimal model (`a = k * grad_hint`) requires environment to return analytic gradient
* Scaling to real-world environments (high-dimensional, partial observability) untested
* Long-term offline stability (model deployed without training) not evaluated
* No adversarial robustness testing

---

## Open Problems

1. Differentiable environment dynamics (end-to-end gradient from state to risk)
2. Multi-agent closed-loop emergence without explicit coordination
3. Information-theoretic characterization: mutual information I(action; risk) over training
4. Scaling laws for convergence time vs environment dimensionality
5. Partial observability: can the loop self-organize when state is hidden?
6. Adversarial drift: maximum drift rate the loop can counteract before failure
7. Transfer: can a loop learned in one environment stabilize in another?

---

## Core Result

Closed-loop control is not an optimization choice, an architectural preference, or a reward function. It is a **conditionally necessary mechanism for stability** that emerges whenever:

1. The environment imposes drift (inaction → increasing risk)
2. The model has a gradient pathway from risk back to action
3. The loss function includes the risk signal

Under these conditions, the model has no alternative but to learn control. Blocking the gradient pathway causes collapse. Compressing the loop to a single gain parameter preserves stability. The framework is minimal, falsifiable, and cross-environment validated. This claim holds under environments where inaction increases risk (drift-driven systems).