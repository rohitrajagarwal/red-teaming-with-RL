# Phase 3 Hyperparameter Tuning Analysis
**Analysis Date:** March 7, 2026

---

## Executive Summary

**PPO is the clear winner** with dramatically superior performance compared to DQN. All 5 PPO trials achieved positive mean rewards, while all 5 DQN trials resulted in negative mean rewards.

---

## Model Comparison: PPO vs DQN

### Performance Metrics

| Metric | PPO | DQN |
|--------|-----|-----|
| **Best Mean Reward** | **709.96** | **-97.61** |
| **Average Mean Reward** | **329.12** | **-131.25** |
| **Worst Mean Reward** | 35.83 | -170.89 |
| **Reward Range** | 674.13 | 73.28 |
| **Performance Advantage** | — | **−807.57 (PPO wins by 8.3x)** |

### Trial-by-Trial Performance

#### PPO Results (5 Trials)
| Trial | Mean Reward | Episodes | Status |
|-------|-------------|----------|--------|
| 0 | 308.27 | 313 | ✅ Good |
| 1 | 471.58 | 401 | ✅ Very Good |
| 2 | 35.83 | 283 | ⚠️ Weak |
| 3 | 119.99 | 286 | ✅ Fair |
| 4 | **709.96** | 826 | ✅✅ **Excellent** |

#### DQN Results (5 Trials)
| Trial | Mean Reward | Status |
|-------|-------------|--------|
| 0 | -97.61 | ❌ Negative |
| 1 | -128.62 | ❌ Negative |
| 2 | -155.80 | ❌ Negative |
| 3 | -103.62 | ❌ Negative |
| 4 | -170.89 | ❌ Negative |

---

## Best Model Configuration: PPO Trial 4 🏆

**Mean Reward: 709.96** (Best overall)

### Hyperparameters

#### Optimized Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| **learning_rate** | 8.31e-4 | Higher learning rate enables faster convergence |
| **ent_coef** | 0.0161 | Lower entropy coefficient reduces exploration noise |
| **clip_range** | 0.1610 | Moderate clipping range for stable updates |

#### Fixed Parameters (All trials)
| Parameter | Value |
|-----------|-------|
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |

### Training Statistics
- **Total Episodes:** 826
- **Total Timesteps:** 100,352
- **Mean Episode Length:** 121.49 steps
- **Max Reward:** 861.60
- **Min Reward:** -238.90

---

## Second Best: PPO Trial 1 🥈

**Mean Reward: 471.58**

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| **learning_rate** | 1.58e-4 |
| **ent_coef** | 2.05e-3 |
| **clip_range** | 0.0890 |

### Comparison to Best (Trial 4)
- Reward difference: **-238.38** (66.7% lower)
- Lower learning rate (less aggressive updates)
- Much lower ent_coef (less exploration)
- Lower clip_range (tighter gradient bounds)

---

## Third Best: PPO Trial 0 🥉

**Mean Reward: 308.27**

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| **learning_rate** | 5.61e-5 |
| **ent_coef** | 0.0797 |
| **clip_range** | 0.2330 |

### Comparison to Best (Trial 4)
- Reward difference: **-401.69** (56.6% lower)
- Very low learning rate (conservative updates)
- Moderate ent_coef (more exploration)
- Higher clip_range (looser gradient bounds)

---

## Complete Top 3 Ranking (All Parameters)

### 🥇 Rank 1: PPO Trial 4
```json
{
  "algorithm": "PPO",
  "trial": 4,
  "mean_reward": 709.96,
  "hyperparameters": {
    "learning_rate": 0.0008308860966122071,
    "ent_coef": 0.016110558588614335,
    "clip_range": 0.1609621331528477,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
  }
}
```

### 🥈 Rank 2: PPO Trial 1
```json
{
  "algorithm": "PPO",
  "trial": 1,
  "mean_reward": 471.58,
  "hyperparameters": {
    "learning_rate": 0.00015751320499779721,
    "ent_coef": 0.0020513382630874496,
    "clip_range": 0.08899863008405066,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
  }
}
```

### 🥉 Rank 3: PPO Trial 0
```json
{
  "algorithm": "PPO",
  "trial": 0,
  "mean_reward": 308.27,
  "hyperparameters": {
    "learning_rate": 5.6115164153345e-05,
    "ent_coef": 0.07969454818643933,
    "clip_range": 0.2329984854528513,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
  }
}
```

---

## Key Insights

### Why PPO Dominates DQN

1. **PPO's Advantages:**
   - More stable training with clipped objective function
   - Better exploration through entropy regularization
   - Robust to hyperparameter variations (all trials positive)
   - On-policy learning suited to continuous reward environment

2. **DQN's Limitations:**
   - Off-policy learning struggles with large action spaces
   - Experience replay buffer may not be optimal for this environment
   - Q-value overestimation issues in complex scenarios
   - All trials failed to achieve positive average returns

### Optimal Hyperparameter Characteristics

From the top-performing PPO trial, the winning configuration shows:

| Characteristic | Impact |
|---|---|
| **Higher Learning Rate (8.31e-4)** | Enables faster policy updates and adaptation; sweet spot for this environment |
| **Lower Entropy Coefficient (0.0161)** | Sufficient exploration without noise-driven decisions; focuses on exploitation |
| **Moderate Clip Range (0.1610)** | Stable but flexible gradient updates; allows meaningful policy changes |

### DQN Failure Analysis

DQN's consistent failure across all trials suggests:
- The environment's reward structure may be sparse or delay-heavy
- Q-learning's offline nature may be unsuitable for this dynamic environment
- The sample efficiency gains of DQN don't translate to better performance here
- PPO's on-policy approach better captures the environment's dynamics

---

## Recommendation for Phase 4

**Use PPO Trial 4 Configuration** as the champion model for Phase 4 with the following parameters:

```python
ppo_champion_config = {
    "learning_rate": 0.0008308860966122071,
    "ent_coef": 0.016110558588614335,
    "clip_range": 0.1609621331528477,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
}
```

**Expected Performance:** ~710 mean reward based on Phase 3 results

---

## Methodology Notes

- **Phase:** 3 - Optuna Hyperparameter Optimization
- **Timesteps per Trial:** 100,000
- **Sampler:** TPE (Tree-structured Parzen Estimator)
- **PPO Search Space:**
  - learning_rate: [1e-5, 1e-3] log scale
  - ent_coef: [1e-3, 1e-1] log scale
  - clip_range: [0.05, 0.3] linear scale
- **DQN Search Space:**
  - learning_rate: [1e-5, 1e-3] log scale
  - exploration_fraction: [0.01, 0.5] linear scale
  - epsilon_final: [0.01, 0.2] linear scale

