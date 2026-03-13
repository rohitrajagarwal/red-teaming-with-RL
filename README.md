# RL-Based Cyber Exploitation: Hyperparameter Optimization

## Project Overview

This project implements RL algorithms (DQN and PPO) to optimize autonomous cyber exploitation in a simulated network environment. The primary goal is to compare exploration-exploitation strategies and hyperparameter tuning approaches using CyberBattleSim and Optuna.

## Quick Start

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify environment setup:**
   ```bash
   python -c "import cyberbattle; import stable_baselines3; print('Setup OK')"
   ```

## Project Structure

```
Group project/
├── src/
│   ├── __init__.py
│   ├── environment_wrapper.py          # Gymnasium wrapper for CyberBattleSim
│   ├── dqn_agent.py                    # DQN agent implementation
│   ├── ppo_agent.py                    # PPO agent implementation
│   ├── logger.py                       # Training metrics logger
│   └── __pycache__/
├── scripts/
│   ├── train_dqn.py                    # Phase 2: Train baseline DQN model
│   ├── train_ppo.py                    # Phase 2: Train baseline PPO model
│   ├── optuna_tuning.py                # Phase 3: Optuna optimization for DQN
│   ├── optuna_ppo_tuning.py            # Phase 3: Optuna optimization for PPO
│   ├── run_optuna_phase3.py            # Phase 3: Complete optimization workflow
│   ├── phase3_summary.py               # Phase 3: Generate summary statistics
│   ├── extract_phase3_results.py       # Phase 3: Extract trial configurations
│   ├── analyze_phase3.py               # Phase 3: Comparative analysis
│   ├── phase4_visualization.py         # Phase 4: Generate learning curves & plots
│   ├── phase4_champions.py             # Phase 4: Evaluate champion models
│   └── __init__.py
├── configs/
│   └── hyperparameters.json            # Stored hyperparameter configurations
├── models/                             # Saved trained model checkpoints
├── logs/
│   ├── dqn_metrics.json                # Phase 2 DQN baseline metrics
│   ├── dqn_training.csv                # Phase 2 DQN training logs
│   ├── ppo_metrics.json                # Phase 2 PPO baseline metrics
│   ├── ppo_training.csv                # Phase 2 PPO training logs
│   ├── optuna_dqn_results.json         # Phase 3 DQN Optuna results
│   ├── optuna_ppo_results.json         # Phase 3 PPO Optuna results
│   ├── phase3_best_ppo_config.json     # Phase 3 Best PPO configuration
│   ├── phase4_visualization_summary.txt # Phase 4 Visualization analysis
│   ├── dqn_checkpoints/                # Phase 2 DQN model checkpoints
│   ├── ppo_checkpoints/                # Phase 2 PPO model checkpoints
│   ├── optuna_dqn_trial_*/             # Phase 3 Individual DQN trial results
│   └── optuna_ppo_trial_*/             # Phase 3 Individual PPO trial results
├── PHASE3_ANALYSIS.md                  # Detailed Phase 3 hyperparameter analysis
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore patterns
└── README.md                           # This file
```

## Phase 1: Environment Setup & Sanity Check

### Verification
- ✅ All imports work without errors
- ✅ ToyCtf environment initializes properly
- ✅ Random agent completes training steps
- ✅ No crashes or hanging processes

## Phase 2: Baseline Model Training

Train baseline DQN and PPO models with default hyperparameters to establish performance baselines.

### Train DQN Baseline

```bash
python scripts/train_dqn.py \
  --total_timesteps 100000 \
  --learning_rate 0.0001 \
  --output_dir logs/
```

**Expected Output:**
- Baseline DQN model checkpoint in `logs/dqn_checkpoints/`
- Training metrics in `logs/dqn_metrics.json`
- Training logs in `logs/dqn_training.csv`

### Train PPO Baseline

```bash
python scripts/train_ppo.py \
  --total_timesteps 100000 \
  --learning_rate 0.0003 \
  --output_dir logs/
```

**Expected Output:**
- Baseline PPO model checkpoint in `logs/ppo_checkpoints/`
- Training metrics in `logs/ppo_metrics.json`
- Training logs in `logs/ppo_training.csv`

**Phase 2 Results Summary:**
- DQN: Mean reward fluctuates significantly, struggles with convergence
- PPO: More stable training, positive mean rewards observed
- *Conclusion: PPO shows superior baseline performance*

## Phase 3: Hyperparameter Optimization with Optuna

Optimize DQN and PPO hyperparameters using Optuna's Tree-structured Parzen Estimator (TPE) sampler across 5 trials each.

### Run Complete Phase 3 Pipeline

```bash
python scripts/run_optuna_phase3.py \
  --n_trials 5 \
  --timesteps_per_trial 100000 \
  --output_dir logs/
```

This will:
1. Run 5 DQN optimization trials
2. Run 5 PPO optimization trials
3. Generate comprehensive analysis reports

### Individual Algorithm Tuning

**Optuna-based DQN Tuning:**
```bash
python scripts/optuna_tuning.py \
  --n_trials 5 \
  --timesteps 100000 \
  --output_dir logs/
```

**Optuna-based PPO Tuning:**
```bash
python scripts/optuna_ppo_tuning.py \
  --n_trials 5 \
  --timesteps 100000 \
  --output_dir logs/
```

### Phase 3 Results: PPO Dominates DQN

| Metric | PPO | DQN |
|--------|-----|-----|
| **Best Mean Reward** | **709.96** | **-97.61** |
| **Average Mean Reward** | **329.12** | **-131.25** |
| **Worst Mean Reward** | 35.83 | -170.89 |
| **Winner** | ✅ **All 5 trials positive** | ❌ **All 5 trials negative** |

### 🏆 Champion Model: PPO Trial 4

**Mean Reward: 709.96** (best overall performance)

#### Optimal Hyperparameters

```json
{
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

#### Key Hyperparameter Insights

| Parameter | Value | Insight |
|-----------|-------|---------|
| **Learning Rate** | 8.31e-4 | Higher LR enables faster convergence in this environment |
| **Entropy Coefficient** | 0.0161 | Low entropy reduces exploration noise while maintaining sufficient exploration |
| **Clip Range** | 0.1610 | Moderate clipping enables stable yet flexible policy updates |

#### Training Statistics
- **Total Episodes:** 826
- **Total Timesteps:** 100,352
- **Mean Episode Length:** 121.49 steps
- **Max Reward:** 861.60
- **Min Reward:** -238.90

### Runner-up: PPO Trial 1

**Mean Reward: 471.58** (66.7% lower than best)

#### Hyperparameters
```json
{
  "learning_rate": 0.00015751320499779721,
  "ent_coef": 0.0020513382630874496,
  "clip_range": 0.08899863008405066
}
```

### Key Findings

**Why PPO Outperforms DQN:**
1. More stable training with clipped objective function
2. Better exploration through entropy regularization
3. Robust across hyperparameter variations (all trials positive)
4. On-policy learning suited to this reward environment

**Why DQN Failed:**
1. Off-policy learning struggles with large action spaces
2. Q-value overestimation issues in complex scenarios
3. Environment's reward structure favors on-policy approaches
4. Experience replay buffer not optimal for this dynamic environment

For detailed Phase 3 analysis, see [PHASE3_ANALYSIS.md](PHASE3_ANALYSIS.md).

## Phase 4: Visualization & Champion Analysis

Generate comprehensive visualizations and evaluate champion model performance.

### Run Phase 4 Analysis

```bash
python scripts/phase4_visualization.py \
  --input_dir logs/ \
  --output_dir logs/
```

This generates:
- **Learning Curves:** PPO vs DQN convergence comparison
- **Exploration Analysis:** Entropy and exploration fraction dynamics
- **Sample Efficiency:** Reward per timestep across trials
- **Hyperparameter Impact:** Effect of key parameters on performance

### Evaluate Champion Models

```bash
python scripts/phase4_champions.py \
  --champion_trial 4 \
  --algorithm ppo \
  --eval_episodes 50
```

### Phase 4 Key Observations

#### Learning Curves (Primary Figure)
- PPO shows clear convergence trends in 4/5 trials with positive rewards
- PPO Trial 4 demonstrates fastest convergence to reward plateau (~710)
- DQN exhibits stagnation—all trials plateau at negative rewards
- PPO's learning is more stable across different hyperparameter settings

#### Exploration Dynamics (Secondary Figure)
- Optimal entropy coefficient (β) ≈ 0.016–0.093 for this environment
- PPO Trial 4 (β=0.0161): Focused exploitation → 709.96 reward ✓
- PPO Trial 1 (β=0.00205): Minimal exploration → still good (471.58)
- PPO Trial 2 (β=0.0734): Moderate exploration → weak performance (35.83)
- DQN exploration variations show no correlation with performance

#### Sample Efficiency (Tertiary Figure)
- PPO quickly (within 100 episodes) reaches positive rewards
- DQN shows no convergence—remains consistently negative
- PPO Trial 4 efficiency: **0.0071 reward/timestep**
- DQN best efficiency: **-0.00098 reward/timestep** (7.2x worse)

## Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'cyberbattle'`**
```bash
pip install git+https://github.com/microsoft/cyberbattlesim.git
```

**Issue: PyTorch installation fails or slow**
```bash
# CPU-only version (faster installation):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Issue: Optuna trials fail or hang**
- Ensure sufficient RAM (at least 4GB for 100K timestep trials)
- Reduce `timesteps_per_trial` to 50000 or less
- Check CyberBattleSim compatibility: `python -c "from cyberbattle._env.toy_ctf import toy_ctf_params; print('OK')"`

**Issue: Environment hangs or crashes**
- Check available RAM and CPU resources
- Try reducing max_actions in `src/environment_wrapper.py`
- Verify TensorFlow/PyTorch installation with `python -c "import torch; print(torch.__version__)"`

## Results Summary

### Overall Winner: PPO Algorithm
- **Best Mean Reward:** 709.96 (PPO Trial 4)
- **Recommendation:** Use PPO Trial 4 configuration for deployment
- **Expected Performance:** ~710 mean reward

### Model Files Location
- **Best PPO Model:** `logs/optuna_ppo_trial_4/ppo_checkpoints/`
- **Configuration:** `logs/phase3_best_ppo_config.json`
- **Detailed Results:** `logs/optuna_ppo_results.json`

## File Descriptions

### Source Code (`src/`)
- `environment_wrapper.py` - Wraps CyberBattleSim as Gymnasium environment
- `dqn_agent.py` - Deep Q-Network implementation with stable-baselines3
- `ppo_agent.py` - Proximal Policy Optimization implementation
- `logger.py` - Metrics and training progress logging

### Training Scripts (`scripts/`)
- `train_dqn.py` - Baseline DQN training (Phase 2)
- `train_ppo.py` - Baseline PPO training (Phase 2)
- `optuna_tuning.py` - DQN hyperparameter optimization (Phase 3)
- `optuna_ppo_tuning.py` - PPO hyperparameter optimization (Phase 3)
- `run_optuna_phase3.py` - Complete Phase 3 workflow
- `phase3_summary.py` - Generate Phase 3 statistics
- `analyze_phase3.py` - Comparative analysis of Phase 3 results
- `phase4_visualization.py` - Generate Phase 4 plots and visualizations
- `phase4_champions.py` - Evaluate champion model performance

### Analysis Documents
- `PHASE3_ANALYSIS.md` - Comprehensive Phase 3 hyperparameter analysis
- `logs/phase4_visualization_summary.txt` - Phase 4 visualization insights

## Next Steps

For further improvements:
1. Fine-tune PPO Trial 4 configuration with smaller learning rates
2. Implement curriculum learning to gradually increase environment difficulty
3. Test on larger CyberBattleSim environments (CyberBattle)
4. Implement policy distillation for deployment-friendly models
venv/
.venv
*.egg-info/
dist/
build/

# Models & Data
models/*.pkl
models/*.zip
logs/*.csv
logs/*.txt

# IDEs
.vscode/
.idea/
*.swp

# OS
.DS_Store


