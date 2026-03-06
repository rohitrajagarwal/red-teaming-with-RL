# RL-Based Cyber Exploitation: Hyperparameter Optimization

## Project Overview

This project implements RL algorithms (DQN and PPO) to optimize autonomous cyber exploitation in a simulated network environment. The primary goal is to compare exploration-exploitation strategies and hyperparameter tuning approaches using CyberBattleSim and Optuna.

## Phase 1: Environment Setup & Sanity Check

### Quick Start

1. **Create conda environment:**
   ```bash
   conda create -n rl-cybersim python=3.11 -y
   conda activate rl-cybersim
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test environment initialization:**
   ```bash
   python scripts/test_environment.py
   ```

4. **Run random agent sanity check:**
   ```bash
   python scripts/random_agent_test.py
   ```

### Verification Checklist

After running the above commands, verify:

- ✅ All imports work without errors
- ✅ ToyCtf environment initializes
- ✅ Random agent completes 1,000 steps
- ✅ No crashes or hanging processes

### Project Structure

```
Group project/
├── src/
│   └── environment_wrapper.py          # Gymnasium wrapper for CyberBattleSim
├── scripts/
│   ├── test_environment.py             # Task 3: Environment initialization
│   └── random_agent_test.py            # Task 4: Random agent test
├── notebooks/                          # Jupyter experiments (Phase 2+)
├── configs/                            # Hyperparameter configs (Phase 3)
├── models/                             # Saved trained models (Phase 2+)
├── logs/                               # Training logs and metrics (Phase 2+)
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore patterns
└── README.md                           # This file
```

### Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'cyberbattle'`**
```bash
pip install git+https://github.com/microsoft/cyberbattlesim.git
```

**Issue: PyTorch installation fails or slow**
```bash
# CPU-only version (faster installation):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Issue: Environment hangs or crashes**
- Check available RAM and CPU resources
- Verify CyberBattleSim compatibility: `python -c "from cyberbattle._env.toy_ctf import toy_ctf_params; print('OK')"`
- Try reducing `max_actions` in `src/environment_wrapper.py`

### Next Steps

After Phase 1 succeeds, proceed to **Phase 2: Baseline Models** to train DQN and PPO with default hyperparameters.
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


