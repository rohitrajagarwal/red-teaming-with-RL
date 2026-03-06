#!/usr/bin/env python3
"""
Phase 3: Optuna-based Hyperparameter Optimization
Searches for optimal hyperparameters for DQN and PPO agents.

Objectives:
- DQN: Optimize learning_rate, exploration_fraction, epsilon_final
- PPO: Optimize learning_rate, ent_coef, clip_range

Each algorithm: 25 trials, 100K timesteps per trial
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment_wrapper import create_cyberbattle_env
from src.logger import MetricsLogger

# Configuration
TIMESTEPS_PER_TRIAL = 100_000
EVAL_EPISODES = 5
EVAL_FREQUENCY = 10_000
DQN_N_TRIALS = 25
PPO_N_TRIALS = 25

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
OPTUNA_DIR = LOGS_DIR / "optuna_phase3"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)


class OptunaCallback:
    """Callback to track metrics during training for pruning."""
    def __init__(self):
        self.eval_rewards = []
        self.timesteps = []
    
    def update(self, timestep, eval_reward):
        self.timesteps.append(timestep)
        self.eval_rewards.append(eval_reward)
    
    def get_mean_reward(self):
        if not self.eval_rewards:
            return -np.inf
        return np.mean(self.eval_rewards[-3:]) if len(self.eval_rewards) >= 3 else np.mean(self.eval_rewards)


def evaluate_policy(model, env, n_episodes=EVAL_EPISODES):
    """Evaluate policy on environment."""
    episode_rewards = []
    episode_successes = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        success = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            if info.get("success", False):
                success = True
        
        episode_rewards.append(episode_reward)
        episode_successes.append(1 if success else 0)
    
    return np.mean(episode_rewards), np.mean(episode_successes)


def objective_dqn(trial: optuna.Trial) -> float:
    """Objective function for DQN hyperparameter search."""
    print(f"\n{'='*70}")
    print(f"DQN Trial {trial.number + 1}/{DQN_N_TRIALS}")
    print(f"{'='*70}")
    
    # Hyperparameter search space
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    exploration_fraction = trial.suggest_float('exp_frac', 0.1, 0.5)
    epsilon_final = trial.suggest_float('eps_final', 0.01, 0.2)
    buffer_size = trial.suggest_int('buffer_size', 10000, 100000, step=10000)
    
    print(f"Hyperparameters:")
    print(f"  learning_rate: {learning_rate:.6f}")
    print(f"  exploration_fraction: {exploration_fraction:.3f}")
    print(f"  epsilon_final: {epsilon_final:.3f}")
    print(f"  buffer_size: {buffer_size}")
    
    try:
        # Create environment
        env = create_cyberbattle_env()
        
        # Create DQN model
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=epsilon_final,
            buffer_size=buffer_size,
            learning_starts=1000,
            train_freq=4,
            target_update_interval=1000,
            verbose=0,
            device='cpu'
        )
        
        # Callback for periodic evaluation
        eval_env = create_cyberbattle_env()
        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(OPTUNA_DIR / f"dqn_trial_{trial.number}"),
            eval_freq=EVAL_FREQUENCY,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            render=False,
            callback_on_new_best=None
        )
        
        # Train
        print(f"Training for {TIMESTEPS_PER_TRIAL} timesteps...")
        model.learn(total_timesteps=TIMESTEPS_PER_TRIAL, callback=callback, log_interval=10)
        
        # Final evaluation
        mean_reward, success_rate = evaluate_policy(model, eval_env, n_episodes=10)
        
        print(f"Trial Results:")
        print(f"  Mean Reward: {mean_reward:.3f}")
        print(f"  Success Rate: {success_rate:.1%}")
        
        env.close()
        eval_env.close()
        
        # Return success rate as objective (higher is better)
        return success_rate
    
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return -1.0


def objective_ppo(trial: optuna.Trial) -> float:
    """Objective function for PPO hyperparameter search."""
    print(f"\n{'='*70}")
    print(f"PPO Trial {trial.number + 1}/{PPO_N_TRIALS}")
    print(f"{'='*70}")
    
    # Hyperparameter search space
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float('ent_coef', 0.001, 0.1, log=True)
    clip_range = trial.suggest_float('clip_range', 0.05, 0.3)
    n_steps = trial.suggest_int('n_steps', 512, 2048, step=128)
    
    print(f"Hyperparameters:")
    print(f"  learning_rate: {learning_rate:.6f}")
    print(f"  ent_coef: {ent_coef:.6f}")
    print(f"  clip_range: {clip_range:.3f}")
    print(f"  n_steps: {n_steps}")
    
    try:
        # Create environment
        env = create_cyberbattle_env()
        
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_steps=n_steps,
            batch_size=128,
            n_epochs=5,
            verbose=0,
            device='cpu'
        )
        
        # Callback for periodic evaluation
        eval_env = create_cyberbattle_env()
        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(OPTUNA_DIR / f"ppo_trial_{trial.number}"),
            eval_freq=EVAL_FREQUENCY,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            render=False,
            callback_on_new_best=None
        )
        
        # Train
        print(f"Training for {TIMESTEPS_PER_TRIAL} timesteps...")
        model.learn(total_timesteps=TIMESTEPS_PER_TRIAL, callback=callback, log_interval=10)
        
        # Final evaluation
        mean_reward, success_rate = evaluate_policy(model, eval_env, n_episodes=10)
        
        print(f"Trial Results:")
        print(f"  Mean Reward: {mean_reward:.3f}")
        print(f"  Success Rate: {success_rate:.1%}")
        
        env.close()
        eval_env.close()
        
        # Return success rate as objective (higher is better)
        return success_rate
    
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return -1.0


def run_optuna_search():
    """Run full Optuna hyperparameter search for both algorithms."""
    print("\n" + "="*70)
    print("PHASE 3: OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DQN Trials: {DQN_N_TRIALS} × {TIMESTEPS_PER_TRIAL:,} timesteps")
    print(f"PPO Trials: {PPO_N_TRIALS} × {TIMESTEPS_PER_TRIAL:,} timesteps")
    print(f"Output Directory: {OPTUNA_DIR}")
    print("="*70)
    
    results = {}
    
    # ========== DQN Search ==========
    print("\n" + "="*70)
    print("STARTING DQN HYPERPARAMETER SEARCH")
    print("="*70)
    
    sampler_dqn = TPESampler(seed=42, n_startup_trials=5)
    pruner_dqn = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study_dqn = optuna.create_study(
        direction='maximize',
        sampler=sampler_dqn,
        pruner=pruner_dqn,
        study_name='dqn_optimization'
    )
    
    study_dqn.optimize(objective_dqn, n_trials=DQN_N_TRIALS, show_progress_bar=True)
    
    # Extract top 5 DQN trials
    dqn_trials_df = study_dqn.trials_dataframe()
    dqn_trials_df = dqn_trials_df.sort_values('value', ascending=False).head(5)
    
    results['dqn'] = {
        'best_trial': study_dqn.best_trial.number,
        'best_value': study_dqn.best_value,
        'best_params': study_dqn.best_params,
        'top_5_trials': dqn_trials_df.to_dict(orient='records')
    }
    
    print("\n" + "="*70)
    print("DQN TOP 5 CONFIGURATIONS")
    print("="*70)
    print(dqn_trials_df[['number', 'value', 'params_lr', 'params_exp_frac', 'params_eps_final']])
    
    # ========== PPO Search ==========
    print("\n" + "="*70)
    print("STARTING PPO HYPERPARAMETER SEARCH")
    print("="*70)
    
    sampler_ppo = TPESampler(seed=42, n_startup_trials=5)
    pruner_ppo = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study_ppo = optuna.create_study(
        direction='maximize',
        sampler=sampler_ppo,
        pruner=pruner_ppo,
        study_name='ppo_optimization'
    )
    
    study_ppo.optimize(objective_ppo, n_trials=PPO_N_TRIALS, show_progress_bar=True)
    
    # Extract top 5 PPO trials
    ppo_trials_df = study_ppo.trials_dataframe()
    ppo_trials_df = ppo_trials_df.sort_values('value', ascending=False).head(5)
    
    results['ppo'] = {
        'best_trial': study_ppo.best_trial.number,
        'best_value': study_ppo.best_value,
        'best_params': study_ppo.best_params,
        'top_5_trials': ppo_trials_df.to_dict(orient='records')
    }
    
    print("\n" + "="*70)
    print("PPO TOP 5 CONFIGURATIONS")
    print("="*70)
    print(ppo_trials_df[['number', 'value', 'params_lr', 'params_ent_coef', 'params_clip_range']])
    
    # Save results
    results_file = OPTUNA_DIR / "optuna_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save studies as pickle files
    study_dqn.trials_dataframe().to_csv(OPTUNA_DIR / "dqn_trials.csv", index=False)
    study_ppo.trials_dataframe().to_csv(OPTUNA_DIR / "ppo_trials.csv", index=False)
    
    print("\n" + "="*70)
    print("PHASE 3 COMPLETE")
    print("="*70)
    print(f"Results saved to: {results_file}")
    print(f"CSV export: {OPTUNA_DIR / 'dqn_trials.csv'}")
    print(f"CSV export: {OPTUNA_DIR / 'ppo_trials.csv'}")
    print(f"\nDQN Best Configuration:")
    for k, v in study_dqn.best_params.items():
        print(f"  {k}: {v}")
    print(f"  Success Rate: {study_dqn.best_value:.1%}")
    print(f"\nPPO Best Configuration:")
    for k, v in study_ppo.best_params.items():
        print(f"  {k}: {v}")
    print(f"  Success Rate: {study_ppo.best_value:.1%}")
    print("="*70)
    
    return results, study_dqn, study_ppo


if __name__ == "__main__":
    results, study_dqn, study_ppo = run_optuna_search()
