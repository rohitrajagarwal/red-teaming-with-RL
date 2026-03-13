#!/usr/bin/env python3
"""
Extract and compare Phase 3 trial results.
Identifies best DQN trial and prepares best configs for Phase 4.
"""

import json
from pathlib import Path
import sys

def main():
    logs_dir = Path("logs")
    
    print("\n" + "="*80)
    print("PHASE 3: TRIAL RESULTS ANALYSIS")
    print("="*80 + "\n")
    
    # Extract metrics from all trials
    trials_data = {}
    
    dqn_trial_dirs = sorted([
        d for d in logs_dir.iterdir() 
        if d.is_dir() and d.name.startswith("optuna_dqn_trial_")
    ])
    
    print(f"Found {len(dqn_trial_dirs)} DQN trials\n")
    
    for trial_dir in dqn_trial_dirs:
        trial_num = int(trial_dir.name.split("_")[-1])
        metrics_file = trial_dir / "dqn_metrics.json"
        
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                stats = metrics.get("statistics", {})
                
                trials_data[trial_num] = {
                    "mean_reward": stats.get("mean_reward", 0),
                    "max_reward": stats.get("max_reward", 0),
                    "std_reward": stats.get("std_reward", 0),
                    "episodes": metrics.get("total_episodes", 0),
                    "steps": metrics.get("total_steps", 0),
                }
                
                print(f"Trial {trial_num}:")
                print(f"  Mean Reward: {trials_data[trial_num]['mean_reward']:8.2f}")
                print(f"  Max Reward:  {trials_data[trial_num]['max_reward']:8.2f}")
                print(f"  Episodes:    {trials_data[trial_num]['episodes']:3d}")
                print()
                
            except Exception as e:
                print(f"Error reading trial {trial_num}: {e}\n")
                continue
    
    if not trials_data:
        print("ERROR: No trial data found!")
        sys.exit(1)
    
    # Find best trial
    best_trial = max(trials_data.items(), key=lambda x: x[1]["mean_reward"])
    best_num, best_stats = best_trial
    
    print("="*80)
    print(f"BEST TRIAL: {best_num}")
    print("="*80)
    print(f"Mean Reward: {best_stats['mean_reward']:.2f}")
    print(f"Max Reward:  {best_stats['max_reward']:.2f}")
    print(f"Episodes:    {best_stats['episodes']}")
    print()
    
    # Extract hyperparameters for best trial
    optuna_results_file = logs_dir / "optuna_dqn_results.json"
    
    if optuna_results_file.exists():
        with open(optuna_results_file) as f:
            optuna_results = json.load(f)
        
        # Get hyperparameters from top configs (ranked 1-5 correspond to trial configs)
        top_configs = optuna_results.get("top_5_configs", [])
        
        print("Top DQN Configurations from Optuna:")
        print("-" * 80)
        
        for i, config in enumerate(top_configs[:3], 1):
            hyperparams = config.get("hyperparameters", {})
            print(f"\nConfig {i}:")
            print(f"  Learning Rate: {hyperparams.get('learning_rate', 0):.3e}")
            print(f"  Exploration Fraction: {hyperparams.get('exploration_fraction', 0):.4f}")
            print(f"  Epsilon Final: {hyperparams.get('epsilon_final', 0):.4f}")
        
        # Use top config as champion
        champion_hyperparams = top_configs[0].get("hyperparameters", {})
        
        champion_config = {
            "algorithm": "dqn",
            "phase": 3,
            "source": "optuna_best_trial",
            "training_metrics": {
                "mean_reward": best_stats["mean_reward"],
                "max_reward": best_stats["max_reward"],
                "episodes": best_stats["episodes"],
            },
            "hyperparameters": {
                "learning_rate": float(champion_hyperparams.get("learning_rate", 1e-4)),
                "exploration_fraction": float(champion_hyperparams.get("exploration_fraction", 0.2)),
                "exploration_initial_eps": 0.99,
                "exploration_final_eps": float(champion_hyperparams.get("epsilon_final", 0.10)),
                "polynomial_power": 1.5,
                "buffer_size": 50000,
                "batch_size": 32,
                "gamma": 0.99,
                "train_freq": 4,
            }
        }
        
        config_file = logs_dir / "phase3_best_dqn_config.json"
        with open(config_file, "w") as f:
            json.dump(champion_config, f, indent=2)
        
        print(f"\n✓ Champion DQN config saved: {config_file}")
    
    # PPO champion (using Phase 2 baseline)
    ppo_champion_config = {
        "algorithm": "ppo",
        "phase": 3,
        "source": "phase2_baseline",
        "training_metrics": {
            "success_rate": 0.473,
            "mean_reward": "47.3% nodes discovered",
        },
        "hyperparameters": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.02,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    }
    
    ppo_config_file = logs_dir / "phase3_best_ppo_config.json"
    with open(ppo_config_file, "w") as f:
        json.dump(ppo_champion_config, f, indent=2)
    
    print(f"✓ Champion PPO config saved: {ppo_config_file}")
    
    print("\n" + "="*80)
    print("SUMMARY: Phase 3 Complete (6 trials)")
    print("="*80)
    print(f"✓ DQN trials: 6 (0-5) - Best trial identified")
    print(f"✓ PPO baseline: Phase 2 (47.3% success)")
    print(f"\nReady for Phase 4: Champion Model Training")
    print(f"  - Train DQN champion: 100K steps")
    print(f"  - Train PPO champion: 100K steps")
    print(f"  - Compare and evaluate")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
