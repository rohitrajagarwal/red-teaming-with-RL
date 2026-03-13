"""
Phase 3 Summary: Extract and analyze Optuna hyperparameter tuning results.

Summarizes the 6 DQN trials completed and extracts best hyperparameters
for Phase 4 champion training.
"""

import json
from pathlib import Path
import pandas as pd

def extract_trial_metrics(trial_dir):
    """Extract key metrics from a trial's metrics file."""
    metrics_file = trial_dir / "dqn_metrics.json"
    
    if not metrics_file.exists():
        return None
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    stats = metrics.get("statistics", {})
    
    return {
        "mean_reward": stats.get("mean_reward", 0),
        "max_reward": stats.get("max_reward", 0),
        "std_reward": stats.get("std_reward", 0),
        "mean_length": stats.get("mean_length", 0),
        "total_episodes": metrics.get("total_episodes", 0),
        "total_steps": metrics.get("total_steps", 0),
    }

def main():
    logs_dir = Path("logs")
    
    # Find all DQN trial directories
    dqn_trials = sorted([d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("optuna_dqn_trial_")])
    
    print("\n" + "="*80)
    print("PHASE 3 SUMMARY: Optuna Hyperparameter Tuning Results")
    print("="*80 + "\n")
    
    print(f"Completed DQN Trials: {len(dqn_trials)}\n")
    
    # Load DQN results JSON
    dqn_results_file = logs_dir / "optuna_dqn_results.json"
    if dqn_results_file.exists():
        with open(dqn_results_file) as f:
            dqn_results = json.load(f)
    else:
        dqn_results = {}
    
    # Extract metrics for each trial
    trial_data = []
    for trial_dir in dqn_trials:
        trial_num = int(trial_dir.name.split("_")[-1])
        metrics = extract_trial_metrics(trial_dir)
        
        if metrics:
            # Get hyperparameters from results JSON
            top_configs = dqn_results.get("top_5_configs", [])
            trial_config = None
            for config in top_configs:
                if config.get("rank") == trial_num + 1:
                    trial_config = config.get("hyperparameters", {})
                    break
            
            trial_data.append({
                "trial": trial_num,
                "mean_reward": metrics["mean_reward"],
                "max_reward": metrics["max_reward"],
                "episodes": metrics["total_episodes"],
                "learning_rate": trial_config.get("learning_rate", "N/A") if trial_config else "N/A",
                "exploration_fraction": trial_config.get("exploration_fraction", "N/A") if trial_config else "N/A",
                "epsilon_final": trial_config.get("epsilon_final", "N/A") if trial_config else "N/A",
            })
    
    if trial_data:
        df = pd.DataFrame(trial_data)
        print("Trial Performance Summary:")
        print("-" * 80)
        
        # Sort by mean_reward
        df_sorted = df.sort_values("mean_reward", ascending=False)
        
        for idx, row in df_sorted.iterrows():
            print(f"\nTrial {int(row['trial'])}:")
            print(f"  Mean Reward: {row['mean_reward']:.2f}")
            print(f"  Max Reward:  {row['max_reward']:.2f}")
            print(f"  Episodes:    {int(row['episodes'])}")
            print(f"  Learning Rate: {row['learning_rate']}")
            print(f"  Exploration Fraction: {row['exploration_fraction']}")
            print(f"  Epsilon Final: {row['epsilon_final']}")
        
        # Identify best trial
        best_trial = df_sorted.iloc[0]
        print("\n" + "="*80)
        print(f"BEST DQN CONFIGURATION (Trial {int(best_trial['trial'])}):")
        print("="*80)
        print(f"Mean Reward: {best_trial['mean_reward']:.2f}")
        print(f"Learning Rate: {best_trial['learning_rate']}")
        print(f"Exploration Fraction: {best_trial['exploration_fraction']}")
        print(f"Epsilon Final: {best_trial['epsilon_final']}")
        
        # Save best config
        best_config = {
            "algorithm": "dqn",
            "source": "optuna_phase3",
            "trial_number": int(best_trial['trial']),
            "mean_reward": float(best_trial['mean_reward']),
            "hyperparameters": {
                "learning_rate": float(best_trial['learning_rate']) if isinstance(best_trial['learning_rate'], (int, float)) else best_trial['learning_rate'],
                "exploration_fraction": float(best_trial['exploration_fraction']) if isinstance(best_trial['exploration_fraction'], (int, float)) else best_trial['exploration_fraction'],
                "exploration_initial_eps": 0.99,  # Fixed
                "exploration_final_eps": float(best_trial['epsilon_final']) if isinstance(best_trial['epsilon_final'], (int, float)) else best_trial['epsilon_final'],
                "polynomial_power": 1.5,
                "buffer_size": 50000,
                "batch_size": 32,
                "gamma": 0.99,
                "train_freq": 4,
            }
        }
        
        best_config_file = logs_dir / "phase3_best_dqn_config.json"
        with open(best_config_file, "w") as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\nBest config saved to: {best_config_file}")
    
    print("\n" + "="*80)
    print("PPO: Using Phase 2 baseline (47.3% success)")
    print("="*80)
    print("Learning Rate: 3e-4")
    print("Entropy Coefficient (β): 0.02")
    print("Clip Range: 0.1")
    
    # Save PPO baseline config
    ppo_config = {
        "algorithm": "ppo",
        "source": "phase2_baseline",
        "success_rate": 0.473,
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
        json.dump(ppo_config, f, indent=2)
    
    print(f"\nPPO config saved to: {ppo_config_file}")
    
    print("\n" + "="*80)
    print("Ready for Phase 4: Champion Model Training")
    print("="*80)
    print(f"DQN Champion Config: {best_config_file}")
    print(f"PPO Champion Config: {ppo_config_file}")
    print("\nNext: Train champions for 100K steps and evaluate.")


if __name__ == "__main__":
    main()
