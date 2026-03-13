"""
Phase 3: PPO Hyperparameter Tuning with Optuna

Systematically search for optimal PPO hyperparameters using Optuna's TPE sampler.
Focuses on 5 trials with carefully chosen search ranges.

Hyperparameters being optimized:
1. learning_rate: Policy and value network learning rate
2. ent_coef (β): Entropy coefficient for exploration maintenance
3. clip_range: PPO clipping range for gradient updates

Execution:
    python scripts/optuna_ppo_tuning.py --n_trials 5
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cyberbattle
from cyberbattle._env.cyberbattle_chain import CyberBattleChain
from src.environment_wrapper import CyberBattleWrapper
from src.ppo_agent import PPOAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

CHAIN_SIZE = 4   # 6 nodes (start + 4 chain + 1 extra)
NUM_NODES = 6


def create_env():
    """Create the CyberBattle environment."""
    cyber_env = CyberBattleChain(size=CHAIN_SIZE, attacker_goal=None)
    env = CyberBattleWrapper(cyber_env, max_episode_steps=2000, num_nodes=NUM_NODES)
    return env


def objective_ppo(trial: optuna.Trial, timesteps: int = 100000) -> float:
    """
    PPO objective function for Optuna.
    
    Optimized search ranges based on Phase 2 baseline (47.3% success):
    
    Hyperparameters to optimize:
    - learning_rate: Controls policy gradient update magnitude
      Range: [1e-5, 1e-3] - Phase 2 baseline is 3e-4
    - ent_coef (β): Entropy coefficient for exploration maintenance
      Range: [1e-3, 1e-1] - Phase 2 baseline is 0.02 (2e-2)
    - clip_range: PPO gradient clipping
      Range: [0.05, 0.3] - Phase 2 baseline is 0.1
    
    Args:
        trial: Optuna trial object
        timesteps: Total training timesteps (100K for full evaluation)
    
    Returns:
        Objective value (higher is better: mean_reward + success_bonus)
    """
    
    # Suggest hyperparameters with optimized ranges
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-3, 1e-1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.05, 0.3)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"PPO Trial {trial.number}")
    logger.info(f"{'='*70}")
    logger.info(f"Learning Rate: {learning_rate:.3e}")
    logger.info(f"Entropy Coefficient (β): {ent_coef:.3e}")
    logger.info(f"Clip Range: {clip_range:.3f}")
    
    try:
        # Create environment
        env = create_env()
        
        # Initialize agent with fixed supporting hyperparameters from Phase 2
        agent = PPOAgent(
            env=env,
            log_dir=f"logs/optuna_ppo_trial_{trial.number}",
            # Optimized hyperparameters
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            clip_range=clip_range,
            # Fixed hyperparameters (proven from Phase 2)
            n_steps=2048,           # Rollout buffer size
            batch_size=64,          # Batch size per gradient update
            n_epochs=10,            # PPO updates per rollout
            gamma=0.99,             # Discount factor
            gae_lambda=0.95,        # GAE lambda for advantage estimation
            vf_coef=0.5,            # Value function loss coefficient
            max_grad_norm=0.5,      # Gradient clipping
        )
        
        # Train
        logger.info(f"Training PPO for {timesteps:,} steps...")
        stats = agent.train(total_timesteps=timesteps)
        
        # Evaluate on 10 episodes
        logger.info("Evaluating PPO (10 episodes, deterministic)...")
        eval_rewards = []
        eval_nodes = []
        
        for ep in range(10):
            obs, _ = env.reset()
            episode_reward = 0
            nodes_discovered = set()
            done = False
            step = 0
            
            while not done and step < 500:
                action = agent.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step += 1
                
                # Track node discovery
                if "discovered_nodes" in info:
                    nodes_discovered.update(info["discovered_nodes"])
            
            eval_rewards.append(episode_reward)
            eval_nodes.append(len(nodes_discovered))
            
            logger.info(f"  Eval Ep {ep+1:2d}: Reward={episode_reward:7.1f}, Nodes={len(nodes_discovered)}/6")
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        mean_nodes = np.mean(eval_nodes)
        
        logger.info(f"\nPPO Trial {trial.number} Results:")
        logger.info(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"  Mean Nodes Discovered: {mean_nodes:.2f}/6")
        logger.info(f"  Success Rate: {(mean_nodes/6.0)*100:.1f}%")
        
        # Composite objective: reward + success bonus
        success_rate = mean_nodes / 6.0
        success_bonus = success_rate * 100  # Bonus for discovering all nodes
        objective_value = mean_reward + (success_bonus * 0.5)
        
        logger.info(f"  Objective Value: {objective_value:.2f}")
        
        # Store trial info for analysis
        trial.set_user_attr("mean_reward", float(mean_reward))
        trial.set_user_attr("std_reward", float(std_reward))
        trial.set_user_attr("mean_nodes", float(mean_nodes))
        trial.set_user_attr("success_rate", float(success_rate))
        
        env.close()
        return objective_value
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return float('-inf')


def run_ppo_optuna_study(n_trials: int = 5):
    """
    Run Optuna hyperparameter search for PPO.
    
    Args:
        n_trials: Number of trials to run (5 recommended for balanced search)
    """
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# Phase 3: PPO Hyperparameter Tuning with Optuna")
    logger.info(f"# Trials: {n_trials}")
    logger.info(f"# Timesteps per trial: 100,000")
    logger.info(f"# Total computation: ~{n_trials * 6} hours")
    logger.info(f"{'#'*70}\n")
    
    # Create sampler (TPE = Tree-structured Parzen Estimator)
    # TPE is Bayesian optimization that learns from previous trials
    sampler = TPESampler(
        seed=42,
        n_startup_trials=2,  # Random exploration for first 2 trials
        n_ei_candidates=24,  # Candidates for EI calculation
    )
    
    # Create study with median pruner (stop underperforming trials early)
    pruner = MedianPruner(
        n_startup_trials=2,
        n_warmup_steps=0,
    )
    
    # Create study
    study_name = f"optuna_ppo_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///logs/optuna_ppo.db"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize objective (reward + success bonus)
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=False,
    )
    
    # Optimize
    logger.info(f"Starting TPE-based optimization with {n_trials} trials...\n")
    study.optimize(objective_ppo, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    # Get best trial
    best_trial = study.best_trial
    
    logger.info(f"\n{'='*70}")
    logger.info(f"OPTIMIZATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Best Trial: #{best_trial.number}")
    logger.info(f"Best Objective Value: {best_trial.value:.4f}")
    logger.info(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.3e}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Get top 5 trials
    logger.info(f"\n{'='*70}")
    logger.info(f"Top 5 PPO Configurations (sorted by objective value)")
    logger.info(f"{'='*70}")
    
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
    top_configs = []
    
    logger.info(f"\n{'Rank':<5} {'Objective':<12} {'LR':<12} {'Ent Coef':<12} {'Clip':<8} {'Reward':<10} {'Nodes':<8}")
    logger.info("-" * 80)
    
    for rank, trial in enumerate(sorted_trials[:5], 1):
        if trial.value is None:
            continue
            
        params = trial.params
        attrs = trial.user_attrs
        
        lr = params.get('learning_rate', 0)
        ent = params.get('ent_coef', 0)
        clip = params.get('clip_range', 0)
        reward = attrs.get('mean_reward', 0)
        nodes = attrs.get('mean_nodes', 0)
        
        logger.info(
            f"{rank:<5} {trial.value:<12.4f} {lr:<12.3e} {ent:<12.3e} {clip:<8.4f} "
            f"{reward:<10.2f} {nodes:<8.2f}"
        )
        
        top_configs.append({
            "rank": rank,
            "trial_number": trial.number,
            "objective_value": float(trial.value),
            "hyperparameters": trial.params,
            "metrics": trial.user_attrs,
        })
    
    # Save results
    results = {
        "algorithm": "ppo",
        "phase": 3,
        "n_trials": n_trials,
        "best_trial": {
            "number": best_trial.number,
            "objective_value": float(best_trial.value),
            "hyperparameters": best_trial.params,
            "metrics": best_trial.user_attrs,
        },
        "top_5_configs": top_configs,
        "study_name": study_name,
        "storage": storage,
        "timestamp": datetime.now().isoformat(),
        "search_space": {
            "learning_rate": "[1e-5, 1e-3] log scale",
            "ent_coef": "[1e-3, 1e-1] log scale",
            "clip_range": "[0.05, 0.3] linear scale",
        },
        "fixed_hyperparameters": {
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    }
    
    results_file = f"logs/optuna_ppo_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {results_file}")
    
    # Save best config for Phase 4
    best_config = {
        "algorithm": "ppo",
        "phase": 3,
        "source": "optuna_best_trial",
        "trial_number": best_trial.number,
        "objective_value": float(best_trial.value),
        "training_metrics": best_trial.user_attrs,
        "hyperparameters": {
            "learning_rate": float(best_trial.params.get("learning_rate", 3e-4)),
            "ent_coef": float(best_trial.params.get("ent_coef", 0.02)),
            "clip_range": float(best_trial.params.get("clip_range", 0.1)),
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    }
    
    best_config_file = "logs/phase3_best_ppo_config.json"
    with open(best_config_file, "w") as f:
        json.dump(best_config, f, indent=2)
    
    logger.info(f"✓ Best PPO config saved to: {best_config_file}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Phase 3 (PPO) Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Ready for Phase 4: Champion Model Training\n")
    
    return study, best_trial


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: PPO Hyperparameter Tuning with Optuna"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of trials to run (default: 5)",
    )
    
    args = parser.parse_args()
    
    # Create logs directory if needed
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("\n" + "+"*70)
    logger.info("+ Starting PPO Hyperparameter Optimization")
    logger.info("+"*70)
    
    ppo_study, ppo_best = run_ppo_optuna_study(n_trials=args.n_trials)
    
    logger.info("\n" + "="*70)
    logger.info("PPO Hyperparameter Tuning Complete!")
    logger.info("="*70)
    logger.info("Next: Run Phase 4 for champion model training")
    logger.info("  Command: python scripts/phase4_champions.py --algorithm both")


if __name__ == "__main__":
    main()
