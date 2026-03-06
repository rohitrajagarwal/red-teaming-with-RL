"""
Phase 3: Hyperparameter Tuning with Optuna

Systematically search for optimal DQN and PPO hyperparameters using Optuna's TPE sampler.

Execution:
    python scripts/optuna_tuning.py --algorithm dqn --n_trials 25
    python scripts/optuna_tuning.py --algorithm ppo --n_trials 25
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler
import gymnasium as gym
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
from src.dqn_agent import DQNAgent
from src.ppo_agent import PPOAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def create_env():
    """Create the CyberBattle environment."""
    base_env = CyberBattleChain(size=4, attacker_goal=None)
    env = CyberBattleWrapper(base_env, max_episode_steps=2000, num_nodes=6)
    return env


def objective_dqn(trial: optuna.Trial, timesteps: int = 100000) -> float:
    """
    DQN objective function for Optuna.
    
    Hyperparameters to optimize:
    - learning_rate: Controls Q-network update speed
    - exploration_fraction: Duration of exploration phase
    - epsilon_final: Minimum exploration rate
    
    Args:
        trial: Optuna trial object
        timesteps: Total training timesteps (100K for full evaluation)
    
    Returns:
        Mean cumulative reward over last 10 episodes
    """
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.05, 0.30)
    epsilon_final = trial.suggest_float("epsilon_final", 0.01, 0.10)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"DQN Trial {trial.number}")
    logger.info(f"{'='*70}")
    logger.info(f"Learning Rate: {learning_rate:.2e}")
    logger.info(f"Exploration Fraction: {exploration_fraction:.3f}")
    logger.info(f"Epsilon Final: {epsilon_final:.3f}")
    
    try:
        # Create environment
        env = create_env()
        
        # Initialize agent
        agent = DQNAgent(
            env=env,
            log_dir=f"logs/optuna_dqn_trial_{trial.number}",
            learning_rate=learning_rate,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=0.99,  # Fixed from Phase 2
            exploration_final_eps=epsilon_final,
            buffer_size=50000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            polynomial_power=1.5,
        )
        
        # Train
        stats = agent.train(total_timesteps=timesteps)
        
        # Evaluate
        eval_stats = agent.evaluate(num_episodes=10, deterministic=True)
        
        # Primary metric: mean reward
        mean_reward = eval_stats["eval_mean_reward"]
        
        # Secondary metric: success rate (6/6 nodes)
        mean_nodes = eval_stats["eval_mean_nodes"]
        success_bonus = (mean_nodes / 6.0) * 100  # Bonus if discovering all nodes
        
        # Combined objective
        objective_value = mean_reward + (success_bonus * 0.5)
        
        logger.info(f"Mean Reward: {mean_reward:.2f}")
        logger.info(f"Mean Nodes: {mean_nodes:.2f}/6")
        logger.info(f"Objective Value: {objective_value:.2f}")
        
        # Store trial info
        trial.set_user_attr("mean_reward", float(mean_reward))
        trial.set_user_attr("mean_nodes", float(mean_nodes))
        trial.set_user_attr("success_bonus", float(success_bonus))
        
        env.close()
        return objective_value
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return float('-inf')


def objective_ppo(trial: optuna.Trial, timesteps: int = 100000) -> float:
    """
    PPO objective function for Optuna.
    
    Hyperparameters to optimize:
    - learning_rate: Controls policy gradient update magnitude
    - ent_coef (β): Entropy coefficient for exploration maintenance
    - clip_range: PPO gradient clipping (optional)
    
    Args:
        trial: Optuna trial object
        timesteps: Total training timesteps (100K for full evaluation)
    
    Returns:
        Mean cumulative reward over last 10 episodes
    """
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-3, 1e-1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.05, 0.3)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"PPO Trial {trial.number}")
    logger.info(f"{'='*70}")
    logger.info(f"Learning Rate: {learning_rate:.2e}")
    logger.info(f"Entropy Coefficient (β): {ent_coef:.2e}")
    logger.info(f"Clip Range: {clip_range:.3f}")
    
    try:
        # Create environment
        env = create_env()
        
        # Initialize agent
        agent = PPOAgent(
            env=env,
            log_dir=f"logs/optuna_ppo_trial_{trial.number}",
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        
        # Train
        stats = agent.train(total_timesteps=timesteps)
        
        # Evaluate
        eval_stats = agent.evaluate(num_episodes=10, deterministic=True)
        
        # Primary metric: mean reward
        mean_reward = eval_stats["eval_mean_reward"]
        
        # Secondary metric: success rate (6/6 nodes)
        mean_nodes = eval_stats["eval_mean_nodes"]
        success_bonus = (mean_nodes / 6.0) * 100  # Bonus if discovering all nodes
        
        # Combined objective
        objective_value = mean_reward + (success_bonus * 0.5)
        
        logger.info(f"Mean Reward: {mean_reward:.2f}")
        logger.info(f"Mean Nodes: {mean_nodes:.2f}/6")
        logger.info(f"Objective Value: {objective_value:.2f}")
        
        # Store trial info
        trial.set_user_attr("mean_reward", float(mean_reward))
        trial.set_user_attr("mean_nodes", float(mean_nodes))
        trial.set_user_attr("success_bonus", float(success_bonus))
        
        env.close()
        return objective_value
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return float('-inf')


def run_optuna_study(algorithm: str, n_trials: int = 25, n_jobs: int = 1):
    """
    Run Optuna hyperparameter search for the specified algorithm.
    
    Args:
        algorithm: "dqn" or "ppo"
        n_trials: Number of trials to run (25 recommended)
        n_jobs: Number of parallel jobs (1 = sequential, >1 = parallel)
    """
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# Phase 3: Optuna Hyperparameter Tuning")
    logger.info(f"# Algorithm: {algorithm.upper()}")
    logger.info(f"# Trials: {n_trials}")
    logger.info(f"# Timesteps per trial: 100,000")
    logger.info(f"{'#'*70}\n")
    
    # Create sampler (TPE = Tree-structured Parzen Estimator)
    sampler = TPESampler(seed=42)
    
    # Create study
    study_name = f"optuna_{algorithm}_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///logs/optuna_{algorithm}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=False,
    )
    
    # Select objective
    if algorithm == "dqn":
        objective = objective_dqn
    elif algorithm == "ppo":
        objective = objective_ppo
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")
    
    # Optimize
    logger.info(f"Starting optimization with {n_trials} trials...\n")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    # Get best trial
    best_trial = study.best_trial
    
    logger.info(f"\n{'='*70}")
    logger.info(f"OPTIMIZATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Best Trial: #{best_trial.number}")
    logger.info(f"Best Objective Value: {best_trial.value:.4f}")
    logger.info(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Get top 5 trials
    logger.info(f"\nTop 5 Configurations:")
    logger.info(f"{'Rank':<5} {'Objective':<12} {'Hyperparameters':<50}")
    logger.info(f"{'-'*70}")
    
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    top_configs = []
    
    for rank, trial in enumerate(sorted_trials[:5], 1):
        params_str = ", ".join([f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in trial.params.items()])
        logger.info(f"{rank:<5} {trial.value:<12.4f} {params_str:<50}")
        top_configs.append({
            "rank": rank,
            "objective": float(trial.value),
            "hyperparameters": trial.params,
            "user_attrs": trial.user_attrs,
        })
    
    # Save results
    results = {
        "algorithm": algorithm,
        "n_trials": n_trials,
        "best_trial": {
            "number": best_trial.number,
            "objective_value": float(best_trial.value),
            "hyperparameters": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        },
        "top_5_configs": top_configs,
        "study_name": study_name,
        "storage": storage,
        "timestamp": datetime.now().isoformat(),
    }
    
    results_file = f"logs/optuna_{algorithm}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    
    return study, best_trial


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Optuna Hyperparameter Tuning"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["dqn", "ppo", "both"],
        default="both",
        help="Algorithm to optimize (default: both)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=25,
        help="Number of trials per algorithm (default: 25)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, sequential)",
    )
    
    args = parser.parse_args()
    
    # Create logs directory if needed
    Path("logs").mkdir(exist_ok=True)
    
    if args.algorithm in ["dqn", "both"]:
        logger.info("\n" + "+"*70)
        logger.info("+ Starting DQN Optimization")
        logger.info("+"*70)
        dqn_study, dqn_best = run_optuna_study("dqn", n_trials=args.n_trials, n_jobs=args.n_jobs)
    
    if args.algorithm in ["ppo", "both"]:
        logger.info("\n" + "+"*70)
        logger.info("+ Starting PPO Optimization")
        logger.info("+"*70)
        ppo_study, ppo_best = run_optuna_study("ppo", n_trials=args.n_trials, n_jobs=args.n_jobs)
    
    logger.info("\n" + "="*70)
    logger.info("Phase 3 Complete!")
    logger.info("="*70)
    logger.info("Results saved in logs/optuna_*.json")


if __name__ == "__main__":
    main()
