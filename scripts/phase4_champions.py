"""
Phase 4: Champion Model Training

Train the best DQN and PPO configurations from Phase 3 Optuna tuning
on 100,000 steps each, then evaluate and compare.

Usage:
    python scripts/phase4_champions.py --algorithm dqn
    python scripts/phase4_champions.py --algorithm ppo
    python scripts/phase4_champions.py --algorithm both
"""

import json
import argparse
import logging
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment_wrapper import CyberBattleEnvWrapper
from src.dqn_agent import DQNAgent
from src.ppo_agent import PPOAgent
import gymnasium as gym

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def create_env():
    """Create the CyberBattle environment."""
    base_env = gym.make("CyberBattleSim-v0")
    env = CyberBattleEnvWrapper(base_env)
    return env


def load_best_config(algorithm: str):
    """Load best hyperparameters from Phase 3."""
    config_file = Path(f"logs/phase3_best_{algorithm}_config.json")
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}")
        # Return defaults
        if algorithm == "dqn":
            return {
                "learning_rate": 1e-4,
                "exploration_fraction": 0.2,
                "exploration_initial_eps": 0.99,
                "exploration_final_eps": 0.10,
                "polynomial_power": 1.5,
                "buffer_size": 50000,
                "batch_size": 32,
                "gamma": 0.99,
                "train_freq": 4,
            }
        else:  # PPO
            return {
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
    
    with open(config_file) as f:
        result = json.load(f)
    
    return result.get("hyperparameters", {})


def train_dqn_champion(timesteps: int = 100000):
    """Train DQN champion model."""
    logger.info("\n" + "="*70)
    logger.info("Phase 4: DQN Champion Training")
    logger.info("="*70)
    
    config = load_best_config("dqn")
    
    logger.info(f"Hyperparameters:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    try:
        env = create_env()
        
        agent = DQNAgent(
            env=env,
            log_dir="logs/phase4_dqn_champion",
            learning_rate=config.get("learning_rate", 1e-4),
            exploration_fraction=config.get("exploration_fraction", 0.2),
            exploration_initial_eps=config.get("exploration_initial_eps", 0.99),
            exploration_final_eps=config.get("exploration_final_eps", 0.10),
            buffer_size=config.get("buffer_size", 50000),
            batch_size=config.get("batch_size", 32),
            gamma=config.get("gamma", 0.99),
            train_freq=config.get("train_freq", 4),
            polynomial_power=config.get("polynomial_power", 1.5),
        )
        
        logger.info(f"\nTraining DQN for {timesteps:,} steps...")
        stats = agent.train(total_timesteps=timesteps)
        
        # Evaluate
        logger.info("\nEvaluating DQN champion (10 episodes)...")
        eval_episodes = []
        eval_rewards = []
        
        for eval_ep in range(10):
            obs, _ = env.reset()
            episode_reward = 0
            nodes_discovered = set()
            done = False
            
            while not done:
                action = agent.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                # Track node discovery
                if "discovered_nodes" in info:
                    nodes_discovered.update(info["discovered_nodes"])
            
            eval_episodes.append({
                "episode": eval_ep,
                "reward": episode_reward,
                "nodes": len(nodes_discovered),
            })
            eval_rewards.append(episode_reward)
            
            logger.info(f"  Eval Episode {eval_ep+1}: Reward={episode_reward:.1f}, Nodes={len(nodes_discovered)}/6")
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        logger.info(f"\nDQN Champion Results:")
        logger.info(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"  Max Reward: {np.max(eval_rewards):.2f}")
        logger.info(f"  Min Reward: {np.min(eval_rewards):.2f}")
        
        # Save model and results
        model_path = "models/phase4_dqn_champion.zip"
        agent.model.save(model_path)
        logger.info(f"  Model saved to: {model_path}")
        
        results = {
            "algorithm": "dqn",
            "phase": 4,
            "timesteps": timesteps,
            "hyperparameters": config,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "max_reward": float(np.max(eval_rewards)),
            "min_reward": float(np.min(eval_rewards)),
            "eval_episodes": eval_episodes,
        }
        
        results_file = "logs/phase4_dqn_champion_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"  Results saved to: {results_file}")
        
        env.close()
        return results
        
    except Exception as e:
        logger.error(f"DQN training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_ppo_champion(timesteps: int = 100000):
    """Train PPO champion model."""
    logger.info("\n" + "="*70)
    logger.info("Phase 4: PPO Champion Training")
    logger.info("="*70)
    
    config = load_best_config("ppo")
    
    logger.info(f"Hyperparameters:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    try:
        env = create_env()
        
        agent = PPOAgent(
            env=env,
            log_dir="logs/phase4_ppo_champion",
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.1),
            ent_coef=config.get("ent_coef", 0.02),
            vf_coef=config.get("vf_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
        )
        
        logger.info(f"\nTraining PPO for {timesteps:,} steps...")
        stats = agent.train(total_timesteps=timesteps)
        
        # Evaluate
        logger.info("\nEvaluating PPO champion (10 episodes)...")
        eval_episodes = []
        eval_rewards = []
        
        for eval_ep in range(10):
            obs, _ = env.reset()
            episode_reward = 0
            nodes_discovered = set()
            done = False
            
            while not done:
                action = agent.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                # Track node discovery
                if "discovered_nodes" in info:
                    nodes_discovered.update(info["discovered_nodes"])
            
            eval_episodes.append({
                "episode": eval_ep,
                "reward": episode_reward,
                "nodes": len(nodes_discovered),
            })
            eval_rewards.append(episode_reward)
            
            logger.info(f"  Eval Episode {eval_ep+1}: Reward={episode_reward:.1f}, Nodes={len(nodes_discovered)}/6")
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        logger.info(f"\nPPO Champion Results:")
        logger.info(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"  Max Reward: {np.max(eval_rewards):.2f}")
        logger.info(f"  Min Reward: {np.min(eval_rewards):.2f}")
        
        # Save model and results
        model_path = "models/phase4_ppo_champion.zip"
        agent.model.save(model_path)
        logger.info(f"  Model saved to: {model_path}")
        
        results = {
            "algorithm": "ppo",
            "phase": 4,
            "timesteps": timesteps,
            "hyperparameters": config,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "max_reward": float(np.max(eval_rewards)),
            "min_reward": float(np.min(eval_rewards)),
            "eval_episodes": eval_episodes,
        }
        
        results_file = "logs/phase4_ppo_champion_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"  Results saved to: {results_file}")
        
        env.close()
        return results
        
    except Exception as e:
        logger.error(f"PPO training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Champion Model Training")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["dqn", "ppo", "both"],
        default="both",
        help="Which algorithm to train (default: both)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Training timesteps per algorithm (default: 100000)",
    )
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    results_summary = {}
    
    if args.algorithm in ["dqn", "both"]:
        dqn_results = train_dqn_champion(timesteps=args.timesteps)
        results_summary["dqn"] = dqn_results
    
    if args.algorithm in ["ppo", "both"]:
        ppo_results = train_ppo_champion(timesteps=args.timesteps)
        results_summary["ppo"] = ppo_results
    
    # Print comparison
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: CHAMPION COMPARISON")
    logger.info("="*70)
    
    if "dqn" in results_summary and results_summary["dqn"]:
        logger.info(f"\nDQN Champion:")
        logger.info(f"  Mean Reward: {results_summary['dqn']['mean_reward']:.2f}")
        logger.info(f"  Std Reward:  {results_summary['dqn']['std_reward']:.2f}")
    
    if "ppo" in results_summary and results_summary["ppo"]:
        logger.info(f"\nPPO Champion:")
        logger.info(f"  Mean Reward: {results_summary['ppo']['mean_reward']:.2f}")
        logger.info(f"  Std Reward:  {results_summary['ppo']['std_reward']:.2f}")
    
    # Save summary
    summary_file = "logs/phase4_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
