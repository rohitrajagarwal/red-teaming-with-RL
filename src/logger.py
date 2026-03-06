"""
Training metrics logger for exploration-exploitation analysis.

Tracks and logs:
- Episode rewards and lengths
- Exploration metrics (epsilon decay for DQN, entropy for PPO)
- Mean Q-values and advantage estimates
- Training speed and convergence statistics
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


class TrainingLogger:
    """Logs training metrics for RL algorithms."""
    
    def __init__(self, log_dir: str = "logs", agent_name: str = "agent"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            agent_name: Name of agent (used for log filenames)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent_name = agent_name
        self.episode = 0
        
        # Storage for metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.exploration_metrics: List[Dict[str, float]] = []
        self.training_metrics: List[Dict[str, Any]] = []
        
        # Log file paths
        self.metrics_file = self.log_dir / f"{agent_name}_metrics.json"
        self.csv_file = self.log_dir / f"{agent_name}_training.csv"
        self.checkpoint_dir = self.log_dir / f"{agent_name}_checkpoints"
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV header
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_file, 'w') as f:
            headers = [
                'episode', 'reward', 'length', 'avg_reward_100', 
                'exploration_metric', 'loss', 'timestamp'
            ]
            f.write(','.join(headers) + '\n')
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        exploration_value: Optional[float] = None,
        loss: Optional[float] = None,
        **kwargs
    ):
        """
        Log episode completion.
        
        Args:
            episode: Episode number
            reward: Total episodic reward
            length: Episode length
            exploration_value: Exploration metric (epsilon for DQN, entropy for PPO)
            loss: Loss/td-error from training
            **kwargs: Additional metrics to log
        """
        self.episode = episode
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Append row to CSV
        import time as _time
        avg_reward_100 = float(np.mean(self.episode_rewards[-100:]))
        with open(self.csv_file, 'a') as f:
            f.write(','.join([
                str(episode),
                f"{reward:.4f}",
                str(length),
                f"{avg_reward_100:.4f}",
                f"{exploration_value:.6f}" if exploration_value is not None else "",
                f"{loss:.6f}" if loss is not None else "",
                str(int(_time.time())),
            ]) + '\n')
        
        # Store exploration metric
        exploration_dict = {"episode": episode}
        if exploration_value is not None:
            exploration_dict["value"] = exploration_value
            exploration_dict["type"] = kwargs.get("exploration_type", "unknown")
        self.exploration_metrics.append(exploration_dict)
        
        # Store training metrics
        training_dict = {
            "episode": episode,
            "reward": reward,
            "length": length,
            **kwargs
        }
        if loss is not None:
            training_dict["loss"] = loss
        self.training_metrics.append(training_dict)
        
        # Print progress every 10 episodes
        if episode % 10 == 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            exploration_str = f"{exploration_value:.4f}" if exploration_value is not None else "N/A"
            print(f"[Episode {episode:4d}] Reward: {reward:7.2f} | "
                  f"Avg(100): {avg_reward:7.2f} | "
                  f"Length: {length:4d} | "
                  f"Exploration: {exploration_str}")
    
    def save_metrics(self):
        """Save all metrics to JSON."""
        metrics = {
            "agent_name": self.agent_name,
            "total_episodes": self.episode,
            "total_steps": sum(self.episode_lengths),
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "exploration_metrics": self.exploration_metrics,
            "training_metrics": self.training_metrics,
            "statistics": {
                "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
                "std_reward": float(np.std(self.episode_rewards)) if self.episode_rewards else 0,
                "max_reward": float(np.max(self.episode_rewards)) if self.episode_rewards else 0,
                "min_reward": float(np.min(self.episode_rewards)) if self.episode_rewards else 0,
                "mean_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0,
            }
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Metrics saved to {self.metrics_file}")
    
    def save_checkpoint(self, model, step: int):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.zip"
        model.save(str(checkpoint_path))
        return checkpoint_path
    
    def get_stats(self) -> Dict[str, float]:
        """Get current training statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "max_reward": float(np.max(self.episode_rewards)),
            "min_reward": float(np.min(self.episode_rewards)),
            "episodes": len(self.episode_rewards),
            "total_steps": int(sum(self.episode_lengths)),
        }


class ExplorationTracker:
    """Tracks exploration metrics during training."""
    
    def __init__(self, algorithm: str):
        """
        Initialize tracker.
        
        Args:
            algorithm: Algorithm name ('DQN' or 'PPO')
        """
        self.algorithm = algorithm
        self.history: List[Dict[str, float]] = []
    
    def log_epsilon(self, episode: int, epsilon: float):
        """Log epsilon value (DQN)."""
        self.history.append({
            "episode": episode,
            "epsilon": epsilon,
            "algorithm": "DQN"
        })
    
    def log_entropy(self, episode: int, entropy: float, entropy_loss: Optional[float] = None):
        """Log entropy value (PPO)."""
        entry = {
            "episode": episode,
            "entropy": entropy,
            "algorithm": "PPO"
        }
        if entropy_loss is not None:
            entry["entropy_loss"] = entropy_loss
        self.history.append(entry)
    
    def get_history(self) -> List[Dict[str, float]]:
        """Get exploration history."""
        return self.history
    
    def summary(self) -> Dict[str, float]:
        """Get summary of exploration behavior."""
        if not self.history:
            return {}
        
        if self.algorithm == "DQN":
            epsilons = [h.get("epsilon", 0) for h in self.history]
            return {
                "initial_epsilon": float(epsilons[0]) if epsilons else 0,
                "final_epsilon": float(epsilons[-1]) if epsilons else 0,
                "mean_epsilon": float(np.mean(epsilons)) if epsilons else 0,
            }
        else:  # PPO
            entropies = [h.get("entropy", 0) for h in self.history]
            return {
                "initial_entropy": float(entropies[0]) if entropies else 0,
                "final_entropy": float(entropies[-1]) if entropies else 0,
                "mean_entropy": float(np.mean(entropies)) if entropies else 0,
            }
