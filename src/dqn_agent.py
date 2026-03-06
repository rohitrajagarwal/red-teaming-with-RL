"""
Deep Q-Network (DQN) implementation with polynomial epsilon decay.

Features:
- Polynomial epsilon decay schedule (smooth exploration-exploitation transition)
- Environment-based intrinsic rewards (credential discovery, state novelty, loop penalties)
  shared with PPO agent
- Q-value statistics logging
- Exploration tracking
"""

import numpy as np
import torch
from typing import Optional, Dict, Any
import logging
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.dqn.policies import MlpPolicy
import gymnasium as gym

from src.logger import TrainingLogger, ExplorationTracker

# Suppress SB3 verbose logging
logging.getLogger("stable_baselines3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class PolynomialDecay:
    """
    Polynomial epsilon decay schedule.
    ε(t) = ε_final + (ε_initial - ε_final) * ((1 - t/T)^power)
    
    This provides smoother exploration-exploitation transition compared to linear decay.
    """

    def __init__(
        self,
        initial_eps: float = 0.99,
        final_eps: float = 0.10,
        total_timesteps: int = 50000,
        power: float = 1.5,
    ):
        """
        Args:
            initial_eps: Starting exploration rate (high = aggressive exploration)
            final_eps: Ending exploration rate (maintains some exploration)
            total_timesteps: Total training timesteps
            power: Polynomial exponent
                  - power=1.0 → linear decay
                  - power=1.5 → smooth early decay, then plateau (recommended)
                  - power=2.0 → aggressive early decay
        """
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.total_timesteps = total_timesteps
        self.power = power

    def __call__(self, timestep: int) -> float:
        """
        Compute epsilon at given timestep.

        Args:
            timestep: current timestep (0 to total_timesteps)

        Returns:
            epsilon: exploration rate in [final_eps, initial_eps]
        """
        if timestep >= self.total_timesteps:
            return self.final_eps

        progress = timestep / self.total_timesteps
        decay_factor = (1.0 - progress) ** self.power
        epsilon = self.final_eps + (self.initial_eps - self.final_eps) * decay_factor

        return epsilon


class DQNAgent:
    """
    DQN Agent with polynomial epsilon decay.
    
    Intrinsic rewards (credential discovery, state novelty, loop penalties) are
    provided by the environment wrapper, same as PPO.
    """

    def __init__(
        self,
        env: gym.Env,
        log_dir: str = "logs",
        learning_rate: float = 1e-4,
        exploration_fraction: float = 0.5,
        exploration_initial_eps: float = 0.99,
        exploration_final_eps: float = 0.10,
        buffer_size: int = 50000,
        batch_size: int = 32,
        gamma: float = 0.99,
        train_freq: int = 4,
        polynomial_power: float = 1.5,
        verbose: int = 0,
    ):
        """
        Initialize DQN agent with polynomial decay.
        
        Args:
            env: Gymnasium environment
            log_dir: Directory for logging
            learning_rate: Q-network learning rate
            exploration_fraction: Fraction of training spent in exploration phase (for SB3)
            exploration_initial_eps: Initial epsilon (0.99 = 99% random)
            exploration_final_eps: Final epsilon (0.10 = 10% random always)
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            gamma: Discount factor
            train_freq: Update frequency (every N steps)
            polynomial_power: Power for polynomial decay (1.5 recommended)
            verbose: SB3 verbosity level
        """
        self.env = env
        self.log_dir = log_dir
        self.logger = TrainingLogger(log_dir, "dqn")
        self.exploration_tracker = ExplorationTracker("DQN")

        self.model = DQN(
            MlpPolicy,
            env,
            learning_rate=learning_rate,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=train_freq,
            verbose=0,
            tensorboard_log=None,
            device="cpu",
            policy_kwargs=dict(net_arch=[256, 256]),
        )

        # Polynomial decay schedule
        self.epsilon_scheduler = PolynomialDecay(
            initial_eps=exploration_initial_eps,
            final_eps=exploration_final_eps,
            total_timesteps=1000000,  # Will be set during training
            power=polynomial_power,
        )

        self.hyperparams = {
            "learning_rate": learning_rate,
            "exploration_fraction": exploration_fraction,
            "exploration_initial_eps": exploration_initial_eps,
            "exploration_final_eps": exploration_final_eps,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "train_freq": train_freq,
            "polynomial_power": polynomial_power,
        }

    def _compute_mean_q(self, n_samples: int = 32) -> float:
        """Estimate mean Q-value by sampling from the replay buffer."""
        try:
            buf = self.model.replay_buffer
            if buf.size() < n_samples:
                return 0.0
            replay_data = buf.sample(n_samples)
            obs_tensor = replay_data.observations
            with torch.no_grad():
                q_values = self.model.q_net(obs_tensor)
                return float(q_values.max(dim=1).values.mean().item())
        except Exception:
            return 0.0

    def train(
        self,
        total_timesteps: int,
        num_episodes: Optional[int] = None,
        callback: Optional[BaseCallback] = None,
    ) -> Dict[str, Any]:
        """
        Train the DQN agent with polynomial epsilon decay.
        
        Intrinsic rewards come from the environment wrapper (shared with PPO):
        - Credential discovery: +5.0
        - State novelty: +1.0 (first visit) or +0.5 (rare visit)
        - Loop detection: -0.3 penalty
        - Node discovery: +20.0 (discovery) + +15.0 (ownership)
        """

        logger.info("=" * 80)
        logger.info("DQN Agent Training (Polynomial Decay)")
        logger.info("=" * 80)
        logger.info("Hyperparameters:")
        for key, value in self.hyperparams.items():
            logger.info(f"  {key:30s}: {value}")
        logger.info(f"\nIntrinsic Rewards (from environment):")
        logger.info(f"  {'Credential Discovery':30s}: +5.0")
        logger.info(f"  {'State Novelty (first)':30s}: +1.0")
        logger.info(f"  {'State Novelty (rare)':30s}: +0.5")
        logger.info(f"  {'Loop Detection Penalty':30s}: -0.3")
        logger.info(f"\nTraining for {total_timesteps} timesteps...\n")

        # Update epsilon scheduler with actual training budget
        self.epsilon_scheduler.total_timesteps = total_timesteps

        agent_ref = self

        class MetricsCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.ep_reward = 0.0
                self.ep_length = 0
                self.ep_nodes = 0
                self.ep_count = 0
                self.q_values_log = []
                self.epsilon_log = []

            def _on_step(self) -> bool:
                # Accumulate extrinsic + intrinsic reward
                rewards = self.locals.get("rewards")
                if rewards is not None:
                    self.ep_reward += float(rewards[0])
                self.ep_length += 1

                # Node discovery
                infos = self.locals.get("infos", [{}])
                info = infos[0] if infos else {}
                if "discovered_node_count" in info:
                    self.ep_nodes = int(info["discovered_node_count"])

                # Episode boundary
                dones = self.locals.get("dones")
                done = bool(dones[0]) if dones is not None else False

                if done:
                    self.ep_count += 1

                    # Get epsilon from polynomial scheduler
                    epsilon = agent_ref.epsilon_scheduler(self.model.num_timesteps)

                    # Mean Q-value
                    mean_q = agent_ref._compute_mean_q()

                    self.q_values_log.append((self.model.num_timesteps, mean_q))
                    self.epsilon_log.append((self.model.num_timesteps, float(epsilon)))

                    agent_ref.logger.log_episode(
                        episode=self.ep_count,
                        reward=self.ep_reward,
                        length=self.ep_length,
                        exploration_value=float(epsilon),
                        exploration_type="epsilon_polynomial",
                        mean_q=mean_q,
                    )
                    agent_ref.exploration_tracker.log_epsilon(
                        self.ep_count, float(epsilon)
                    )

                    if self.ep_count % 5 == 0:
                        logger.info(
                            f"Episode {self.ep_count:4d} | "
                            f"Nodes: {self.ep_nodes:2d}/6 | "
                            f"Reward: {self.ep_reward:8.1f} | "
                            f"Length: {self.ep_length:4d} | "
                            f"Eps (poly): {epsilon:.4f} | "
                            f"Q: {mean_q:7.2f} | "
                            f"Steps: {self.model.num_timesteps:7d}"
                        )

                    self.ep_reward = 0.0
                    self.ep_length = 0
                    self.ep_nodes = 0

                return True

        metrics_cb = MetricsCallback()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=metrics_cb if callback is None else callback,
        )

        self.logger.save_metrics()

        stats = self.logger.get_stats()
        stats.update(self.exploration_tracker.summary())
        stats["q_values_log"] = metrics_cb.q_values_log
        stats["epsilon_log"] = metrics_cb.epsilon_log

        logger.info("\n" + "=" * 80)
        logger.info("Training Complete (Polynomial Decay)")
        logger.info("=" * 80)
        for k, v in stats.items():
            if isinstance(v, float):
                logger.info(f"  {k:30s}: {v:10.4f}")
            elif isinstance(v, (int, str, bool)):
                logger.info(f"  {k:30s}: {v}")

        return stats

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the trained agent."""
        logger.info(
            f"\nEvaluating DQN (Polynomial Decay) "
            f"for {num_episodes} episodes (deterministic={deterministic})..."
        )

        episode_rewards = []
        episode_lengths = []
        episode_nodes = []

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0
            nodes = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += reward
                ep_length += 1
                nodes = info.get("discovered_node_count", nodes)
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            episode_nodes.append(nodes)

            if (ep + 1) % 5 == 0:
                logger.info(
                    f"  Eval ep {ep + 1}: Reward={ep_reward:.1f}, "
                    f"Length={ep_length}, Nodes={nodes}/6"
                )

        stats = {
            "eval_mean_reward": float(np.mean(episode_rewards)),
            "eval_std_reward": float(np.std(episode_rewards)),
            "eval_max_reward": float(np.max(episode_rewards)),
            "eval_min_reward": float(np.min(episode_rewards)),
            "eval_mean_length": float(np.mean(episode_lengths)),
            "eval_mean_nodes": float(np.mean(episode_nodes)),
        }

        logger.info("Evaluation Results:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v:.4f}")

        return stats

    def save(self, path: str):
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        self.model = DQN.load(path, env=self.env)
        logger.info(f"Model loaded from {path}")
