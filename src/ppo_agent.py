"""
Proximal Policy Optimization (PPO) implementation with entropy regularization.

Based on Stable-Baselines3 PPO with custom logging of:
- Policy entropy H(pi) over time
- Mean advantage estimates
- GAE statistics
"""

import numpy as np
import torch
from typing import Optional, Dict, Any
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.ppo.policies import MlpPolicy
import gymnasium as gym

from src.logger import TrainingLogger, ExplorationTracker

# Suppress SB3 verbose logging
logging.getLogger("stable_baselines3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class PPOAgent:
    """PPO Agent with entropy regularization and advantage tracking."""

    def __init__(
        self,
        env: gym.Env,
        log_dir: str = "logs",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 0,
    ):
        self.env = env
        self.log_dir = log_dir
        self.logger = TrainingLogger(log_dir, "ppo")
        self.exploration_tracker = ExplorationTracker("PPO")

        self.model = PPO(
            MlpPolicy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=0,
            tensorboard_log=None,
            device="cpu",
            policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])),
        )

        self.hyperparams = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
        }

    def _compute_policy_entropy(self, obs: Optional[np.ndarray] = None) -> float:
        """Compute current policy entropy from the given observation.

        Args:
            obs: A flat observation vector.  If *None* the last observation
                 stored by the SB3 model is used (safe — no env reset).
        """
        try:
            if obs is None:
                # Use SB3's internally-stored last observation (VecEnv-shaped)
                obs = self.model._last_obs
                if obs is None:
                    return 0.0
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            obs_tensor = obs_tensor.to(self.model.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                entropy = float(dist.entropy().mean().item())
            return entropy
        except Exception:
            return 0.0

    def train(
        self,
        total_timesteps: int,
        num_episodes: Optional[int] = None,
        callback: Optional[BaseCallback] = None,
    ) -> Dict[str, Any]:
        """Train the PPO agent."""

        logger.info("=" * 70)
        logger.info("PPO Agent Training")
        logger.info("=" * 70)
        logger.info("Hyperparameters:")
        for key, value in self.hyperparams.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Training for {total_timesteps} timesteps...\n")

        agent_ref = self

        class MetricsCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.ep_reward = 0.0
                self.ep_length = 0
                self.ep_nodes = 0
                self.ep_count = 0
                self.entropy_log = []
                self.advantage_log = []

            def _on_step(self) -> bool:
                # Accumulate reward (SB3 uses plural "rewards")
                rewards = self.locals.get("rewards")
                if rewards is not None:
                    self.ep_reward += float(rewards[0])
                self.ep_length += 1

                # Node discovery
                infos = self.locals.get("infos", [{}])
                info = infos[0] if infos else {}
                if "discovered_node_count" in info:
                    self.ep_nodes = int(info["discovered_node_count"])

                # Episode boundary (SB3 uses plural "dones")
                dones = self.locals.get("dones")
                done = bool(dones[0]) if dones is not None else False

                if done:
                    self.ep_count += 1

                    entropy = agent_ref._compute_policy_entropy()
                    self.entropy_log.append((self.model.num_timesteps, entropy))

                    agent_ref.logger.log_episode(
                        episode=self.ep_count,
                        reward=self.ep_reward,
                        length=self.ep_length,
                        exploration_value=entropy,
                        exploration_type="entropy",
                    )
                    if entropy > 0:
                        agent_ref.exploration_tracker.log_entropy(
                            self.ep_count, entropy
                        )

                    if self.ep_count % 5 == 0:
                        logger.info(
                            f"Episode {self.ep_count:4d} | "
                            f"Nodes: {self.ep_nodes:2d}/6 | "
                            f"Reward: {self.ep_reward:8.1f} | "
                            f"Length: {self.ep_length:4d} | "
                            f"H(pi): {entropy:.4f} | "
                            f"Steps: {self.model.num_timesteps:7d}"
                        )

                    self.ep_reward = 0.0
                    self.ep_length = 0
                    self.ep_nodes = 0

                return True

            def _on_rollout_end(self) -> None:
                """Read advantage stats from rollout buffer after each PPO update."""
                try:
                    buf = self.model.rollout_buffer
                    if buf is not None and hasattr(buf, "advantages") and buf.advantages is not None:
                        mean_adv = float(buf.advantages.mean())
                        self.advantage_log.append((self.model.num_timesteps, mean_adv))
                except Exception:
                    pass

        metrics_cb = MetricsCallback()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=metrics_cb if callback is None else callback,
        )

        self.logger.save_metrics()

        stats = self.logger.get_stats()
        stats.update(self.exploration_tracker.summary())
        stats["entropy_log"] = metrics_cb.entropy_log
        stats["advantage_log"] = metrics_cb.advantage_log

        logger.info("\n" + "=" * 70)
        logger.info("Training Complete")
        logger.info("=" * 70)
        for k, v in stats.items():
            if isinstance(v, float):
                logger.info(f"  {k:25s}: {v:10.4f}")
            elif isinstance(v, (int, str)):
                logger.info(f"  {k:25s}: {v}")

        return stats

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the trained agent."""
        logger.info(f"\nEvaluating PPO for {num_episodes} episodes (deterministic={deterministic})...")

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
                logger.info(f"  Eval ep {ep + 1}: Reward={ep_reward:.1f}, "
                            f"Length={ep_length}, Nodes={nodes}/6")

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
        self.model = PPO.load(path, env=self.env)
        logger.info(f"Model loaded from {path}")
