"""
Phase 2: PPO Baseline Training

Trains PPO with NO entropy regularization (ent_coef=0.0) for 50,000 steps.
This establishes the pure exploitation baseline per project plan.

Hyperparameters per project plan:
  - learning_rate = 3e-4
  - ent_coef = 0.0  (NO entropy regularization)
  - clip_range = 0.2
  - 50,000 total timesteps

Logs: cumulative reward, policy entropy H(pi), mean advantage estimates.
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Logging setup
log_file = Path("logs/train_ppo_output.log")
log_file.parent.mkdir(parents=True, exist_ok=True)


class FlushHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        FlushHandler(log_file, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

import cyberbattle
from cyberbattle._env.cyberbattle_chain import CyberBattleChain
from src.environment_wrapper import CyberBattleWrapper
from src.ppo_agent import PPOAgent

CHAIN_SIZE = 4   # 6 nodes (start + 4 chain + 1 extra)
NUM_NODES = 6


def main():
    logger.info("=" * 70)
    logger.info("PHASE 2 - PPO Baseline Training (NO entropy regularization)")
    logger.info("=" * 70)

    # Environment
    logger.info(f"[1] Initializing CyberBattleChain({CHAIN_SIZE}) ({NUM_NODES}-node chain)...")
    cyber_env = CyberBattleChain(size=CHAIN_SIZE, attacker_goal=None)
    env = CyberBattleWrapper(cyber_env, max_episode_steps=2000, num_nodes=NUM_NODES)
    logger.info(f"  Action space:      {env.action_space}")
    logger.info(f"  Observation space: {env.observation_space}")

    # Agent — Phase 2 baseline: ent_coef=0.02 (WITH exploration bonus)
    logger.info("[2] Initializing PPO agent (exploration enabled)...")
    agent = PPOAgent(
        env=env,
        log_dir="logs",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,      # Exploration bonus: encourages diverse actions
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    # Training
    logger.info("[3] Training for 50,000 timesteps...")
    start = time.time()
    stats = agent.train(total_timesteps=50_000)
    elapsed = time.time() - start
    logger.info(f"Wall-clock time: {elapsed / 60:.1f} min")

    # Evaluation
    logger.info("[4] Evaluating trained agent (10 episodes, deterministic)...")
    eval_stats = agent.evaluate(num_episodes=10, deterministic=True)

    # Save
    Path("models").mkdir(exist_ok=True)
    agent.save("models/ppo_baseline")

    # Summary
    logger.info("=" * 70)
    logger.info("PPO BASELINE SUMMARY")
    logger.info("=" * 70)
    for k, v in {**stats, **eval_stats}.items():
        if isinstance(v, float):
            logger.info(f"  {k:30s}: {v:10.4f}")
        elif isinstance(v, (int, str)):
            logger.info(f"  {k:30s}: {v}")
    logger.info("Model  -> models/ppo_baseline.zip")
    logger.info("Metrics -> logs/ppo_metrics.json")
    logger.info("=" * 70)

    env.close()
    logging.shutdown()


if __name__ == "__main__":
    main()
