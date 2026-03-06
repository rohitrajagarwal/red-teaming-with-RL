"""
Phase 2: DQN Baseline Training

Trains DQN with default epsilon-greedy exploration for 50,000 steps.
Hyperparameters per project plan:
  - learning_rate = 1e-4
  - exploration_fraction = 1.0 (decay over entire training run)
  - epsilon: 0.95 -> 0.05
  - 50,000 total timesteps

Logs: cumulative reward, epsilon decay curve, mean Q-values.
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Logging setup
log_file = Path("logs/train_dqn_output.log")
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
from src.dqn_agent import DQNAgent

CHAIN_SIZE = 4   # 6 nodes (start + 4 chain + 1 extra)
NUM_NODES = 6


def main():
    logger.info("=" * 70)
    logger.info("PHASE 2 - DQN Baseline Training (epsilon-greedy)")
    logger.info("=" * 70)

    # Environment
    logger.info(f"[1] Initializing CyberBattleChain({CHAIN_SIZE}) ({NUM_NODES}-node chain)...")
    cyber_env = CyberBattleChain(size=CHAIN_SIZE, attacker_goal=None)
    env = CyberBattleWrapper(cyber_env, max_episode_steps=2000, num_nodes=NUM_NODES)
    logger.info(f"  Action space:      {env.action_space}")
    logger.info(f"  Observation space: {env.observation_space}")

    # Agent with improved exploration for sparse-reward network discovery
    logger.info("[2] Initializing DQN agent (improved exploration)...")
    agent = DQNAgent(
        env=env,
        log_dir="logs",
        learning_rate=1e-4,
        exploration_fraction=0.7,      # INCREASED: explore for 70% of training
        exploration_initial_eps=0.99,  # INCREASED: start with 99% random exploration
        exploration_final_eps=0.05,    # LOWERED: maintain 5% exploration
        buffer_size=5_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
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
    agent.save("models/dqn_baseline")

    # Summary
    logger.info("=" * 70)
    logger.info("DQN BASELINE SUMMARY")
    logger.info("=" * 70)
    for k, v in {**stats, **eval_stats}.items():
        if isinstance(v, float):
            logger.info(f"  {k:30s}: {v:10.4f}")
        elif isinstance(v, (int, str)):
            logger.info(f"  {k:30s}: {v}")
    logger.info("Model  -> models/dqn_baseline.zip")
    logger.info("Metrics -> logs/dqn_metrics.json")
    logger.info("=" * 70)

    env.close()
    logging.shutdown()


if __name__ == "__main__":
    main()
