"""
Gymnasium-compatible wrapper for CyberBattleSim environments.

Adapts the CyberBattleEnv (dict-based actions/observations) to Gymnasium API
for use with Stable-Baselines3 (DQN, PPO).

Supports both ToyCtf (10 nodes) and Chain (size+2 nodes) environments.

Observation encoding (256-dim vector):
    [0:6]    - Scalar fields (discovered_node_count, lateral_move, etc.)
    [6:106]  - Node discovery bitmap from discovered_nodes_properties
    [106:206]- Node privilege level (from nodes_privilegelevel)
    [206:216]- Action availability fractions per action type
    [216:218]- Credential cache density, leaked credentials count
    [218:256]- Reserved
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


class CyberBattleWrapper(gym.Env):
    """Gymnasium wrapper for CyberBattleEnv."""

    def __init__(self, cyberbattle_env, max_episode_steps=200, num_nodes=6):
        super().__init__()
        self.env = cyberbattle_env
        self.max_episode_steps = max_episode_steps
        self.num_nodes = num_nodes
        self.current_step = 0
        self.no_progress_counter = 0
        self.last_discovered_count = 0
        self.no_progress_threshold = 200  # Increased from 50 to 200 for better exploration
        self._prev_discovered = 0  # for reward shaping
        self.action_history_per_node = {}  # {node_id: {action_type: count}} - track actions per node in episode
        self.discovered_nodes = set()  # Track which nodes have been discovered in current episode
        self.episode_count = 0  # Track total episodes
        self.episode_cumulative_reward = 0.0  # Track cumulative reward per episode
        
        # ----- Intrinsic Motivation (Option A: Curiosity-Driven Exploration) -----
        self.credential_cache = set()  # Track discovered credentials for novelty bonus
        self.action_sequence_cache = []  # Track recent (action, node) pairs to detect loops
        self.state_visit_counts = {}  # Count visits to each state for exploration bonus
        self.trap_trigger_count = 0  # Track number of traps triggered in episode

        # Local RNG — never corrupt global np.random state
        self._rng = np.random.RandomState(42)

        self.metadata = {"render_modes": []}
        self.spec = None

        # Probe initial observation for space dimensions
        obs, _ = self.env.reset()
        
        # ----- Grouped Action Space (256 actions) -----
        # More than 36 to cover all valid action combinations from the environment
        # The _decode_action method enumerates actual valid actions and wraps the index
        self._action_types_list = ["local_vulnerability", "remote_vulnerability", "connect"]
        self._num_targets = 6
        self._max_actions = 256  # Reasonable upper bound for valid discrete actions
        self.action_space = spaces.Discrete(self._max_actions)

        # Observation space
        obs_size = 256
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        self.obs_size = obs_size

        self.last_action_mask = obs.get("action_mask", {})
        self._action_types = (
            sorted(self.last_action_mask.keys())
            if isinstance(self.last_action_mask, dict)
            else []
        )

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _flatten_observation(self, obs_dict):
        """Encode the dict observation into a 256-dim normalised vector."""
        obs = np.zeros(self.obs_size, dtype=np.float32)

        # --- Scalars [0:6] ---
        scalar_fields = [
            "discovered_node_count",
            "newly_discovered_nodes_count",
            "lateral_move",
            "customer_data_found",
            "escalation",
            "probe_result",
        ]
        for i, field in enumerate(scalar_fields):
            if field in obs_dict and i < 6:
                val = float(obs_dict[field])
                obs[i] = min(val / 10.0, 1.0) if "count" in field else min(val, 1.0)

        # --- Node discovery bitmap [6:106] ---
        # discovered_nodes_properties: shape (100, 10)
        mat = obs_dict.get("discovered_nodes_properties")
        if mat is not None:
            try:
                arr = np.asarray(mat, dtype=np.float32)
                if arr.ndim >= 2:
                    flags = (np.abs(arr).sum(axis=-1) > 0).astype(np.float32)
                else:
                    flags = (np.abs(arr) > 0).astype(np.float32)
                n = min(len(flags), 100)
                obs[6 : 6 + n] = flags[:n]
            except (ValueError, TypeError):
                pass

        # --- Node privilege level [106:206] ---
        # nodes_privilegelevel: shape (100,), integer privilege per node slot
        priv = obs_dict.get("nodes_privilegelevel")
        if priv is not None:
            try:
                arr = np.asarray(priv, dtype=np.float32)
                n = min(len(arr), 100)
                obs[106 : 106 + n] = np.clip(arr[:n] / 5.0, 0.0, 1.0)
            except (ValueError, TypeError):
                pass

        # --- Action availability fractions [206:216] ---
        action_mask = obs_dict.get("action_mask", {})
        if isinstance(action_mask, dict):
            for j, atype in enumerate(sorted(action_mask.keys())):
                if j >= 10:
                    break
                m = np.asarray(action_mask[atype])
                total = max(m.size, 1)
                obs[206 + j] = float(np.sum(m > 0)) / total

        # --- Credential cache density [216] ---
        cred = obs_dict.get("credential_cache_matrix")
        if cred is not None:
            try:
                arr = np.asarray(cred, dtype=np.float32)
                obs[216] = float(np.count_nonzero(arr)) / max(arr.size, 1)
            except (ValueError, TypeError):
                pass

        # --- Leaked credentials [217] ---
        leaked = obs_dict.get("leaked_credentials")
        if leaked is not None:
            try:
                arr = np.asarray(leaked, dtype=np.float32)
                obs[217] = min(float(np.count_nonzero(arr)) / 20.0, 1.0)
            except (ValueError, TypeError):
                pass

        return obs

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    def _decode_action(self, action_idx):
        """Convert discrete action index to valid environment action.
        
        Enumerates all valid actions from action_mask and selects based on index.
        """
        action_mask = self.last_action_mask
        if not action_mask or not isinstance(action_mask, dict):
            # Fallback to a safe default if no mask
            return {"local_vulnerability": (0, 0)}
        
        # Collect all valid (action_type, coords) pairs from the mask
        valid_actions = []
        for atype in ["local_vulnerability", "remote_vulnerability", "connect"]:
            mask = action_mask.get(atype)
            if mask is None:
                continue
            
            mask_array = np.asarray(mask)
            if mask_array.size == 0:
                continue
            
            # Find all valid coordinates for this action type
            valid_coords = np.argwhere(mask_array > 0)
            for coord in valid_coords:
                coords = tuple(int(x) for x in coord)
                valid_actions.append((atype, coords))
        
        # If no valid actions, return a safe fallback
        if not valid_actions:
            return {"local_vulnerability": (0, 0)}
        
        # Map action_idx to a valid action (with wrapping)
        action_idx = int(action_idx) % len(valid_actions)
        action_type, coords = valid_actions[action_idx]
        
        return {action_type: coords}

    def _get_valid_action(self, action_idx):
        """Convert integer action to a valid dict action."""
        action_mask = self.last_action_mask
        if not action_mask or not isinstance(action_mask, dict):
            return {"probe": 0}

        # Collect all valid (type, coords) pairs
        valid_actions = []
        for atype in self._action_types:
            mask = action_mask.get(atype)
            if mask is None:
                continue
            mask = np.asarray(mask)
            if mask.size == 0:
                continue
            coords = np.argwhere(mask > 0)
            for c in coords:
                valid_actions.append((atype, tuple(int(x) for x in c)))

        # Warn if valid actions exceed action space size
        if len(valid_actions) > self._max_actions:
            logger.warning(f"Valid actions ({len(valid_actions)}) exceed max_actions ({self._max_actions})")

        if not valid_actions:
            return {"probe": 0}

        selected_idx = int(action_idx) % len(valid_actions)
        atype, coords = valid_actions[selected_idx]
        return {atype: coords}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self.current_step = 0
        self.no_progress_counter = 0
        self.last_discovered_count = 0
        self._prev_discovered = 0
        self.action_history_per_node = {}  # {node_id: {action_type: count}}
        self.discovered_nodes = set()  # Track which nodes have been discovered in this episode
        self.episode_cumulative_reward = 0.0  # Reset cumulative reward for new episode
        
        # Clear intrinsic motivation trackers
        self.credential_cache.clear()
        self.action_sequence_cache.clear()
        self.state_visit_counts.clear()
        self.trap_trigger_count = 0

        obs_dict, info = self.env.reset()
        self.last_action_mask = obs_dict.get("action_mask", {})
        self._action_types = (
            sorted(self.last_action_mask.keys())
            if isinstance(self.last_action_mask, dict)
            else []
        )
        self.last_discovered_count = obs_dict.get("discovered_node_count", 0)
        self._prev_discovered = self.last_discovered_count
        obs = self._flatten_observation(obs_dict)
        return obs, info

    def step(self, action):
        self.current_step += 1
        
        # Decode grouped action to proper environment format
        dict_action = self._decode_action(action)
        obs_dict, reward, terminated, truncated, info = self.env.step(dict_action)

        self.last_action_mask = obs_dict.get("action_mask", {})
        obs = self._flatten_observation(obs_dict)

        current_discovered = obs_dict.get("discovered_node_count", 0)
        
        # ----- Intrinsic Motivation Rewards (Option A) -----
        intrinsic_reward = 0.0
        
        # Extract action type from dict_action for tracking
        action_type = None
        target_node = None
        if dict_action:
            for atype, coords in dict_action.items():
                action_type = atype
                # For local_vulnerability: coords = (node_id, vuln_id)
                # For remote_vulnerability: coords = (source, target, vuln_id)
                # For connect: coords = (source, target, service, port)
                if isinstance(coords, (tuple, list)) and len(coords) > 0:
                    target_node = int(coords[0])  # Use first element as primary node
                else:
                    target_node = int(coords) if coords else 0
                break
        
        # 1. Reward for discovering new credentials
        discovered_credentials = obs_dict.get("discovered_credentials", {})
        if discovered_credentials:
            cred_str = str(discovered_credentials)
            if cred_str not in self.credential_cache:
                intrinsic_reward += 10.0  # INCREASED to +10.0 for stronger signal to DQN
                self.credential_cache.add(cred_str)
                logger.info(f"NEW CREDENTIAL: Intrinsic Reward: +10.0")
        
        # 2. Penalize repeated action sequences (loop detection)
        if action_type is not None and target_node is not None:
            action_key = (action_type, target_node)
            self.action_sequence_cache.append(action_key)
            if len(self.action_sequence_cache) > 10:
                self.action_sequence_cache.pop(0)
            
            loop_count = self.action_sequence_cache.count(action_key)
            if loop_count >= 3:
                intrinsic_reward -= 0.3  # Reduced penalty for looping (was -2.0, too aggressive)
                logger.warning(f"LOOP DETECTED: {action_type} on node {target_node} | Penalty: -0.3")
        
        # 3. Reward state novelty (first time seeing this state)
        state_hash = hash(frozenset([(i, current_discovered) for i in range(current_discovered)]))
        self.state_visit_counts[state_hash] = self.state_visit_counts.get(state_hash, 0) + 1
        
        if self.state_visit_counts[state_hash] == 1:  # First time seeing state
            intrinsic_reward += 1.0
        elif self.state_visit_counts[state_hash] <= 3:  # Rarely seen states
            intrinsic_reward += 0.5
        
        # Add intrinsic rewards to base reward
        reward += intrinsic_reward

        # ----- Track and penalize repeated actions on already-owned nodes -----
        # (action_type and target_node already extracted above in intrinsic motivation block)
        # Track action attempts per node and penalize only on already-owned nodes
        if action_type is not None and target_node is not None:
            # Initialize per-node dict if needed
            if target_node not in self.action_history_per_node:
                self.action_history_per_node[target_node] = {}
            
            # Track this action type for this node
            if action_type not in self.action_history_per_node[target_node]:
                self.action_history_per_node[target_node][action_type] = 0
            self.action_history_per_node[target_node][action_type] += 1
            
            # Skip penalty on nodes still being explored (not yet discovered)
            # Only penalize if node is already owned
            if target_node in self.discovered_nodes:
                times_tried = self.action_history_per_node[target_node][action_type]
                if times_tried > 1:
                    penalty = 0.1  # Small penalty for repeating action on already-owned node
                    reward -= penalty
                    info["repeated_action_penalty"] = penalty

        # ----- Reward shaping (per modified report Section 3.2) -----
        # +20 per newly discovered node + ownership bonus
        new_nodes = current_discovered - self._prev_discovered
        if new_nodes > 0:
            # Track newly discovered nodes in our discovered set
            # Nodes 0 to current_discovered-1 are the ones we know about
            for node_id in range(current_discovered):
                self.discovered_nodes.add(node_id)
            reward += 30.0 * new_nodes  # INCREASED discovery reward from 20 to 30
            reward += 25.0 * new_nodes  # INCREASED ownership bonus from 15 to 25
        
        # ----- Trap Penalty (Enhanced) -----
        # Detect and penalize traps more leniently
        is_trap = False
        if "Trap" in str(info) or "trap" in str(obs_dict):
            is_trap = True
            self.trap_trigger_count += 1
            # Reduced penalty: -0.5 instead of harsh penalties
            reward -= 0.5
            logger.warning(f"Trap triggered | Trap count: {self.trap_trigger_count} | Penalty: -0.5")
        
        # -0.1 step penalty to discourage dawdling
        reward -= 0.1
        # -1 penalty for failed actions (no new nodes, no env reward) - only if NOT a trap
        if reward <= -0.1 + 1e-6 and new_nodes == 0 and not is_trap:
            reward -= 1.0
        self._prev_discovered = current_discovered

        # +100 completion bonus: all nodes discovered
        if current_discovered >= self.num_nodes:
            terminated = True
            reward += 100.0
            info["episode_success"] = True

        # Progress tracking — force reset when stuck
        if current_discovered > self.last_discovered_count:
            self.no_progress_counter = 0
            self.last_discovered_count = current_discovered
        else:
            self.no_progress_counter += 1
            if self.no_progress_counter >= self.no_progress_threshold:
                terminated = True
                info["episode_failure"] = True
                info["failure_reason"] = (
                    f"No progress for {self.no_progress_threshold} steps"
                )

        # No valid actions
        if isinstance(self.last_action_mask, dict):
            has_valid = any(
                np.any(np.asarray(m) > 0) for m in self.last_action_mask.values()
            )
            if not has_valid:
                terminated = True
                info["episode_failure"] = True
                info["failure_reason"] = "No valid actions"

        truncated = truncated or (self.current_step >= self.max_episode_steps)
        info["step"] = self.current_step
        info["discovered_node_count"] = current_discovered
        
        # Track cumulative episode reward
        self.episode_cumulative_reward += reward
        
        # Log episode summary when episode ends
        if terminated or truncated:
            self.episode_count += 1
            logger.info(
                f"Episode {self.episode_count:4d} | "
                f"Nodes: {current_discovered}/{self.num_nodes} | "
                f"Reward: {self.episode_cumulative_reward:10.1f} | "
                f"Steps: {self.current_step:4d}"
            )
        
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
