import logging
import numpy as np
import torch
from zenml import step
from src.snake_game.game import SnakeEnv
from src.snake_game.agent import DQNAgent

logger = logging.getLogger(__name__)


@step
def training_step(
        cell_number: int = 25,
        cell_size: int = 30,
        rendering: bool = False,
        num_episodes: int = 10_000,
        model_path: str = "models/dqn_model.pth",
        buffer_path: str = "models/replay_buffer.pkl",
        step_counter_path: str = "models/step_counter.pkl",
        resume: bool = True
) -> None:
    try:
        env = SnakeEnv(cell_number=cell_number, cell_size=cell_size, rendering=rendering)
        state = env.reset()
        logger.info(f"Initial state shape from SnakeEnv: {state.shape}")
        state_shape = (11,)
        num_actions = len(env.action_space)
        logger.info(f"Initializing DQNAgent with state_shape={state_shape}, num_actions={num_actions}")
        agent = DQNAgent(state_shape=state_shape, num_actions=num_actions)
        if resume:
            logger.info(f"Resuming training. Attempting to load model from {model_path}")
            agent.load(model_path, buffer_path, step_counter_path)
        else:
            logger.info("Starting training from scratch. Ignoring any existing saved files.")
            agent.epsilon_start = 1.0
        if len(agent.replay_buffer) < agent.batch_size:
            logger.info(f"Replay buffer has {len(agent.replay_buffer)} experiences. Starting warm-up phase.")
            warm_up_steps = agent.batch_size - len(agent.replay_buffer)
            state = env.reset()
            for _ in range(warm_up_steps):
                action = agent.select_action(state)
                next_state, base_reward, done, _ = env.step(action)
                reward = 0
                if base_reward > 0:
                    reward += 10.0
                reward += 0.01
                if state[0] > 0.5:  # Adjusted for float danger level
                    reward -= 0.5
                if state[1] > 0.5 or state[2] > 0.5:
                    reward -= 0.3
                agent.store_experience(state, action, reward, next_state, done)
                state = next_state
                if done:
                    state = env.reset()
            logger.info(f"Warm-up phase complete. Replay buffer now has {len(agent.replay_buffer)} experiences.")
        if agent.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(agent.device) / 1024 ** 2
            memory_reserved = torch.cuda.memory_reserved(agent.device) / 1024 ** 2
            logger.info(f"GPU Memory Allocated: {memory_allocated:.2f} MB, Reserved: {memory_reserved:.2f} MB")
        total_rewards = []
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            step_count = 0
            while True:
                action = agent.select_action(state)
                next_state, base_reward, done, _ = env.step(action)
                reward = 0
                if base_reward > 0:
                    reward += 10.0
                reward += 0.01
                if state[0] > 0.5:
                    reward -= 0.5
                if state[1] > 0.5 or state[2] > 0.5:
                    reward -= 0.3
                agent.store_experience(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                episode_reward += reward
                step_count += 1
                if done:
                    break
            total_rewards.append(episode_reward)
            avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
            logger.info(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Reward: {episode_reward:.2f}, "
                f"Avg Reward (last 100): {avg_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.4f}"
            )
            if (episode + 1) % 100 == 0:
                agent.save(model_path, buffer_path, step_counter_path)
                logger.info(f"Model saved at episode {episode + 1} to {model_path}")
        agent.save(model_path, buffer_path, step_counter_path)
        logger.info(f"Final model saved to {model_path}")
    except Exception as e:
        logger.error("Training failed: %s", str(e))
        raise e
