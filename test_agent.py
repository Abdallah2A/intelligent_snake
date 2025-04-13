import pygame
import torch
import numpy as np
import logging
import cv2
import os
from src.snake_game.game import SnakeEnv
from src.snake_game.agent import DQN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def test_model(model_path, episodes=5):
    # Initialize the environment with rendering
    env = SnakeEnv(rendering=True)
    state = env.reset()
    input_size = len(state)  # Should be 11 based on SnakeEnv
    output_size = len(env.action_space)  # Should be 3 (0: straight, 1: left, 2: right)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(input_size, output_size).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return
    model.eval()  # Set to evaluation mode

    # Initialize variables to track the best game
    best_score = -1
    best_video_path = None

    # Run episodes
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        video_writer = None
        temp_video_path = None

        # Initialize video writer if rendering is enabled
        if env.rendering:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
            temp_video_path = "temp.mp4"
            # Resolution matches game window (cell_number * cell_size)
            video_writer = cv2.VideoWriter(temp_video_path, fourcc, 30.0,
                                           (env.cell_number * env.cell_size, env.cell_number * env.cell_size))

        logger.info(f"Starting episode {episode + 1}")
        while not done:
            # Handle Pygame events to keep the window responsive
            if env.rendering:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        logger.info("Pygame window closed by user.")
                        if 'video_writer' in locals():
                            video_writer.release()
                        env.close()
                        return

            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).to(device)

            # Get action from model
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            step += 1

            # Capture and write frame to video
            if env.rendering:
                screen = env.screen
                frame = pygame.surfarray.array3d(screen)  # Shape: (width, height, 3), RGB
                frame = np.transpose(frame, (1, 0, 2))   # Shape: (height, width, 3), RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
                video_writer.write(frame)

        # Release video writer after episode
        if env.rendering:
            video_writer.release()

        # Calculate score
        score = len(env.snake.body) - 5
        logger.info(f"Episode {episode + 1} finished. Steps: {step}, Score: {score}, Total Reward: {total_reward}")

        # Handle video saving if rendering
        if env.rendering:
            if score > best_score:
                best_score = score
                new_best_video_path = f"best_game_{score}.mp4"
                # Delete old best video if it exists
                if best_video_path and os.path.exists(best_video_path):
                    os.remove(best_video_path)
                # Rename temp video to final name with score
                os.rename(temp_video_path, new_best_video_path)
                best_video_path = new_best_video_path
                logger.info(f"New best score: {score}. Saved video to {best_video_path}")
            else:
                # Remove temp video if not the best
                os.remove(temp_video_path)

    env.close()


if __name__ == "__main__":
    # Path to your trained model
    path = "models/dqn_model.pth"
    test_model(path, episodes=100)
