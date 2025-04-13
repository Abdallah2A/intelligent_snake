
# Intelligent Snake

<p align="center">

  <img src="/assets/logo.png" alt="Snake RL Logo" width="180" align="right">

</p>

**Intelligent Snake** is a reinforcement learning (RL) project that leverages a classic snake game environment built with [Pygame](https://www.pygame.org/) and a Deep Q-Network (DQN) agent implemented with [PyTorch](https://pytorch.org/). The project integrates state-of-the-art RL techniques, replay buffers, and a ZenML pipeline to train an agent that learns to play the snake game effectively. It also includes functionality to visualize and save gameplay videos for evaluation.

Example of the model:
<p align="center">

  <img src="/assets/example.gif" alt="Snake RL Logo" width="300">

</p>

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Snake Game Environment](#snake-game-environment)
4. [Deep Q-Network (DQN) Agent](#deep-q-network-dqn-agent)
5. [Training Pipeline (ZenML)](#training-pipeline-zenml)
6. [Testing & Video Capture](#testing--video-capture)
7. [Setup & Usage](#setup--usage)
8. [Dependencies](#dependencies)
9. [License](#license)
10. [References](#references)

---

## 🚀 Overview

**Intelligent Snake** combines a traditional snake game with modern reinforcement learning techniques. The core components include:

- **Game Environment:**  
  A Pygame-based snake game that handles snake movements, fruit spawning, collision detection, and rendering.

- **DQN Agent:**  
  An RL agent that interacts with the game environment. The agent leverages a neural network, a replay buffer for experience storage, and utilizes techniques such as dropout and early target network updates to stabilize training.

- **ZenML Pipeline:**  
  The training phase is orchestrated with ZenML, enabling modular, reproducible, and scalable experiments.

- **Video Testing:**  
  The project includes functionality to capture gameplay, record videos of agent performance, and save the best performing episodes.

---

## 🗂 Project Structure

```bash
intelligent_snake/
├── assets/                   # Game-related images
├── models/                   # Trained models and replay buffers
├── pipelines/                # ZenML training pipeline
├── src/                      # Source code: game and agent logic
├── steps/                    # ZenML pipeline steps
├── example.mp4               # Example gameplay video
├── requirements.txt          # Python package dependencies
├── run_pipeline.py           # Script to run ZenML training pipeline
├── test_agent.py             # Script to test agent and record video
```

---

## 🎮 Snake Game Environment

- **SNAKE Class:**  
  Handles the snake’s body, direction, movement logic, and image rendering (head, tail, and body segments).

- **FRUIT Class:**  
  Spawns fruit at random locations and handles rendering.

- **SnakeEnv Class:**  
  Wraps the game logic into a Gym-like environment, provides state representation, handles game resets, and processes agent actions.

---

## 🧠 Deep Q-Network (DQN) Agent

- **Neural Network:**  
  Fully connected layers with dropout and LeakyReLU to predict Q-values for 3 possible actions.

- **Replay Buffer:**  
  Stores past experiences and samples mini-batches for training.

- **Epsilon-Greedy Strategy:**  
  Balances exploration and exploitation during training.

- **Model Saving & Loading:**  
  Saves model weights and replay buffer state to continue training later.

---

## 🔁 Training Pipeline (ZenML)

- **ZenML Integration:**  
  Modular training setup with a ZenML pipeline defined in `pipelines/pipeline.py` and steps in `steps/`.

- **Execution:**

```bash
python run_pipeline.py
```

This trains the agent using the environment and stores checkpoints.

---

## 🎥 Testing & Video Capture

- **Script:** `test_agent.py`
- **Functionality:**
  - Loads the trained agent.
  - Plays multiple episodes.
  - Captures the best-performing episode as an `.mp4` file using OpenCV.

Run with:

```bash
python test_agent.py
```

---

## ⚙️ Setup & Usage

1. **Clone the Repository:**

```bash
git clone https://github.com/Abdallah2A/intelligent_snake.git
cd intelligent_snake
```

2. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

3. **Train the Agent:**

```bash
python run_pipeline.py
```

4. **Test and Record Agent:**

```bash
python test_agent.py
```

---

## 📦 Dependencies

- Python 3.12+
- Pygame
- PyTorch
- OpenCV
- ZenML
- NumPy

Install them all with:

```bash
pip install -r requirements.txt
```

---

## 🪪 License

This project is open-source and licensed under the MIT License.

---

## 📚 References

- [Pygame](https://www.pygame.org/)
- [PyTorch](https://pytorch.org/)
- [ZenML](https://docs.zenml.io/)
- [OpenAI Gym](https://www.gymlibrary.dev/)
- [OpenCV](https://opencv.org/)

---

🎉 **Enjoy building and training your own Snake RL agent!**
