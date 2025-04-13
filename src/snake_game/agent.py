import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import pickle
import logging

logger = logging.getLogger(__name__)


class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, num_actions)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path):
        try:
            with open(path, 'rb') as f:
                buffer_data = pickle.load(f)
                self.buffer = deque(buffer_data, maxlen=self.buffer.maxlen)
        except FileNotFoundError:
            print(f"Replay buffer file {path} not found. Starting with an empty buffer.")


class DQNAgent:
    def __init__(self, state_shape, num_actions):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.q_network = DQN(input_dim=state_shape[0], num_actions=num_actions).to(self.device)
        self.target_network = DQN(input_dim=state_shape[0], num_actions=num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.scaler = GradScaler('cuda') if self.device.type == "cuda" else None
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 5000
        self.step_counter = 0
        self.batch_size = 256
        self.target_update_freq = 1000

    @property
    def epsilon(self):
        decay_factor = np.exp(-self.step_counter / self.epsilon_decay)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * decay_factor
        return max(self.epsilon_end, epsilon)

    def select_action(self, state):
        self.step_counter += 1
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                with autocast('cuda', enabled=self.device.type == "cuda"):
                    q_values = self.q_network(state)
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        self.optimizer.zero_grad()
        with autocast('cuda', enabled=self.device.type == "cuda"):
            q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + (1 - dones) * self.gamma * next_q_values
            loss = F.mse_loss(q_values, targets)
        if self.device.type == "cuda":
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
        if self.step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, model_path, buffer_path, step_counter_path):
        torch.save(self.q_network.state_dict(), model_path)
        self.replay_buffer.save(buffer_path)
        with open(step_counter_path, 'wb') as f:
            pickle.dump(self.step_counter, f)

    def load(self, model_path, buffer_path, step_counter_path):
        try:
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"Successfully loaded model from {model_path}")
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Starting with a fresh model.")
        self.replay_buffer.load(buffer_path)
        try:
            with open(step_counter_path, 'rb') as f:
                self.step_counter = pickle.load(f)
            print(f"Loaded step counter: {self.step_counter}")
        except FileNotFoundError:
            print(f"Step counter file {step_counter_path} not found. Starting with step_counter=0.")
            self.step_counter = 0
