import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        # Deeper network with larger hidden layers
        self.fc1 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 9)
        # Using LeakyReLU for better gradient flow
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.flatten(x)
        
        x = self.fc1(x)
        if batch_size > 1:  # Only apply batch norm if batch size > 1
            x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        if batch_size > 1:
            x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.fc4(x)
        return x

class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # how much prioritization to use (0 = uniform, 1 = full prioritization)
        self.beta = beta    # importance-sampling correction (0 = no correction, 1 = full correction)
        self.beta_increment = beta_increment  # annealing the importance sampling weight
        self.epsilon = epsilon  # small constant to ensure non-zero priority
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        max_priority = self.max_priority if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # Increase beta for importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
            
        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
            
        # Sample batch according to probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
            
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
            
        states, actions, rewards, next_states, dones = zip(*samples)
            
        return (
            states, actions, rewards, next_states, dones,
            indices, weights
        )
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            priority = priority + self.epsilon  # Add small constant to prevent zero priority
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.997, 
                 gamma=0.99, lr=0.0005, batch_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # Use a learning rate scheduler for better convergence
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500, verbose=True
        )
        # Use Huber loss which is less sensitive to outliers than MSE
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        
        # Better exploration strategy with slower decay
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Double DQN parameters
        self.use_double_dqn = True
        
        # Use prioritized experience replay for better learning
        self.memory = PrioritizedReplayBuffer(capacity=50000)
        self.update_target_counter = 0
        
        # Track metrics for monitoring learning
        self.loss_history = []
        
    def board_to_state(self, board):
        return torch.FloatTensor(board.flatten()).unsqueeze(0).to(self.device)
        
    def action_to_position(self, action):
        return (action // 3, action % 3)
        
    def position_to_action(self, position):
        i, j = position
        return i * 3 + j
        
    def choose_action(self, state, valid_moves):
        if not valid_moves:
            return None
            
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
            
        state_tensor = self.board_to_state(state)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]
            
        # Filter for valid moves only
        valid_actions = [self.position_to_action(move) for move in valid_moves]
        best_action = valid_actions[0]
        best_value = q_values[best_action]
        
        for action in valid_actions[1:]:
            if q_values[action] > best_value:
                best_value = q_values[action]
                best_action = action
                
        return self.action_to_position(best_action)
        
    def remember(self, state, action, reward, next_state, done):
        action_idx = self.position_to_action(action)
        self.memory.add(state, action_idx, reward, next_state, done)
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        # Get batch with importance sampling weights from prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([s.flatten() for s in states])).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array([s.flatten() for s in next_states])).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Get current Q values
        curr_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use online network to select actions and target network to evaluate
                next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
                next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_model(next_states).max(1)[0]
            
            # Calculate target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Calculate TD errors for prioritized replay
        td_errors = torch.abs(curr_q_values - target_q_values).detach().cpu().numpy()
        
        # Calculate weighted loss for importance sampling
        elementwise_loss = self.loss_fn(curr_q_values, target_q_values)
        loss = (elementwise_loss * weights).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Track loss
        self.loss_history.append(loss.item())
        
        # Update priorities in the replay buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Update learning rate based on loss
        self.scheduler.step(loss)
        
        # Update epsilon (exploration rate) with a slower decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Periodically update target network (more frequent updates)
        self.update_target_counter += 1
        if self.update_target_counter >= 5:  # Update more frequently
            self.target_model.load_state_dict(self.model.state_dict())
            self.update_target_counter = 0
            
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(self.model.state_dict())