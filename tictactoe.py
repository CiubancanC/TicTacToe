#!/usr/bin/env python3
"""
TicTacToe AI - Complete Implementation
=====================================

A streamlined TicTacToe game with AI training, playing, and analysis capabilities.
Combines all functionality into a single, well-organized file.

Features:
- Deep Q-Network (DQN) AI with balanced training (both X and O positions)
- Perfect Minimax AI that never loses
- Interactive gameplay modes
- AI training and evaluation tools
- Model comparison and analysis

Usage:
    python3 tictactoe.py            # Interactive menu
    python3 tictactoe.py train      # Train new AI
    python3 tictactoe.py play       # Play against AI
    python3 tictactoe.py battle     # AI vs AI battles
    python3 tictactoe.py analyze    # Analyze AI performance
"""

import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import argparse

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Game imports
import pygame


# ==========================================
# GAME LOGIC
# ==========================================

class TicTacToe:
    """Core TicTacToe game logic with strategic reward shaping"""
    
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        self.game_over = False
        self.winner = None
        
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        return self.get_state()
        
    def get_state(self):
        return self.board.copy()
        
    def get_valid_moves(self):
        if self.game_over:
            return []
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
        
    def make_move(self, position, use_advanced_rewards=True):
        if self.game_over:
            return self.get_state(), 0, True
            
        i, j = position
        if self.board[i, j] != 0:
            return self.get_state(), -10, False  # Invalid move penalty
            
        old_board = self.board.copy()
        self.board[i, j] = self.current_player
        
        # Check for win or draw
        reward = 0
        done = False
        
        # Check for win (rows, columns, diagonals)
        win_conditions = [
            # Rows
            [self.board[0, :], self.board[1, :], self.board[2, :]],
            # Columns  
            [self.board[:, 0], self.board[:, 1], self.board[:, 2]],
            # Diagonals
            [np.array([self.board[0,0], self.board[1,1], self.board[2,2]]),
             np.array([self.board[0,2], self.board[1,1], self.board[2,0]])]
        ]
        
        for condition_set in win_conditions:
            for condition in condition_set:
                if np.sum(condition) == 3 * self.current_player:
                    self.game_over = True
                    self.winner = self.current_player
                    reward = 10 if self.current_player == 1 else -10
                    done = True
                    break
            if done:
                break
                
        # Advanced reward shaping for strategic moves
        if use_advanced_rewards and not done:
            reward += self._calculate_strategic_reward(old_board, position, self.current_player)
                
        # Check for draw
        if not self.game_over and len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0
            reward = 1  # Draw reward
            done = True
            
        # Switch player
        if not self.game_over:
            self.current_player *= -1
            
        return self.get_state(), reward, done
    
    def _calculate_strategic_reward(self, old_board, position, player):
        """Calculate strategic reward bonuses"""
        reward = 0
        i, j = position
        
        # Center control bonus
        if i == 1 and j == 1:
            reward += 0.5
        
        # Corner control bonus
        if (i, j) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            reward += 0.3
            
        # Fork creation bonus (multiple ways to win)
        fork_opportunities = self._count_winning_lines(self.board, player)
        if fork_opportunities >= 2:
            reward += 2.0
        elif fork_opportunities == 1:
            reward += 0.5
            
        # Blocking opponent's win
        if self._blocks_immediate_win(old_board, position, -player):
            reward += 1.5
            
        # Creating win threat
        if self._count_winning_lines(self.board, player) > 0:
            reward += 1.0
            
        return reward
    
    def _count_winning_lines(self, board, player):
        """Count number of lines where player can win in one move"""
        count = 0
        
        # Check rows
        for r in range(3):
            if list(board[r, :]).count(player) == 2 and list(board[r, :]).count(0) == 1:
                count += 1
                
        # Check columns  
        for c in range(3):
            if list(board[:, c]).count(player) == 2 and list(board[:, c]).count(0) == 1:
                count += 1
                
        # Check diagonals
        diag1 = [board[i, i] for i in range(3)]
        if diag1.count(player) == 2 and diag1.count(0) == 1:
            count += 1
            
        diag2 = [board[i, 2-i] for i in range(3)]
        if diag2.count(player) == 2 and diag2.count(0) == 1:
            count += 1
            
        return count
    
    def _blocks_immediate_win(self, old_board, position, opponent):
        """Check if move blocks opponent's immediate win"""
        test_board = old_board.copy()
        test_board[position[0], position[1]] = opponent
        
        # Check if opponent would win
        i, j = position
        
        # Check row, column, diagonals
        conditions = [
            np.sum(test_board[i, :]) == 3 * opponent,
            np.sum(test_board[:, j]) == 3 * opponent,
            (i == j) and np.sum([test_board[k, k] for k in range(3)]) == 3 * opponent,
            (i + j == 2) and np.sum([test_board[k, 2-k] for k in range(3)]) == 3 * opponent
        ]
        
        return any(conditions)
        
    def get_result(self):
        return self.winner


# ==========================================
# MINIMAX AI
# ==========================================

class MinimaxAgent:
    """Perfect Minimax AI that never loses"""
    
    def __init__(self):
        pass
    
    def choose_action(self, state, valid_moves):
        if not valid_moves:
            return None
        
        # Use minimax to find best move
        best_score = float('-inf')
        best_move = valid_moves[0]
        
        for move in valid_moves:
            # Simulate move
            test_board = state.copy()
            test_board[move[0], move[1]] = -1  # Minimax always plays as O
            
            score = self._minimax(test_board, 0, False, float('-inf'), float('inf'))
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move
    
    def _minimax(self, board, depth, is_maximizing, alpha, beta):
        """Minimax algorithm with alpha-beta pruning"""
        result = self._evaluate_board(board)
        
        if result is not None:
            return result
        
        if is_maximizing:
            max_eval = float('-inf')
            for i in range(3):
                for j in range(3):
                    if board[i, j] == 0:
                        board[i, j] = -1  # Minimax player
                        eval_score = self._minimax(board, depth + 1, False, alpha, beta)
                        board[i, j] = 0
                        max_eval = max(max_eval, eval_score)
                        alpha = max(alpha, eval_score)
                        if beta <= alpha:
                            break
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(3):
                for j in range(3):
                    if board[i, j] == 0:
                        board[i, j] = 1  # Opponent
                        eval_score = self._minimax(board, depth + 1, True, alpha, beta)
                        board[i, j] = 0
                        min_eval = min(min_eval, eval_score)
                        beta = min(beta, eval_score)
                        if beta <= alpha:
                            break
                if beta <= alpha:
                    break
            return min_eval
    
    def _evaluate_board(self, board):
        """Evaluate board position"""
        # Check for win conditions
        for player in [1, -1]:
            # Rows
            for r in range(3):
                if np.sum(board[r, :]) == 3 * player:
                    return 10 if player == -1 else -10
            
            # Columns
            for c in range(3):
                if np.sum(board[:, c]) == 3 * player:
                    return 10 if player == -1 else -10
            
            # Diagonals
            if np.sum([board[i, i] for i in range(3)]) == 3 * player:
                return 10 if player == -1 else -10
            if np.sum([board[i, 2-i] for i in range(3)]) == 3 * player:
                return 10 if player == -1 else -10
        
        # Check for draw
        if len([(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]) == 0:
            return 0
        
        return None  # Game not over


# ==========================================
# DEEP Q-NETWORK AI
# ==========================================

class DQN(nn.Module):
    """Deep Q-Network for tic-tac-toe"""
    
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 9)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, 9)
        
        x = self.fc1(x)
        if batch_size > 1:
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


class BalancedDQNAgent:
    """DQN Agent that can play as both X and O using perspective flipping"""
    
    def __init__(self, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.997, 
                 gamma=0.99, lr=0.0005, batch_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500, verbose=True
        )
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.memory = deque(maxlen=50000)
        self.update_target_counter = 0
        self.loss_history = []
        
        self.target_model.load_state_dict(self.model.state_dict())
        
    def choose_action(self, state, valid_moves, as_player_x=True):
        if not valid_moves:
            return None
            
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        
        # Perspective flipping for O player
        if as_player_x:
            input_state = state
        else:
            input_state = -state  # Flip perspective for O player
            
        state_tensor = torch.FloatTensor(input_state.flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]
            
        # Find best valid move
        valid_actions = [i * 3 + j for i, j in valid_moves]
        best_idx = np.argmax([q_values[action] for action in valid_actions])
        best_action = valid_actions[best_idx]
        return (best_action // 3, best_action % 3)
        
    def remember(self, state, action, reward, next_state, done):
        action_idx = action[0] * 3 + action[1]
        self.memory.append((state, action_idx, reward, next_state, done))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array([s.flatten() for s in states])).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array([s.flatten() for s in next_states])).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
        loss = self.loss_fn(current_q_values, target_q_values).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        self.scheduler.step(loss)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.update_target_counter += 1
        if self.update_target_counter >= 100:
            self.target_model.load_state_dict(self.model.state_dict())
            self.update_target_counter = 0
            
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())


# ==========================================
# TRAINING SYSTEM
# ==========================================

def train_balanced_ai(num_episodes=10000, save_path="tictactoe_ai.pt"):
    """Train a balanced AI that can play as both X and O"""
    print("🎯 Training Balanced TicTacToe AI")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print("Features: Balanced training (X and O), Strategic rewards, Self-play")
    print("=" * 50)
    
    agent = BalancedDQNAgent()
    
    # Training metrics
    x_wins = x_draws = x_losses = 0
    o_wins = o_draws = o_losses = 0
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        game = TicTacToe()
        state = game.reset()
        done = False
        
        # Alternate agent position (X or O)
        agent_is_x = (episode % 2 == 0)
        episode_experiences = []
        
        while not done:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            current_state = state.copy()
            agent_turn = (game.current_player == 1 and agent_is_x) or (game.current_player == -1 and not agent_is_x)
            
            if agent_turn:
                # Agent's turn
                action = agent.choose_action(current_state, valid_moves, as_player_x=agent_is_x)
                if action is None:
                    break
                
                next_state, reward, done = game.make_move(action, use_advanced_rewards=True)
                
                # Store experience with proper perspective
                if agent_is_x:
                    stored_state = current_state
                    stored_next_state = next_state.copy()
                else:
                    # Flip perspective for O player
                    stored_state = -current_state
                    stored_next_state = -next_state.copy() if not done else next_state.copy()
                
                episode_experiences.append((stored_state, action, reward, stored_next_state, done))
                
            else:
                # Opponent's turn (self-play)
                action = agent.choose_action(current_state, valid_moves, as_player_x=not agent_is_x)
                if action is None:
                    break
                
                next_state, reward, done = game.make_move(action, use_advanced_rewards=True)
                
                # Store opponent experience
                if agent_is_x:
                    stored_state = -current_state
                    stored_next_state = -next_state.copy() if not done else next_state.copy()
                    opp_reward = -reward if reward in [10, -10] else reward
                else:
                    stored_state = current_state
                    stored_next_state = next_state.copy()
                    opp_reward = -reward if reward in [10, -10] else reward
                
                episode_experiences.append((stored_state, action, opp_reward, stored_next_state, done))
            
            state = next_state
        
        # Add all experiences to memory
        for exp in episode_experiences:
            agent.remember(*exp)
        
        # Train agent
        agent.replay()
        
        # Record results
        result = game.get_result()
        if agent_is_x:
            if result == 1:
                x_wins += 1
            elif result == -1:
                x_losses += 1
            else:
                x_draws += 1
        else:
            if result == -1:
                o_wins += 1
            elif result == 1:
                o_losses += 1
            else:
                o_draws += 1
        
        # Progress reporting
        if (episode + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            total_games = episode + 1
            x_games = (total_games + 1) // 2
            o_games = total_games - x_games
            
            x_wr = (x_wins / x_games * 100) if x_games > 0 else 0
            x_dr = (x_draws / x_games * 100) if x_games > 0 else 0
            o_wr = (o_wins / o_games * 100) if o_games > 0 else 0
            o_dr = (o_draws / o_games * 100) if o_games > 0 else 0
            
            print(f"Episode {episode + 1}: ε:{agent.epsilon:.4f} ({elapsed/60:.1f}min)")
            print(f"  As X: W:{x_wr:.1f}% D:{x_dr:.1f}% ({x_games} games)")
            print(f"  As O: W:{o_wr:.1f}% D:{o_dr:.1f}% ({o_games} games)")
    
    # Save model
    agent.save(save_path)
    
    training_time = (time.time() - start_time) / 60
    print(f"\n🎉 Training completed in {training_time:.1f} minutes")
    print(f"💾 Model saved: {save_path}")
    
    return agent


# ==========================================
# ANALYSIS & TESTING
# ==========================================

def analyze_ai_performance(model_path, num_games=100):
    """Analyze AI performance against Minimax from both positions"""
    print("🔬 AI Performance Analysis")
    print("=" * 40)
    
    try:
        agent = BalancedDQNAgent(epsilon=0)
        agent.load(model_path)
        print(f"✅ Loaded model: {model_path}")
    except:
        print(f"❌ Could not load model: {model_path}")
        return
    
    minimax = MinimaxAgent()
    
    # Test as X (first player)
    print(f"\n📍 Testing as X ({num_games//2} games)...")
    x_wins = x_draws = x_losses = 0
    
    for _ in range(num_games // 2):
        game = TicTacToe()
        while not game.game_over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            if game.current_player == 1:  # AI as X
                action = agent.choose_action(game.get_state(), valid_moves, as_player_x=True)
            else:  # Minimax as O
                action = minimax.choose_action(game.get_state(), valid_moves)
            
            if action:
                game.make_move(action)
        
        result = game.get_result()
        if result == 1:
            x_wins += 1
        elif result == -1:
            x_losses += 1
        else:
            x_draws += 1
    
    # Test as O (second player)
    print(f"📍 Testing as O ({num_games//2} games)...")
    o_wins = o_draws = o_losses = 0
    
    for _ in range(num_games // 2):
        game = TicTacToe()
        while not game.game_over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            if game.current_player == 1:  # Minimax as X
                action = minimax.choose_action(game.get_state(), valid_moves)
            else:  # AI as O
                action = agent.choose_action(game.get_state(), valid_moves, as_player_x=False)
            
            if action:
                game.make_move(action)
        
        result = game.get_result()
        if result == -1:
            o_wins += 1
        elif result == 1:
            o_losses += 1
        else:
            o_draws += 1
    
    # Results
    x_non_loss = (x_wins + x_draws) / (num_games // 2) * 100
    o_non_loss = (o_wins + o_draws) / (num_games // 2) * 100
    overall_non_loss = (x_wins + x_draws + o_wins + o_draws) / num_games * 100
    
    print(f"\n📊 RESULTS:")
    print(f"   As X: W:{x_wins} D:{x_draws} L:{x_losses} ({x_non_loss:.1f}% non-loss)")
    print(f"   As O: W:{o_wins} D:{o_draws} L:{o_losses} ({o_non_loss:.1f}% non-loss)")
    print(f"   Overall: {overall_non_loss:.1f}% non-loss rate")
    
    # Performance assessment
    if overall_non_loss >= 90:
        print("🏆 EXCELLENT: Near-perfect play!")
    elif overall_non_loss >= 70:
        print("✅ GOOD: Strong strategic play!")
    elif overall_non_loss >= 50:
        print("⚠️  FAIR: Decent performance!")
    else:
        print("❌ NEEDS IMPROVEMENT: Significant gaps remain!")
    
    return {
        'x_performance': x_non_loss,
        'o_performance': o_non_loss,
        'overall_performance': overall_non_loss
    }


# ==========================================
# INTERACTIVE GAMEPLAY
# ==========================================

def play_vs_ai(model_path="tictactoe_ai.pt"):
    """Play against the AI with a simple text interface"""
    print("🎮 Play vs AI")
    print("=" * 30)
    
    try:
        agent = BalancedDQNAgent(epsilon=0)
        agent.load(model_path)
        print(f"✅ AI loaded from {model_path}")
    except:
        print("❌ Could not load AI model. Train one first!")
        return
    
    while True:
        game = TicTacToe()
        
        # Choose who goes first
        human_first = input("Do you want to go first? (y/n): ").lower() == 'y'
        human_is_x = human_first
        
        print(f"\nYou are {'X' if human_is_x else 'O'}")
        print(f"AI is {'O' if human_is_x else 'X'}")
        print()
        
        while not game.game_over:
            print_board(game.board)
            print()
            
            human_turn = (game.current_player == 1 and human_is_x) or (game.current_player == -1 and not human_is_x)
            
            if human_turn:
                # Human's turn
                try:
                    move_input = input("Your move (row,col): ")
                    row, col = map(int, move_input.split(','))
                    if (row, col) in game.get_valid_moves():
                        game.make_move((row, col))
                    else:
                        print("Invalid move! Try again.")
                        continue
                except:
                    print("Invalid input! Use format: row,col (e.g., 1,1)")
                    continue
            else:
                # AI's turn
                print("AI thinking...")
                time.sleep(0.5)
                valid_moves = game.get_valid_moves()
                action = agent.choose_action(game.get_state(), valid_moves, as_player_x=not human_is_x)
                if action:
                    game.make_move(action)
                    print(f"AI plays: {action[0]},{action[1]}")
        
        # Game over
        print_board(game.board)
        result = game.get_result()
        
        if result == 0:
            print("🤝 It's a draw!")
        elif (result == 1 and human_is_x) or (result == -1 and not human_is_x):
            print("🎉 You win!")
        else:
            print("🤖 AI wins!")
        
        if input("\nPlay again? (y/n): ").lower() != 'y':
            break


def print_board(board):
    """Print the game board"""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print("  0 1 2")
    for i in range(3):
        row = f"{i} "
        for j in range(3):
            row += symbols[board[i, j]] + " "
        print(row)


def ai_battle(model1_path="tictactoe_ai.pt", model2_path=None, num_games=100):
    """AI vs AI battle"""
    print("⚔️  AI Battle Arena")
    print("=" * 30)
    
    # Load agents
    try:
        agent1 = BalancedDQNAgent(epsilon=0)
        agent1.load(model1_path)
        print(f"✅ Agent 1 loaded: {model1_path}")
    except:
        print(f"❌ Could not load Agent 1: {model1_path}")
        return
    
    if model2_path:
        try:
            agent2 = BalancedDQNAgent(epsilon=0)
            agent2.load(model2_path)
            print(f"✅ Agent 2 loaded: {model2_path}")
        except:
            print(f"❌ Could not load Agent 2: {model2_path}")
            return
    else:
        # Use Minimax as opponent
        agent2 = MinimaxAgent()
        print("✅ Agent 2: Perfect Minimax AI")
    
    # Battle
    agent1_wins = agent1_draws = agent1_losses = 0
    
    print(f"\n🎮 Running {num_games} games...")
    
    for game_num in range(num_games):
        game = TicTacToe()
        
        # Alternate who goes first
        agent1_is_x = (game_num % 2 == 0)
        
        while not game.game_over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            agent1_turn = (game.current_player == 1 and agent1_is_x) or (game.current_player == -1 and not agent1_is_x)
            
            if agent1_turn:
                action = agent1.choose_action(game.get_state(), valid_moves, as_player_x=agent1_is_x)
            else:
                if hasattr(agent2, 'choose_action') and len(agent2.__class__.__name__) > 10:  # DQN agent
                    action = agent2.choose_action(game.get_state(), valid_moves, as_player_x=not agent1_is_x)
                else:  # Minimax agent
                    action = agent2.choose_action(game.get_state(), valid_moves)
            
            if action:
                game.make_move(action)
        
        # Record results from Agent 1's perspective
        result = game.get_result()
        if result == 0:
            agent1_draws += 1
        elif (result == 1 and agent1_is_x) or (result == -1 and not agent1_is_x):
            agent1_wins += 1
        else:
            agent1_losses += 1
        
        if (game_num + 1) % 25 == 0:
            print(f"Games {game_num + 1}: W:{agent1_wins} D:{agent1_draws} L:{agent1_losses}")
    
    # Final results
    agent1_non_loss = (agent1_wins + agent1_draws) / num_games * 100
    
    print(f"\n🏆 BATTLE RESULTS:")
    print(f"Agent 1: W:{agent1_wins} D:{agent1_draws} L:{agent1_losses} ({agent1_non_loss:.1f}% non-loss)")
    print(f"Agent 2: W:{agent1_losses} D:{agent1_draws} L:{agent1_wins}")
    
    if agent1_wins > agent1_losses:
        print("🎉 Agent 1 WINS!")
    elif agent1_losses > agent1_wins:
        print("🎉 Agent 2 WINS!")
    else:
        print("🤝 TIE!")


# ==========================================
# MAIN INTERFACE
# ==========================================

def main():
    """Main interactive interface"""
    parser = argparse.ArgumentParser(description="TicTacToe AI - Complete Implementation")
    parser.add_argument('command', nargs='?', choices=['train', 'play', 'battle', 'analyze'], 
                       help='Command to run')
    parser.add_argument('--episodes', type=int, default=10000, help='Training episodes')
    parser.add_argument('--model', default='tictactoe_ai.pt', help='Model file path')
    parser.add_argument('--games', type=int, default=100, help='Number of test games')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_balanced_ai(args.episodes, args.model)
    elif args.command == 'play':
        play_vs_ai(args.model)
    elif args.command == 'battle':
        ai_battle(args.model, num_games=args.games)
    elif args.command == 'analyze':
        analyze_ai_performance(args.model, args.games)
    else:
        # Interactive menu
        print("🎮 TicTacToe AI - Complete Implementation")
        print("=" * 50)
        print("1. Train new AI")
        print("2. Play vs AI")
        print("3. AI vs AI Battle")
        print("4. Analyze AI Performance")
        print("5. Exit")
        
        while True:
            try:
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == '1':
                    episodes = int(input("Training episodes (default 10000): ") or "10000")
                    train_balanced_ai(episodes)
                elif choice == '2':
                    play_vs_ai()
                elif choice == '3':
                    games = int(input("Number of battle games (default 100): ") or "100")
                    ai_battle(num_games=games)
                elif choice == '4':
                    games = int(input("Number of test games (default 100): ") or "100")
                    analyze_ai_performance("tictactoe_ai.pt", games)
                elif choice == '5':
                    print("👋 Goodbye!")
                    break
                else:
                    print("Invalid choice! Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()