`#!/usr/bin/env python3
"""
TicTacToe AI - Improved Complete Implementation
==============================================

A robust TicTacToe game with AI training, playing, and analysis capabilities.
Fixes major issues with perspective handling and training balance.

Features:
- Position-aware Deep Q-Network (DQN) AI with proper X/O handling
- Perfect Minimax AI that can play as both X and O
- Interactive gameplay modes
- AI training and evaluation tools
- Model comparison and analysis
- Improved training diagnostics

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
            return self.get_state(), -1, False  # Invalid move penalty
            
        old_board = self.board.copy()
        self.board[i, j] = self.current_player
        
        # Check for win or draw
        reward = 0
        done = False
        
        # Check for win
        if self._check_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player
            reward = 1  # Win reward (always positive for the player who won)
            done = True
        # Check for draw
        elif len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0
            reward = 0  # Draw reward
            done = True
        else:
            # Advanced reward shaping for strategic moves
            if use_advanced_rewards:
                reward = self._calculate_strategic_reward(old_board, position, self.current_player)
        
        # Switch player
        if not self.game_over:
            self.current_player *= -1
            
        return self.get_state(), reward, done
    
    def _check_win(self, player):
        """Check if the given player has won"""
        # Check rows
        for r in range(3):
            if np.all(self.board[r, :] == player):
                return True
        
        # Check columns
        for c in range(3):
            if np.all(self.board[:, c] == player):
                return True
        
        # Check diagonals
        if np.all([self.board[i, i] == player for i in range(3)]):
            return True
        if np.all([self.board[i, 2-i] == player for i in range(3)]):
            return True
        
        return False
    
    def _calculate_strategic_reward(self, old_board, position, player):
        """Calculate strategic reward bonuses"""
        reward = 0
        i, j = position
        
        # Center control bonus
        if i == 1 and j == 1:
            reward += 0.1
        
        # Corner control bonus
        if (i, j) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            reward += 0.05
            
        # Fork creation bonus (multiple ways to win)
        fork_opportunities = self._count_winning_lines(self.board, player)
        if fork_opportunities >= 2:
            reward += 0.3
        elif fork_opportunities == 1:
            reward += 0.1
            
        # Blocking opponent's win
        if self._blocks_immediate_win(old_board, position, -player):
            reward += 0.2
            
        # Creating immediate win threat
        if self._creates_immediate_win(position, player):
            reward += 0.2
            
        return reward
    
    def _count_winning_lines(self, board, player):
        """Count number of lines where player can win in one move"""
        count = 0
        
        # Check rows
        for r in range(3):
            row = list(board[r, :])
            if row.count(player) == 2 and row.count(0) == 1:
                count += 1
                
        # Check columns  
        for c in range(3):
            col = list(board[:, c])
            if col.count(player) == 2 and col.count(0) == 1:
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
        # Temporarily place opponent piece to test
        test_board = old_board.copy()
        test_board[position[0], position[1]] = opponent
        return self._check_win_on_board(test_board, opponent)
    
    def _creates_immediate_win(self, position, player):
        """Check if move creates immediate win for player"""
        return self._check_win(player)
    
    def _check_win_on_board(self, board, player):
        """Check if player has won on given board state"""
        # Check rows
        for r in range(3):
            if np.all(board[r, :] == player):
                return True
        
        # Check columns
        for c in range(3):
            if np.all(board[:, c] == player):
                return True
        
        # Check diagonals
        if np.all([board[i, i] == player for i in range(3)]):
            return True
        if np.all([board[i, 2-i] == player for i in range(3)]):
            return True
        
        return False
        
    def get_result(self):
        return self.winner


# ==========================================
# MINIMAX AI
# ==========================================

class MinimaxAgent:
    """Perfect Minimax AI that can play as both X and O"""
    
    def __init__(self, player=None):
        self.player = player  # 1 for X, -1 for O, None for adaptive
    
    def choose_action(self, state, valid_moves, as_player=None):
        if not valid_moves:
            return None
        
        # Determine which player we're playing as
        if as_player is not None:
            player = as_player
        elif self.player is not None:
            player = self.player
        else:
            # Auto-detect based on current game state
            occupied_squares = np.count_nonzero(state)
            player = 1 if occupied_squares % 2 == 0 else -1
        
        # Use minimax to find best move
        best_score = float('-inf')
        best_move = valid_moves[0]
        
        for move in valid_moves:
            # Simulate move
            test_board = state.copy()
            test_board[move[0], move[1]] = player
            
            score = self._minimax(test_board, 0, False, float('-inf'), float('inf'), player)
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move
    
    def _minimax(self, board, depth, is_maximizing, alpha, beta, max_player):
        """Minimax algorithm with alpha-beta pruning"""
        result = self._evaluate_board(board, max_player)
        
        if result is not None:
            return result
        
        if is_maximizing:
            max_eval = float('-inf')
            for i in range(3):
                for j in range(3):
                    if board[i, j] == 0:
                        board[i, j] = max_player
                        eval_score = self._minimax(board, depth + 1, False, alpha, beta, max_player)
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
            min_player = -max_player
            for i in range(3):
                for j in range(3):
                    if board[i, j] == 0:
                        board[i, j] = min_player
                        eval_score = self._minimax(board, depth + 1, True, alpha, beta, max_player)
                        board[i, j] = 0
                        min_eval = min(min_eval, eval_score)
                        beta = min(beta, eval_score)
                        if beta <= alpha:
                            break
                if beta <= alpha:
                    break
            return min_eval
    
    def _evaluate_board(self, board, max_player):
        """Evaluate board position"""
        # Check for win conditions
        for player in [1, -1]:
            if self._check_win_on_board(board, player):
                if player == max_player:
                    return 10
                else:
                    return -10
        
        # Check for draw
        if len([(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]) == 0:
            return 0
        
        return None  # Game not over
    
    def _check_win_on_board(self, board, player):
        """Check if player has won on given board state"""
        # Check rows
        for r in range(3):
            if np.all(board[r, :] == player):
                return True
        
        # Check columns
        for c in range(3):
            if np.all(board[:, c] == player):
                return True
        
        # Check diagonals
        if np.all([board[i, i] == player for i in range(3)]):
            return True
        if np.all([board[i, 2-i] == player for i in range(3)]):
            return True
        
        return False


# ==========================================
# DEEP Q-NETWORK AI
# ==========================================

class PositionAwareDQN(nn.Module):
    """Position-aware Deep Q-Network for tic-tac-toe"""
    
    def __init__(self):
        super(PositionAwareDQN, self).__init__()
        # Input: 9 (board) + 1 (player position) = 10
        self.fc1 = nn.Linear(10, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ImprovedDQNAgent:
    """Improved DQN Agent with position-aware input"""
    
    def __init__(self, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, 
                 gamma=0.95, lr=0.001, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PositionAwareDQN().to(self.device)
        self.target_model = PositionAwareDQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.memory = deque(maxlen=50000)
        self.update_target_counter = 0
        self.loss_history = []
        
        self.target_model.load_state_dict(self.model.state_dict())
        
    def _prepare_input(self, state, player):
        """Prepare position-aware input for the network"""
        # Flatten board and add player indicator
        board_flat = state.flatten()
        player_indicator = np.array([player])  # 1 for X, -1 for O
        return np.concatenate([board_flat, player_indicator])
        
    def choose_action(self, state, valid_moves, player):
        if not valid_moves:
            return None
            
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        
        # Prepare input
        input_features = self._prepare_input(state, player)
        state_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]
            
        # Find best valid move
        valid_actions = [i * 3 + j for i, j in valid_moves]
        best_idx = np.argmax([q_values[action] for action in valid_actions])
        best_action = valid_actions[best_idx]
        return (best_action // 3, best_action % 3)
        
    def remember(self, state, action, reward, next_state, done, player):
        # Store experience with player context
        input_features = self._prepare_input(state, player)
        next_input_features = self._prepare_input(next_state, player)
        action_idx = action[0] * 3 + action[1]
        self.memory.append((input_features, action_idx, reward, next_input_features, done))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.update_target_counter += 1
        if self.update_target_counter >= 100:
            self.target_model.load_state_dict(self.model.state_dict())
            self.update_target_counter = 0
            
    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'loss_history': self.loss_history
        }, filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(self.model.state_dict())
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']


# ==========================================
# TRAINING SYSTEM
# ==========================================

class TrainingMonitor:
    """Monitor training progress and performance"""
    
    def __init__(self):
        self.x_wins = self.x_draws = self.x_losses = 0
        self.o_wins = self.o_draws = self.o_losses = 0
        self.minimax_test_results = []
        
    def record_result(self, result, agent_is_x):
        if agent_is_x:
            if result == 1:
                self.x_wins += 1
            elif result == -1:
                self.x_losses += 1
            else:
                self.x_draws += 1
        else:
            if result == -1:
                self.o_wins += 1
            elif result == 1:
                self.o_losses += 1
            else:
                self.o_draws += 1
    
    def get_stats(self, episode):
        total_games = episode + 1
        x_games = (total_games + 1) // 2
        o_games = total_games - x_games
        
        x_wr = (self.x_wins / x_games * 100) if x_games > 0 else 0
        x_dr = (self.x_draws / x_games * 100) if x_games > 0 else 0
        x_lr = (self.x_losses / x_games * 100) if x_games > 0 else 0
        
        o_wr = (self.o_wins / o_games * 100) if o_games > 0 else 0
        o_dr = (self.o_draws / o_games * 100) if o_games > 0 else 0
        o_lr = (self.o_losses / o_games * 100) if o_games > 0 else 0
        
        return {
            'x_stats': (x_wr, x_dr, x_lr, x_games),
            'o_stats': (o_wr, o_dr, o_lr, o_games)
        }


def train_improved_ai(num_episodes=15000, save_path="improved_tictactoe_ai.pt", test_interval=2000):
    """Train an improved AI with better monitoring"""
    print("🎯 Training Improved TicTacToe AI")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print("Features: Position-aware network, Balanced training, Regular validation")
    print("=" * 50)
    
    agent = ImprovedDQNAgent()
    monitor = TrainingMonitor()
    minimax = MinimaxAgent()
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        game = TicTacToe()
        state = game.reset()
        done = False
        
        # Alternate agent position (X or O)
        agent_is_x = (episode % 2 == 0)
        agent_player = 1 if agent_is_x else -1
        
        while not done:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            current_state = state.copy()
            agent_turn = (game.current_player == 1 and agent_is_x) or (game.current_player == -1 and not agent_is_x)
            
            if agent_turn:
                # Agent's turn
                action = agent.choose_action(current_state, valid_moves, agent_player)
                if action is None:
                    break
                
                next_state, reward, done = game.make_move(action, use_advanced_rewards=True)
                
                # Adjust reward based on agent's perspective
                if done and game.winner != 0:  # Someone won
                    if game.winner == agent_player:
                        reward = 1.0  # Agent won
                    else:
                        reward = -1.0  # Agent lost
                
                agent.remember(current_state, action, reward, next_state, done, agent_player)
                
            else:
                # Self-play opponent turn
                opp_player = -agent_player
                action = agent.choose_action(current_state, valid_moves, opp_player)
                if action is None:
                    break
                
                next_state, reward, done = game.make_move(action, use_advanced_rewards=True)
                
                # Adjust reward for opponent perspective
                if done and game.winner != 0:  # Someone won
                    if game.winner == opp_player:
                        reward = 1.0  # Opponent won
                    else:
                        reward = -1.0  # Opponent lost
                
                agent.remember(current_state, action, reward, next_state, done, opp_player)
            
            state = next_state
        
        # Train agent
        agent.replay()
        
        # Record results
        result = game.get_result()
        monitor.record_result(result, agent_is_x)
        
        # Progress reporting and validation
        if (episode + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            stats = monitor.get_stats(episode)
            
            print(f"Episode {episode + 1}: ε:{agent.epsilon:.4f} ({elapsed/60:.1f}min)")
            print(f"  As X: W:{stats['x_stats'][0]:.1f}% D:{stats['x_stats'][1]:.1f}% L:{stats['x_stats'][2]:.1f}% ({stats['x_stats'][3]} games)")
            print(f"  As O: W:{stats['o_stats'][0]:.1f}% D:{stats['o_stats'][1]:.1f}% L:{stats['o_stats'][2]:.1f}% ({stats['o_stats'][3]} games)")
        
        # Validation against Minimax
        if (episode + 1) % test_interval == 0:
            print(f"\n🔍 Validation against Minimax...")
            test_results = quick_minimax_test(agent, minimax, num_games=20)
            monitor.minimax_test_results.append((episode + 1, test_results))
            print(f"   vs Minimax: {test_results['overall_non_loss']:.1f}% non-loss rate")
    
    # Save model
    agent.save(save_path)
    
    training_time = (time.time() - start_time) / 60
    print(f"\n🎉 Training completed in {training_time:.1f} minutes")
    print(f"💾 Model saved: {save_path}")
    
    return agent


def quick_minimax_test(agent, minimax, num_games=20):
    """Quick test against Minimax during training"""
    old_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during testing
    
    x_wins = x_draws = x_losses = 0
    o_wins = o_draws = o_losses = 0
    
    for i in range(num_games):
        game = TicTacToe()
        agent_is_x = (i % 2 == 0)
        
        while not game.game_over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            if (game.current_player == 1 and agent_is_x) or (game.current_player == -1 and not agent_is_x):
                # Agent's turn
                player = 1 if agent_is_x else -1
                action = agent.choose_action(game.get_state(), valid_moves, player)
            else:
                # Minimax's turn
                minimax_player = 1 if not agent_is_x else -1
                action = minimax.choose_action(game.get_state(), valid_moves, minimax_player)
            
            if action:
                game.make_move(action)
        
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
    
    agent.epsilon = old_epsilon  # Restore epsilon
    
    x_non_loss = (x_wins + x_draws) / (num_games // 2) * 100 if num_games > 0 else 0
    o_non_loss = (o_wins + o_draws) / (num_games // 2) * 100 if num_games > 0 else 0
    overall_non_loss = (x_wins + x_draws + o_wins + o_draws) / num_games * 100
    
    return {
        'x_non_loss': x_non_loss,
        'o_non_loss': o_non_loss,
        'overall_non_loss': overall_non_loss
    }


# ==========================================
# ANALYSIS & TESTING
# ==========================================

def analyze_ai_performance(model_path, num_games=100):
    """Analyze AI performance against Minimax from both positions"""
    print("🔬 AI Performance Analysis")
    print("=" * 40)
    
    try:
        agent = ImprovedDQNAgent(epsilon=0)
        agent.load(model_path)
        print(f"✅ Loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Could not load model: {model_path} - {e}")
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
                action = agent.choose_action(game.get_state(), valid_moves, 1)
            else:  # Minimax as O
                action = minimax.choose_action(game.get_state(), valid_moves, -1)
            
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
                action = minimax.choose_action(game.get_state(), valid_moves, 1)
            else:  # AI as O
                action = agent.choose_action(game.get_state(), valid_moves, -1)
            
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

def play_vs_ai(model_path="improved_tictactoe_ai.pt"):
    """Play against the AI with a simple text interface"""
    print("🎮 Play vs AI")
    print("=" * 30)
    
    try:
        agent = ImprovedDQNAgent(epsilon=0)
        agent.load(model_path)
        print(f"✅ AI loaded from {model_path}")
    except Exception as e:
        print(f"❌ Could not load AI model: {e}")
        print("Train one first using the train command!")
        return
    
    while True:
        game = TicTacToe()
        
        # Choose who goes first
        human_first = input("Do you want to go first? (y/n): ").lower() == 'y'
        human_is_x = human_first
        human_player = 1 if human_is_x else -1
        ai_player = -human_player
        
        print(f"\nYou are {'X' if human_is_x else 'O'}")
        print(f"AI is {'O' if human_is_x else 'X'}")
        print()
        
        while not game.game_over:
            print_board(game.board)
            print()
            
            human_turn = (game.current_player == human_player)
            
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
                action = agent.choose_action(game.get_state(), valid_moves, ai_player)
                if action:
                    game.make_move(action)
                    print(f"AI plays: {action[0]},{action[1]}")
        
        # Game over
        print_board(game.board)
        result = game.get_result()
        
        if result == 0:
            print("🤝 It's a draw!")
        elif result == human_player:
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


def ai_battle(model1_path="improved_tictactoe_ai.pt", model2_path=None, num_games=100):
    """AI vs AI battle"""
    print("⚔️  AI Battle Arena")
    print("=" * 30)
    
    # Load agents
    try:
        agent1 = ImprovedDQNAgent(epsilon=0)
        agent1.load(model1_path)
        print(f"✅ Agent 1 loaded: {model1_path}")
    except Exception as e:
        print(f"❌ Could not load Agent 1: {model1_path} - {e}")
        return
    
    if model2_path:
        try:
            agent2 = ImprovedDQNAgent(epsilon=0)
            agent2.load(model2_path)
            print(f"✅ Agent 2 loaded: {model2_path}")
        except Exception as e:
            print(f"❌ Could not load Agent 2: {model2_path} - {e}")
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
        agent1_player = 1 if agent1_is_x else -1
        agent2_player = -agent1_player
        
        while not game.game_over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            agent1_turn = (game.current_player == agent1_player)
            
            if agent1_turn:
                action = agent1.choose_action(game.get_state(), valid_moves, agent1_player)
            else:
                if hasattr(agent2, 'choose_action') and hasattr(agent2, 'model'):  # DQN agent
                    action = agent2.choose_action(game.get_state(), valid_moves, agent2_player)
                else:  # Minimax agent
                    action = agent2.choose_action(game.get_state(), valid_moves, agent2_player)
            
            if action:
                game.make_move(action)
        
        # Record results from Agent 1's perspective
        result = game.get_result()
        if result == 0:
            agent1_draws += 1
        elif result == agent1_player:
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
    parser = argparse.ArgumentParser(description="TicTacToe AI - Improved Implementation")
    parser.add_argument('command', nargs='?', choices=['train', 'play', 'battle', 'analyze'], 
                       help='Command to run')
    parser.add_argument('--episodes', type=int, default=15000, help='Training episodes')
    parser.add_argument('--model', default='improved_tictactoe_ai.pt', help='Model file path')
    parser.add_argument('--games', type=int, default=100, help='Number of test games')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_improved_ai(args.episodes, args.model)
    elif args.command == 'play':
        play_vs_ai(args.model)
    elif args.command == 'battle':
        ai_battle(args.model, num_games=args.games)
    elif args.command == 'analyze':
        analyze_ai_performance(args.model, args.games)
    else:
        # Interactive menu
        print("🎮 TicTacToe AI - Improved Implementation")
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
                    episodes = int(input("Training episodes (default 15000): ") or "15000")
                    train_improved_ai(episodes)
                elif choice == '2':
                    play_vs_ai()
                elif choice == '3':
                    games = int(input("Number of battle games (default 100): ") or "100")
                    ai_battle(num_games=games)
                elif choice == '4':
                    games = int(input("Number of test games (default 100): ") or "100")
                    analyze_ai_performance("improved_tictactoe_ai.pt", games)
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