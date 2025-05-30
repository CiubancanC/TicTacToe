import numpy as np
import random
import pickle
import math
from collections import deque
from typing import Optional, Tuple, List, Dict

class TicTacToe:
    def __init__(self):
        # Use numpy array for vectorized operations
        self.board = np.full(9, ' ', dtype=str)
        self.current_winner = None
        
        # Precompute winning positions for faster checking
        self.winning_positions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]

    def print_board(self):
        board_2d = self.board.reshape(3, 3)
        for row in board_2d:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        number_board = np.arange(9).reshape(3, 3)
        for row in number_board:
            print('| ' + ' | '.join(map(str, row)) + ' |')

    def available_moves(self):
        return np.where(self.board == ' ')[0].tolist()

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return np.sum(self.board == ' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner_vectorized(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner_vectorized(self, square, letter):
        # Vectorized winner checking
        for positions in self.winning_positions:
            if square in positions:
                if np.all(self.board[positions] == letter):
                    return True
        return False

    def get_board_hash(self):
        # Convert board to integer hash for efficient storage
        mapping = {' ': 0, 'X': 1, 'O': 2}
        hash_val = 0
        for i, cell in enumerate(self.board):
            hash_val += mapping[cell] * (3 ** i)
        return hash_val

class Player:
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        pass

class RandomPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        return random.choice(game.available_moves())

class SmartRandomPlayer(Player):
    def __init__(self, letter, smart_probability=0.3):
        super().__init__(letter)
        self.smart_probability = smart_probability
        self.opponent = 'O' if letter == 'X' else 'X'

    def get_move(self, game):
        available = game.available_moves()
        
        if random.random() < self.smart_probability:
            # Vectorized check for winning/blocking moves
            for move in available:
                # Check winning move
                game.board[move] = self.letter
                if game.winner_vectorized(move, self.letter):
                    game.board[move] = ' '
                    return move
                game.board[move] = ' '
            
            for move in available:
                # Check blocking move
                game.board[move] = self.opponent
                if game.winner_vectorized(move, self.opponent):
                    game.board[move] = ' '
                    return move
                game.board[move] = ' '
        
        return random.choice(available)

class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(self.letter + '\'s turn. Input move (0-8):')
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.')
        return val

class ExperienceReplay:
    """Experience replay buffer for more stable learning"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class LearningRateScheduler:
    """Learning rate scheduling for better convergence"""
    def __init__(self, initial_lr=0.1, decay_type='exponential', decay_rate=0.9999):
        self.initial_lr = initial_lr
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.step = 0
    
    def get_lr(self):
        if self.decay_type == 'exponential':
            return self.initial_lr * (self.decay_rate ** self.step)
        elif self.decay_type == 'step':
            return self.initial_lr * (0.5 ** (self.step // 10000))
        elif self.decay_type == 'cosine':
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * self.step / 100000))
        else:
            return self.initial_lr
    
    def step_scheduler(self):
        self.step += 1

class QLearningAgent(Player):
    def __init__(self, letter, alpha=0.1, gamma=0.9, epsilon=0.1, 
                 train_mode=True, use_experience_replay=True, 
                 batch_size=32, use_lr_scheduler=True):
        super().__init__(letter)
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.train_mode = train_mode
        
        # Experience replay
        self.use_experience_replay = use_experience_replay
        self.experience_replay = ExperienceReplay(capacity=10000)
        self.batch_size = batch_size
        
        # Learning rate scheduler
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler = LearningRateScheduler(initial_lr=alpha)
        
        # Episode tracking
        self.current_episode = []
        
        # Q-value normalization parameters
        self.q_mean = 0.0
        self.q_std = 1.0
        self.update_count = 0

    def get_state_hash(self, game):
        return game.get_board_hash()

    def get_q_value(self, state_hash, action):
        q_val = self.q_table.get((state_hash, action), 0.0)
        # Clip Q-values for numerical stability
        return np.clip(q_val, -10, 10)

    def normalize_q_values(self):
        """Normalize Q-values to prevent numerical instability"""
        if len(self.q_table) > 100:  # Only normalize after sufficient entries
            q_values = np.array(list(self.q_table.values()))
            self.q_mean = np.mean(q_values)
            self.q_std = np.std(q_values) + 1e-8  # Add epsilon for stability
            
            # Normalize stored Q-values
            for key in self.q_table:
                self.q_table[key] = (self.q_table[key] - self.q_mean) / self.q_std

    def choose_action(self, game):
        state_hash = self.get_state_hash(game)
        available_moves = game.available_moves()

        if not available_moves:
            return None

        if self.train_mode and random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        else:
            # Vectorized Q-value computation
            q_values = np.array([self.get_q_value(state_hash, move) for move in available_moves])
            max_q_indices = np.where(q_values == q_values.max())[0]
            
            # If all Q-values are zero, choose randomly
            if np.all(q_values == 0):
                return random.choice(available_moves)
            
            return available_moves[random.choice(max_q_indices)]

    def get_move(self, game):
        move = self.choose_action(game)
        if self.train_mode and move is not None:
            self.current_episode.append({
                'state': self.get_state_hash(game),
                'action': move,
                'board': game.board.copy()
            })
        return move

    def batch_update(self):
        """Perform batch updates from experience replay"""
        if len(self.experience_replay) < self.batch_size:
            return
        
        batch = self.experience_replay.sample(self.batch_size)
        
        # Get current learning rate
        current_lr = self.lr_scheduler.get_lr() if self.use_lr_scheduler else self.alpha
        
        # Vectorized batch update
        for state, action, reward, next_state, done in batch:
            current_q = self.get_q_value(state, action)
            
            if done:
                target = reward
            else:
                # Create dummy game for next state
                dummy_game = TicTacToe()
                dummy_game.board = next_state
                next_moves = dummy_game.available_moves()
                
                if next_moves:
                    next_state_hash = dummy_game.get_board_hash()
                    next_q_values = [self.get_q_value(next_state_hash, move) for move in next_moves]
                    max_next_q = max(next_q_values)
                else:
                    max_next_q = 0
                
                target = reward + self.gamma * max_next_q
            
            # Q-learning update with current learning rate
            new_q = current_q + current_lr * (target - current_q)
            self.q_table[(state, action)] = new_q
        
        # Step the learning rate scheduler
        if self.use_lr_scheduler:
            self.lr_scheduler.step_scheduler()
        
        # Periodically normalize Q-values
        self.update_count += 1
        if self.update_count % 1000 == 0:
            self.normalize_q_values()

    def learn(self, final_reward):
        if not self.train_mode:
            return

        # Process episode for experience replay
        episode_length = len(self.current_episode)
        
        for i in range(episode_length):
            state = self.current_episode[i]['state']
            action = self.current_episode[i]['action']
            
            if i == episode_length - 1:
                # Terminal state
                reward = final_reward
                next_state = None
                done = True
            else:
                # Intermediate state
                reward = 0
                next_state = self.current_episode[i + 1]['board']
                done = False
            
            if self.use_experience_replay:
                self.experience_replay.add(state, action, reward, next_state, done)
            else:
                # Traditional Q-learning update
                current_q = self.get_q_value(state, action)
                
                if done:
                    target = reward
                else:
                    dummy_game = TicTacToe()
                    dummy_game.board = next_state
                    next_moves = dummy_game.available_moves()
                    
                    if next_moves:
                        next_state_hash = dummy_game.get_board_hash()
                        next_q_values = [self.get_q_value(next_state_hash, move) for move in next_moves]
                        max_next_q = max(next_q_values)
                    else:
                        max_next_q = 0
                    
                    target = reward + self.gamma * max_next_q
                
                new_q = current_q + self.alpha * (target - current_q)
                self.q_table[(state, action)] = new_q
        
        # Perform batch update if using experience replay
        if self.use_experience_replay and len(self.experience_replay) >= self.batch_size:
            self.batch_update()
        
        # Clear episode history
        self.current_episode = []

    def save_model(self, filename="q_table.pkl"):
        save_data = {
            'q_table': self.q_table,
            'q_mean': self.q_mean,
            'q_std': self.q_std,
            'lr_step': self.lr_scheduler.step if self.use_lr_scheduler else 0
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename="q_table.pkl"):
        try:
            with open(filename, 'rb') as f:
                save_data = pickle.load(f)
                if isinstance(save_data, dict) and 'q_table' in save_data:
                    self.q_table = save_data['q_table']
                    self.q_mean = save_data.get('q_mean', 0.0)
                    self.q_std = save_data.get('q_std', 1.0)
                    if self.use_lr_scheduler:
                        self.lr_scheduler.step = save_data.get('lr_step', 0)
                else:
                    # Old format compatibility
                    self.q_table = save_data
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print("No existing model found. Starting with an empty Q-table.")

def play(game, x_player, o_player, print_game=True):
    if print_game:
        game.print_board_nums()

    letter = 'X'
    while game.empty_squares():
        if letter == 'O':
            square = o_player.get_move(game)
        else:
            square = x_player.get_move(game)

        if square is None:
            break

        if game.make_move(square, letter):
            if print_game:
                print(letter + f' makes a move to square {square}')
                game.print_board()
                print('')

            if game.current_winner:
                if print_game:
                    print(letter + ' wins!')
                return letter
            
            letter = 'O' if letter == 'X' else 'X'

    if print_game:
        print('It\'s a tie!')
    return 'Draw'

def diverse_training(agent_x, agent_o, num_episodes, progress_callback=None):
    """Enhanced training with multiple phases and opponent diversity"""
    print("Starting enhanced diverse training...")
    
    # Create diverse opponents
    opponents_x = [
        QLearningAgent('X', epsilon=0.3, alpha=0.15, gamma=0.9, train_mode=True, 
                      use_experience_replay=False),  # Traditional Q-learning
        QLearningAgent('X', epsilon=0.1, alpha=0.05, gamma=0.95, train_mode=True,
                      use_experience_replay=True),   # With experience replay
        RandomPlayer('X'),
        SmartRandomPlayer('X', smart_probability=0.3),
        SmartRandomPlayer('X', smart_probability=0.6)
    ]
    
    opponents_o = [
        QLearningAgent('O', epsilon=0.3, alpha=0.15, gamma=0.9, train_mode=True,
                      use_experience_replay=False),
        QLearningAgent('O', epsilon=0.1, alpha=0.05, gamma=0.95, train_mode=True,
                      use_experience_replay=True),
        RandomPlayer('O'),
        SmartRandomPlayer('O', smart_probability=0.3),
        SmartRandomPlayer('O', smart_probability=0.6)
    ]
    
    episodes_per_phase = num_episodes // 3
    
    # Phase 1: Self-play and Q-learning opponents
    print("Phase 1: Training against Q-learning agents...")
    for i in range(episodes_per_phase):
        game = TicTacToe()
        
        winner = play(game, agent_x, agent_o, print_game=False)
        
        # Reward shaping for better learning
        if winner == agent_x.letter:
            agent_x.learn(1.0)
            agent_o.learn(-1.0)
        elif winner == agent_o.letter:
            agent_x.learn(-1.0)
            agent_o.learn(1.0)
        else:
            # Small positive reward for draws to encourage defensive play
            agent_x.learn(0.1)
            agent_o.learn(0.1)
        
        if (i+1) % (episodes_per_phase // 5) == 0:
            print(f"Phase 1: {i+1}/{episodes_per_phase} episodes completed.")
            if progress_callback:
                progress_callback(i+1, episodes_per_phase, 1)
    
    # Phase 2: Diverse opponents
    print("Phase 2: Training against diverse opponents...")
    for i in range(episodes_per_phase):
        game = TicTacToe()
        
        # Randomly choose opponents
        opp_x = random.choice(opponents_x)
        opp_o = random.choice(opponents_o)
        
        # Train agent_x
        winner = play(game, agent_x, opp_o, print_game=False)
        
        if winner == agent_x.letter:
            agent_x.learn(1.0)
            if hasattr(opp_o, 'learn'):
                opp_o.learn(-1.0)
        elif winner == opp_o.letter:
            agent_x.learn(-1.0)
            if hasattr(opp_o, 'learn'):
                opp_o.learn(1.0)
        else:
            agent_x.learn(0.1)
            if hasattr(opp_o, 'learn'):
                opp_o.learn(0.1)
        
        # Train agent_o
        game = TicTacToe()
        winner = play(game, opp_x, agent_o, print_game=False)
        
        if winner == agent_o.letter:
            agent_o.learn(1.0)
            if hasattr(opp_x, 'learn'):
                opp_x.learn(-1.0)
        elif winner == opp_x.letter:
            agent_o.learn(-1.0)
            if hasattr(opp_x, 'learn'):
                opp_x.learn(1.0)
        else:
            agent_o.learn(0.1)
            if hasattr(opp_x, 'learn'):
                opp_x.learn(0.1)
        
        if (i+1) % (episodes_per_phase // 5) == 0:
            print(f"Phase 2: {i+1}/{episodes_per_phase} episodes completed.")
            if progress_callback:
                progress_callback(i+1, episodes_per_phase, 2)
    
    # Phase 3: Final refinement with reduced exploration
    print("Phase 3: Final refinement phase...")
    original_epsilon_x = agent_x.epsilon
    original_epsilon_o = agent_o.epsilon
    
    agent_x.epsilon = 0.05
    agent_o.epsilon = 0.05
    
    for i in range(episodes_per_phase):
        game = TicTacToe()
        
        winner = play(game, agent_x, agent_o, print_game=False)
        
        if winner == agent_x.letter:
            agent_x.learn(1.0)
            agent_o.learn(-1.0)
        elif winner == agent_o.letter:
            agent_x.learn(-1.0)
            agent_o.learn(1.0)
        else:
            agent_x.learn(0.1)
            agent_o.learn(0.1)
        
        # Gradually reduce epsilon further
        if i % 1000 == 0:
            agent_x.epsilon *= 0.99
            agent_o.epsilon *= 0.99
        
        if (i+1) % (episodes_per_phase // 5) == 0:
            print(f"Phase 3: {i+1}/{episodes_per_phase} episodes completed.")
            if progress_callback:
                progress_callback(i+1, episodes_per_phase, 3)
    
    # Restore original epsilon values
    agent_x.epsilon = original_epsilon_x
    agent_o.epsilon = original_epsilon_o
    
    print("Enhanced diverse training complete.")

class MinimaxPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        return self.minimax(game, self.letter)['position']

    def minimax(self, state, player):
        max_player = self.letter
        other_player = 'O' if player == 'X' else 'X'

        if state.current_winner == other_player:
            return {'position': None, 'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
                state.num_empty_squares() + 1)}
        elif not state.empty_squares():
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -math.inf}
        else:
            best = {'position': None, 'score': math.inf}

        for possible_move in state.available_moves():
            state.make_move(possible_move, player)
            sim_score = self.minimax(state, other_player)
            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move

            if player == max_player:
                if sim_score['score'] > best['score']:
                    best = sim_score
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
        return best

def evaluate_against_variety(agent, num_games_per_opponent=10):
    """Evaluate the agent against various opponent types"""
    print(f"\n--- Comprehensive Evaluation of {type(agent).__name__} ---")
    
    agent.train_mode = False
    
    opponents = {
        'Random': RandomPlayer('O' if agent.letter == 'X' else 'X'),
        'Smart Random (30%)': SmartRandomPlayer('O' if agent.letter == 'X' else 'X', 0.3),
        'Smart Random (60%)': SmartRandomPlayer('O' if agent.letter == 'X' else 'X', 0.6),
        'Minimax (Perfect)': MinimaxPlayer('O' if agent.letter == 'X' else 'X')
    }
    
    total_wins = 0
    total_losses = 0
    total_draws = 0
    
    for opp_name, opponent in opponents.items():
        wins = losses = draws = 0
        
        print(f"\nTesting against {opp_name}...")
        
        for i in range(num_games_per_opponent):
            game = TicTacToe()
            
            if agent.letter == 'X':
                winner = play(game, agent, opponent, print_game=False)
            else:
                winner = play(game, opponent, agent, print_game=False)
            
            if winner == agent.letter:
                wins += 1
            elif winner == opponent.letter:
                losses += 1
            else:
                draws += 1
        
        win_rate = wins / num_games_per_opponent * 100
        loss_rate = losses / num_games_per_opponent * 100
        draw_rate = draws / num_games_per_opponent * 100
        
        print(f"  Results: {wins}W/{losses}L/{draws}D")
        print(f"  Win rate: {win_rate:.1f}%, Loss rate: {loss_rate:.1f}%, Draw rate: {draw_rate:.1f}%")
        
        total_wins += wins
        total_losses += losses
        total_draws += draws
    
    total_games = len(opponents) * num_games_per_opponent
    overall_win_rate = total_wins / total_games * 100
    overall_loss_rate = total_losses / total_games * 100
    overall_draw_rate = total_draws / total_games * 100
    
    print(f"\n--- Overall Performance ---")
    print(f"Total games: {total_games}")
    print(f"Overall win rate: {overall_win_rate:.1f}%")
    print(f"Overall loss rate: {overall_loss_rate:.1f}%")
    print(f"Overall draw rate: {overall_draw_rate:.1f}%")

if __name__ == '__main__':
    # Create enhanced agents
    q_agent_x = QLearningAgent('X', epsilon=0.2, alpha=0.1, gamma=0.9, 
                              train_mode=True, use_experience_replay=True,
                              batch_size=32, use_lr_scheduler=True)
    q_agent_o = QLearningAgent('O', epsilon=0.2, alpha=0.1, gamma=0.9,
                              train_mode=True, use_experience_replay=True,
                              batch_size=32, use_lr_scheduler=True)

    # Train agents
    diverse_training(q_agent_x, q_agent_o, num_episodes=100000)

    # Save models
    q_agent_x.save_model("improved_q_table_X.pkl")
    q_agent_o.save_model("improved_q_table_O.pkl")

    # Evaluate
    evaluate_against_variety(q_agent_x, num_games_per_opponent=100)
    evaluate_against_variety(q_agent_o, num_games_per_opponent=100)

    # Interactive play
    print("\n" + "="*50)
    print("READY TO PLAY!")
    print("="*50)
    
    while True:
        choice = input("\nChoose your role:\n1. Play as X (go first)\n2. Play as O (go second)\n3. Quit\nEnter choice (1/2/3): ")
        
        if choice == '3':
            break
        elif choice == '1':
            human_player = HumanPlayer('X')
            ai_player = QLearningAgent('O', train_mode=False, use_experience_replay=False)
            ai_player.load_model("improved_q_table_O.pkl")
            print("\nYou are X (going first), AI is O")
            play(TicTacToe(), human_player, ai_player, print_game=True)
        elif choice == '2':
            human_player = HumanPlayer('O')
            ai_player = QLearningAgent('X', train_mode=False, use_experience_replay=False)
            ai_player.load_model("improved_q_table_X.pkl")
            print("\nAI is X (going first), you are O")
            play(TicTacToe(), ai_player, human_player, print_game=True)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")