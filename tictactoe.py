import random
import pickle
import math

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        # check the row
        row_ind = square // 3
        row = self.board[row_ind*3 : (row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
        # check the column
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        # check diagonals
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False

class Player:
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        pass

class RandomPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        square = random.choice(game.available_moves())
        return square

class SmartRandomPlayer(Player):
    """A random player that occasionally makes smart moves"""
    def __init__(self, letter, smart_probability=0.3):
        super().__init__(letter)
        self.smart_probability = smart_probability
        self.opponent = 'O' if letter == 'X' else 'X'

    def get_move(self, game):
        available = game.available_moves()
        
        # Sometimes make a smart move
        if random.random() < self.smart_probability:
            # Check for winning moves
            for move in available:
                game.board[move] = self.letter
                if game.winner(move, self.letter):
                    game.board[move] = ' '  # Undo
                    return move
                game.board[move] = ' '  # Undo
            
            # Check for blocking moves
            for move in available:
                game.board[move] = self.opponent
                if game.winner(move, self.opponent):
                    game.board[move] = ' '  # Undo
                    return move
                game.board[move] = ' '  # Undo
        
        # Otherwise, random move
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

class QLearningAgent(Player):
    def __init__(self, letter, alpha=0.1, gamma=0.9, epsilon=0.1, train_mode=True):
        super().__init__(letter)
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.history = []
        self.train_mode = train_mode

    def get_state(self, game):
        return tuple(game.board)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, game):
        state = self.get_state(game)
        available_moves = game.available_moves()

        if not available_moves:
            return None

        if self.train_mode and random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        else:
            q_values_for_moves = {move: self.get_q_value(state, move) for move in available_moves}
            max_q = max(q_values_for_moves.values()) if q_values_for_moves else -math.inf
            best_moves = [move for move, q_val in q_values_for_moves.items() if q_val == max_q]
            
            if not best_moves or max_q == 0.0:
                return random.choice(available_moves)
            
            return random.choice(best_moves)

    def get_move(self, game):
        move = self.choose_action(game)
        if self.train_mode and move is not None:
            self.history.append((self.get_state(game), move))
        return move

    def learn(self, final_reward):
        if not self.train_mode:
            return

        for i in reversed(range(len(self.history))):
            state, action = self.history[i]
            
            if i == len(self.history) - 1:
                target = final_reward
            else:
                next_state_board, _ = self.history[i+1]
                dummy_game = TicTacToe()
                dummy_game.board = list(next_state_board)
                next_available_moves = dummy_game.available_moves()
                
                if next_available_moves:
                    max_q_next = max([self.get_q_value(next_state_board, next_action) for next_action in next_available_moves])
                else:
                    max_q_next = 0
                
                target = final_reward + self.gamma * max_q_next
            
            current_q = self.get_q_value(state, action)
            new_q = current_q + self.alpha * (target - current_q)
            self.q_table[(state, action)] = new_q
        
        self.history = []

    def save_model(self, filename="q_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def load_model(self, filename="q_table.pkl"):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")
        except FileNotFoundError:
            print("No existing Q-table found. Starting with an empty table.")

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

def diverse_training(agent_x, agent_o, num_episodes):
    """Train against a diverse set of opponents"""
    print("Starting diverse training...")
    
    # Create different types of opponents
    opponents_x = [
        QLearningAgent('X', epsilon=0.3, alpha=0.15, gamma=0.9, train_mode=True),
        QLearningAgent('X', epsilon=0.1, alpha=0.05, gamma=0.95, train_mode=True),
        RandomPlayer('X'),
        SmartRandomPlayer('X', smart_probability=0.3),
        SmartRandomPlayer('X', smart_probability=0.6)
    ]
    
    opponents_o = [
        QLearningAgent('O', epsilon=0.3, alpha=0.15, gamma=0.9, train_mode=True),
        QLearningAgent('O', epsilon=0.1, alpha=0.05, gamma=0.95, train_mode=True),
        RandomPlayer('O'),
        SmartRandomPlayer('O', smart_probability=0.3),
        SmartRandomPlayer('O', smart_probability=0.6)
    ]
    
    episodes_per_phase = num_episodes // 3
    
    # Phase 1: Train against other Q-learning agents
    print("Phase 1: Training against other Q-learning agents...")
    for i in range(episodes_per_phase):
        game = TicTacToe()
        agent_x.history = []
        agent_o.history = []
        
        winner = play(game, agent_x, agent_o, print_game=False)
        
        if winner == agent_x.letter:
            agent_x.learn(1)
            agent_o.learn(-1)
        elif winner == agent_o.letter:
            agent_x.learn(-1)
            agent_o.learn(1)
        else:
            agent_x.learn(0.1)
            agent_o.learn(0.1)
        
        if (i+1) % (episodes_per_phase // 5) == 0:
            print(f"Phase 1: {i+1}/{episodes_per_phase} episodes completed.")
    
    # Phase 2: Train against random and smart random opponents
    print("Phase 2: Training against random opponents...")
    for i in range(episodes_per_phase):
        game = TicTacToe()
        agent_x.history = []
        agent_o.history = []
        
        # Randomly choose opponents
        opp_x = random.choice(opponents_x)
        opp_o = random.choice(opponents_o)
        
        if hasattr(opp_x, 'history'):
            opp_x.history = []
        if hasattr(opp_o, 'history'):
            opp_o.history = []
        
        winner = play(game, agent_x, opp_o, print_game=False)
        
        if winner == agent_x.letter:
            agent_x.learn(1)
            if hasattr(opp_o, 'learn'):
                opp_o.learn(-1)
        elif winner == opp_o.letter:
            agent_x.learn(-1)
            if hasattr(opp_o, 'learn'):
                opp_o.learn(1)
        else:
            agent_x.learn(0.1)
            if hasattr(opp_o, 'learn'):
                opp_o.learn(0.1)
        
        # Train agent_o against opponents too
        game = TicTacToe()
        agent_o.history = []
        if hasattr(opp_x, 'history'):
            opp_x.history = []
        
        winner = play(game, opp_x, agent_o, print_game=False)
        
        if winner == agent_o.letter:
            agent_o.learn(1)
            if hasattr(opp_x, 'learn'):
                opp_x.learn(-1)
        elif winner == opp_x.letter:
            agent_o.learn(-1)
            if hasattr(opp_x, 'learn'):
                opp_x.learn(1)
        else:
            agent_o.learn(0.1)
            if hasattr(opp_x, 'learn'):
                opp_x.learn(0.1)
        
        if (i+1) % (episodes_per_phase // 5) == 0:
            print(f"Phase 2: {i+1}/{episodes_per_phase} episodes completed.")
    
    # Phase 3: Final training against each other with reduced exploration
    print("Phase 3: Final polishing phase...")
    agent_x.epsilon = 0.05  # Reduce exploration
    agent_o.epsilon = 0.05
    
    for i in range(episodes_per_phase):
        game = TicTacToe()
        agent_x.history = []
        agent_o.history = []
        
        winner = play(game, agent_x, agent_o, print_game=False)
        
        if winner == agent_x.letter:
            agent_x.learn(1)
            agent_o.learn(-1)
        elif winner == agent_o.letter:
            agent_x.learn(-1)
            agent_o.learn(1)
        else:
            agent_x.learn(0.1)
            agent_o.learn(0.1)
        
        if (i+1) % (episodes_per_phase // 5) == 0:
            print(f"Phase 3: {i+1}/{episodes_per_phase} episodes completed.")
    
    print("Diverse training complete.")

def evaluate_against_variety(agent, num_games_per_opponent=10):
    """Evaluate the agent against various opponent types"""
    print(f"\n--- Comprehensive Evaluation of {type(agent).__name__} ---")
    
    # Set agent to evaluation mode
    agent.train_mode = False
    
    opponents = {
        'Random': RandomPlayer('O' if agent.letter == 'X' else 'X'),
        'Smart Random (30%)': SmartRandomPlayer('O' if agent.letter == 'X' else 'X', 0.3),
        'Smart Random (60%)': SmartRandomPlayer('O' if agent.letter == 'X' else 'X', 0.6),
        'Q-Learning (Aggressive)': QLearningAgent('O' if agent.letter == 'X' else 'X', 
                                                 epsilon=0.0, alpha=0.2, gamma=0.8, train_mode=False),
        'Q-Learning (Conservative)': QLearningAgent('O' if agent.letter == 'X' else 'X', 
                                                   epsilon=0.0, alpha=0.05, gamma=0.95, train_mode=False)
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

class MinimaxPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        return self.minimax(game, self.letter)['position']

    def minimax(self, state, player):
        max_player = self.letter
        other_player = 'O' if player == 'X' else 'X'

        # Check for terminal states
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

def evaluate_against_minimax(agent, num_games=10):
    """Test the agent specifically against Minimax"""
    print(f"\n--- Testing {type(agent).__name__} vs Minimax ---")
    
    agent.train_mode = False
    minimax_opponent = MinimaxPlayer('O' if agent.letter == 'X' else 'X')
    
    wins = losses = draws = 0
    
    for i in range(num_games):
        game = TicTacToe()
        
        if agent.letter == 'X':
            winner = play(game, agent, minimax_opponent, print_game=False)
        else:
            winner = play(game, minimax_opponent, agent, print_game=False)
        
        if winner == agent.letter:
            wins += 1
        elif winner == minimax_opponent.letter:
            losses += 1
        else:
            draws += 1
        
        if (i+1) % (num_games // 10) == 0:
            print(f"Progress: {i+1}/{num_games} games completed.")
    
    win_rate = wins / num_games * 100
    loss_rate = losses / num_games * 100
    draw_rate = draws / num_games * 100
    
    print(f"\n--- Results vs Minimax ---")
    print(f"Agent ({agent.letter}) vs Minimax ({'O' if agent.letter == 'X' else 'X'})")
    print(f"Wins: {wins} ({win_rate:.1f}%)")
    print(f"Losses: {losses} ({loss_rate:.1f}%)")
    print(f"Draws: {draws} ({draw_rate:.1f}%)")
    
    if losses == 0:
        print("🎉 Perfect! Agent never loses to Minimax!")
    elif wins > 0:
        print("🤔 Interesting! Agent occasionally beats Minimax (shouldn't happen with perfect play)")
    elif draw_rate > 90:
        print("✅ Excellent! Agent plays near-optimally against perfect opponent")
    else:
        print("⚠️  Agent loses some games to Minimax - room for improvement")
    
    return wins, losses, draws

if __name__ == '__main__':
    # Create agents with slightly different parameters for diversity
    q_agent_x = QLearningAgent('X', epsilon=0.2, alpha=0.1, gamma=0.9, train_mode=True)
    q_agent_o = QLearningAgent('O', epsilon=0.2, alpha=0.1, gamma=0.9, train_mode=True)

    # Optional: Load pre-trained models
    # q_agent_x.load_model("diverse_q_table_X.pkl")
    # q_agent_o.load_model("diverse_q_table_O.pkl")

    # Diverse training approach
    diverse_training(q_agent_x, q_agent_o, num_episodes=500000)

    # Save the trained models
    q_agent_x.save_model("diverse_q_table_X.pkl")
    q_agent_o.save_model("diverse_q_table_O.pkl")

    # Comprehensive evaluation
    evaluate_against_variety(q_agent_x, num_games_per_opponent=10)
    evaluate_against_variety(q_agent_o, num_games_per_opponent=10)

    # Test against Minimax
    print("\n" + "="*60)
    print("TESTING AGAINST PERFECT MINIMAX PLAYER")
    print("="*60)
    
    evaluate_against_minimax(q_agent_x, num_games=10)
    evaluate_against_minimax(q_agent_o, num_games=10)

    # Play against human
    print("\n" + "="*50)
    print("READY TO PLAY!")
    print("="*50)
    
    while True:
        choice = input("\nChoose your role:\n1. Play as X (go first)\n2. Play as O (go second)\n3. Quit\nEnter choice (1/2/3): ")
        
        if choice == '3':
            break
        elif choice == '1':
            human_player = HumanPlayer('X')
            ai_player = QLearningAgent('O', train_mode=False)
            ai_player.load_model("diverse_q_table_O.pkl")
            print("\nYou are X (going first), AI is O")
            play(TicTacToe(), human_player, ai_player, print_game=True)
        elif choice == '2':
            human_player = HumanPlayer('O')
            ai_player = QLearningAgent('X', train_mode=False)
            ai_player.load_model("diverse_q_table_X.pkl")
            print("\nAI is X (going first), you are O")
            play(TicTacToe(), ai_player, human_player, print_game=True)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    # Play against human
    print("\n" + "="*50)
    print("READY TO PLAY!")
    print("="*50)
    
    while True:
        choice = input("\nChoose your role:\n1. Play as X (go first)\n2. Play as O (go second)\n3. Quit\nEnter choice (1/2/3): ")
        
        if choice == '3':
            break
        elif choice == '1':
            human_player = HumanPlayer('X')
            ai_player = QLearningAgent('O', train_mode=False)
            ai_player.load_model("diverse_q_table_O.pkl")
            print("\nYou are X (going first), AI is O")
            play(TicTacToe(), human_player, ai_player, print_game=True)
        elif choice == '2':
            human_player = HumanPlayer('O')
            ai_player = QLearningAgent('X', train_mode=False)
            ai_player.load_model("diverse_q_table_X.pkl")
            print("\nAI is X (going first), you are O")
            play(TicTacToe(), ai_player, human_player, print_game=True)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")