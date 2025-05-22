import random
import pickle
import math

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None # keep track of the winner!

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        # 0 | 1 | 2 etc (tells us what number corresponds to what box)
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
        self.q_table = {}  # (board_tuple, action) -> Q-value
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon # exploration-exploitation trade-off
        self.history = [] # To store (state, action, reward) for learning
        self.train_mode = train_mode

    def get_state(self, game):
        return tuple(game.board)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, game):
        state = self.get_state(game)
        available_moves = game.available_moves()

        if not available_moves: # No moves available, game is over
            return None

        if self.train_mode and random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random move
            return random.choice(available_moves)
        else:
            # Exploitation: choose the best move based on Q-values
            q_values_for_moves = {move: self.get_q_value(state, move) for move in available_moves}
            
            # Find the maximum Q-value
            max_q = -math.inf
            if q_values_for_moves: # Ensure there are entries to find max from
                max_q = max(q_values_for_moves.values())

            # Collect all moves that have this maximum Q-value
            best_moves = [move for move, q_val in q_values_for_moves.items() if q_val == max_q]
            
            # If all Q-values are 0 (or all are the same, e.g., initial state), pick randomly among them
            if not best_moves or all(q_val == 0.0 for q_val in q_values_for_moves.values()):
                return random.choice(available_moves)
            
            return random.choice(best_moves) # In case of ties, choose randomly

    def get_move(self, game):
        move = self.choose_action(game)
        if self.train_mode and move is not None: # Only record if a move was chosen
            self.history.append((self.get_state(game), move))
        return move

    def learn(self, final_reward):
        if not self.train_mode:
            return

        # Iterate through the history in reverse order
        for i in reversed(range(len(self.history))):
            state, action = self.history[i]
            
            # If it's the last move, the future reward is the final reward
            if i == len(self.history) - 1:
                target = final_reward
            else:
                # Get the next state from the history
                next_state_board, _ = self.history[i+1] # We only need the board from the next state
                
                # Create a dummy game object to get available moves for the next state
                dummy_game = TicTacToe()
                dummy_game.board = list(next_state_board) # Convert tuple back to list
                
                # Get max Q-value for the next state
                next_available_moves = dummy_game.available_moves()
                if next_available_moves: # Check if there are available moves
                    max_q_next = max([self.get_q_value(next_state_board, next_action) for next_action in next_available_moves])
                else: # Game ended (draw or win/loss already accounted for in final_reward)
                    max_q_next = 0 # No future actions possible
                
                target = final_reward + self.gamma * max_q_next
            
            current_q = self.get_q_value(state, action)
            new_q = current_q + self.alpha * (target - current_q)
            self.q_table[(state, action)] = new_q
        self.history = [] # Clear history after learning

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

class MinimaxPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        # The Minimax algorithm will find the optimal move directly from any state.
        # It explores the game tree to find the move that maximizes its score
        # assuming the opponent plays optimally to minimize its score.
        return self.minimax(game, self.letter)['position']

    def minimax(self, state, player):
        max_player = self.letter  # yourself
        other_player = 'O' if player == 'X' else 'X'

        # Check for terminal states
        if state.current_winner == other_player:
            # If the other player just won, it's a loss for the current 'player'
            # Score: positive for max_player win, negative for max_player loss
            # (state.num_empty_squares() + 1) gives preference to quicker wins/slower losses
            return {'position': None, 'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
                state.num_empty_squares() + 1)}
        elif not state.empty_squares():
            # It's a draw
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -math.inf}  # maximize score for max_player
        else:
            best = {'position': None, 'score': math.inf}  # minimize score for other_player

        for possible_move in state.available_moves():
            # Step 1: make a move, try that spot
            state.make_move(possible_move, player)
            
            # Step 2: recurse using minimax to simulate a game after making that move
            sim_score = self.minimax(state, other_player) # simulate opponent's turn

            # Step 3: undo the move (clean up the board for next iteration)
            state.board[possible_move] = ' '
            state.current_winner = None # Reset winner as we're undoing the move

            # Step 4: update the dictionaries if necessary
            sim_score['position'] = possible_move # Store the move that led to this score

            if player == max_player: # Maximizing player
                if sim_score['score'] > best['score']:
                    best = sim_score
            else: # Minimizing player
                if sim_score['score'] < best['score']:
                    best = sim_score
        return best

def play(game, x_player, o_player, print_game=True):
    if print_game:
        game.print_board_nums()

    letter = 'X' # starting letter
    while game.empty_squares():
        if letter == 'O':
            square = o_player.get_move(game)
        else:
            square = x_player.get_move(game)

        # Handle case where choose_action might return None (no available moves)
        if square is None:
            break

        if game.make_move(square, letter):
            if print_game:
                print(letter + f' makes a move to square {square}')
                game.print_board()
                print('') # empty line

            if game.current_winner:
                if print_game:
                    print(letter + ' wins!')
                return letter
            
            letter = 'O' if letter == 'X' else 'X' # switches player

    if print_game:
        print('It\'s a tie!')
    return 'Draw'

def train_q_learning_agent(agent_x, agent_o, num_episodes):
    print("Starting Q-Learning training...")
    for i in range(num_episodes):
        game = TicTacToe()
        
        # Reset game for agents for each episode
        agent_x.history = []
        agent_o.history = []
        
        winner = play(game, agent_x, agent_o, print_game=False) # Don't print during training

        # Assign rewards
        if winner == agent_x.letter:
            agent_x.learn(1)  # X wins
            agent_o.learn(-1) # O loses
        elif winner == agent_o.letter:
            agent_x.learn(-1) # X loses
            agent_o.learn(1)  # O wins
        else: # Draw
            agent_x.learn(0.1) # Small positive reward for a draw
            agent_o.learn(0.1)

        if (i+1) % (num_episodes // 10) == 0:
            print(f"Training episode {i+1}/{num_episodes} completed.")
    print("Q-Learning training complete.")

def evaluate_agent(agent1, agent2, num_games, agent1_is_x=True):
    wins1 = 0
    wins2 = 0
    draws = 0

    print(f"\nEvaluating {type(agent1).__name__} vs {type(agent2).__name__} for {num_games} games...")

    for i in range(num_games):
        game = TicTacToe()
        
        if agent1_is_x:
            winner = play(game, agent1, agent2, print_game=False)
        else:
            # agent1 plays as O, agent2 plays as X
            winner = play(game, agent2, agent1, print_game=False)
        
        # Determine who won from the perspective of agent1 and agent2
        if (agent1_is_x and winner == agent1.letter) or (not agent1_is_x and winner == agent1.letter):
            wins1 += 1
        elif (agent1_is_x and winner == agent2.letter) or (not agent1_is_x and winner == agent2.letter):
            wins2 += 1
        else:
            draws += 1
        
        if (i+1) % (num_games // 10) == 0:
            print(f"Evaluation game {i+1}/{num_games} completed.")

    print("\n--- Evaluation Results ---")
    print(f"{type(agent1).__name__} wins: {wins1}")
    print(f"{type(agent2).__name__} wins: {wins2}")
    print(f"Draws: {draws}")
    
    # Improved print statements for clarity on who is X and O
    if agent1_is_x:
        print(f"Win rate for {type(agent1).__name__} (as X): {wins1 / num_games * 100:.2f}%")
        print(f"Win rate for {type(agent2).__name__} (as O): {wins2 / num_games * 100:.2f}%")
    else:
        print(f"Win rate for {type(agent1).__name__} (as O): {wins1 / num_games * 100:.2f}%")
        print(f"Win rate for {type(agent2).__name__} (as X): {wins2 / num_games * 100:.2f}%")

    print(f"Draw rate: {draws / num_games * 100:.2f}%")
    return wins1, wins2, draws

if __name__ == '__main__':
    # --- 1. Train the Q-Learning AI ---
    # Using slightly higher epsilon for more exploration, which can help find optimal strategies
    # if the initial rewards are sparse or the environment is complex.
    q_agent_x = QLearningAgent('X', epsilon=0.2, alpha=0.1, gamma=0.9, train_mode=True)
    q_agent_o = QLearningAgent('O', epsilon=0.2, alpha=0.1, gamma=0.9, train_mode=True)

    # Optional: Load a pre-trained model if available
    # q_agent_x.load_model("q_table_X.pkl")
    # q_agent_o.load_model("q_table_O.pkl")

    # Train two Q-learning agents against each other.
    # A sufficient number of episodes is crucial for convergence.
    train_q_learning_agent(q_agent_x, q_agent_o, num_episodes=75000) # Increased episodes

    # Save the trained models
    q_agent_x.save_model("q_table_X.pkl")
    q_agent_o.save_model("q_table_O.pkl")

    # Set agents to evaluation mode
    q_agent_x.train_mode = False
    q_agent_o.train_mode = False

    # --- 2. Play against the Trained AI ---
    print("\n--- Playing against the Trained AI (Human as O) ---")
    human_player_o = HumanPlayer('O') # You play as 'O'
    q_agent_x_play = QLearningAgent('X', train_mode=False)
    q_agent_x_play.load_model("q_table_X.pkl") # Load the trained model
    
    # Uncomment the line below to play against the AI as Player X
    # play(TicTacToe(), q_agent_x_play, human_player_o, print_game=True)
    
    print("\n--- Playing against the Trained AI (Human as X) ---")
    human_player_x = HumanPlayer('X') # You play as 'X'
    q_agent_o_play = QLearningAgent('O', train_mode=False)
    q_agent_o_play.load_model("q_table_O.pkl") # Load the trained model

    # Uncomment the line below to play against the AI as Player O
    # play(TicTacToe(), human_player_x, q_agent_o_play, print_game=True)


    # --- 3. Compete against Minimax Perfect Algorithms ---
    minimax_x = MinimaxPlayer('X')
    minimax_o = MinimaxPlayer('O')

    q_agent_x_compete = QLearningAgent('X', train_mode=False)
    q_agent_x_compete.load_model("q_table_X.pkl")

    q_agent_o_compete = QLearningAgent('O', train_mode=False)
    q_agent_o_compete.load_model("q_table_O.pkl")

    num_eval_games = 100

    # Q-Learning AI as Player 1 (X) vs Minimax as Player 2 (O)
    print("\n--- Q-Learning AI (X) vs Minimax (O) ---")
    evaluate_agent(q_agent_x_compete, minimax_o, num_eval_games, agent1_is_x=True)

    # Q-Learning AI as Player 2 (O) vs Minimax as Player 1 (X)
    print("\n--- Q-Learning AI (O) vs Minimax (X) ---")
    evaluate_agent(q_agent_o_compete, minimax_x, num_eval_games, agent1_is_x=False)

    print("\n--- Minimax (X) vs Minimax (O) (Expected 100% Draw Rate) ---")
    # Small number of games for Minimax vs Minimax as it's deterministic and only needs a few to confirm.
    # The outcome will be the same for every game.
    evaluate_agent(minimax_x, minimax_o, 100, agent1_is_x=True)
