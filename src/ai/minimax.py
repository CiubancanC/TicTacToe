"""
Minimax algorithm implementation for perfect Tic Tac Toe play.
This provides a guaranteed optimal strategy that will never lose and will win
whenever possible.
"""

import numpy as np

def evaluate_board(board):
    """
    Evaluate the board from the AI's perspective.
    Returns:
        10 if AI wins
        -10 if human wins
        0 otherwise (draw or game in progress)
    """
    # Check rows
    for i in range(3):
        if np.sum(board[i, :]) == 3:  # AI wins (all 1s)
            return 10
        if np.sum(board[i, :]) == -3:  # Human wins (all -1s)
            return -10

    # Check columns
    for i in range(3):
        if np.sum(board[:, i]) == 3:  # AI wins
            return 10
        if np.sum(board[:, i]) == -3:  # Human wins
            return -10

    # Check diagonals
    if board[0, 0] + board[1, 1] + board[2, 2] == 3:  # AI wins
        return 10
    if board[0, 0] + board[1, 1] + board[2, 2] == -3:  # Human wins
        return -10
    if board[0, 2] + board[1, 1] + board[2, 0] == 3:  # AI wins
        return 10
    if board[0, 2] + board[1, 1] + board[2, 0] == -3:  # Human wins
        return -10

    # No winner yet
    return 0

def is_board_moves_left(board):
    """Check if there are empty cells (moves left) on the board"""
    return 0 in board

def get_empty_cells(board):
    """Return a list of (row, col) tuples for all empty cells"""
    empty_cells = []
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                empty_cells.append((i, j))
    return empty_cells

def minimax(board, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
    """
    Minimax algorithm with alpha-beta pruning for Tic Tac Toe.
    
    Args:
        board: 3x3 numpy array representing the board state (1 for AI, -1 for human, 0 for empty)
        depth: Current depth in the game tree
        is_maximizing: True if maximizing player (AI), False if minimizing player (human)
        alpha, beta: Parameters for alpha-beta pruning
        
    Returns:
        The best score the current player can achieve
    """
    score = evaluate_board(board)
    
    # If AI has won the game, return score with depth consideration
    if score == 10:
        return score - depth  # Prefer winning sooner
    
    # If human has won the game, return score with depth consideration
    if score == -10:
        return score + depth  # Prefer losing later
    
    # If no moves left, it's a draw
    if not is_board_moves_left(board):
        return 0
    
    # AI's turn (maximizing player)
    if is_maximizing:
        best = -float('inf')
        
        for move in get_empty_cells(board):
            # Make the move
            board[move[0], move[1]] = 1
            
            # Recursively compute score for this move
            best = max(best, minimax(board, depth + 1, False, alpha, beta))
            
            # Undo the move
            board[move[0], move[1]] = 0
            
            # Alpha-beta pruning
            alpha = max(alpha, best)
            if beta <= alpha:
                break
                
        return best
    
    # Human's turn (minimizing player)
    else:
        best = float('inf')
        
        for move in get_empty_cells(board):
            # Make the move
            board[move[0], move[1]] = -1
            
            # Recursively compute score for this move
            best = min(best, minimax(board, depth + 1, True, alpha, beta))
            
            # Undo the move
            board[move[0], move[1]] = 0
            
            # Alpha-beta pruning
            beta = min(beta, best)
            if beta <= alpha:
                break
                
        return best

def find_best_move(board, player=1):
    """
    Find the optimal move for the specified player using minimax algorithm.
    
    Args:
        board: 3x3 numpy array representing the board state
        player: Which player to optimize for (1 for X, -1 for O)
        
    Returns:
        (row, col) tuple representing the best move
    """
    if player == 1:
        # Maximize for player 1 (X)
        best_val = -float('inf')
        best_move = (-1, -1)
        
        for move in get_empty_cells(board):
            # Make the move
            board[move[0], move[1]] = 1
            
            # Compute score for this move
            move_val = minimax(board, 0, False)
            
            # Undo the move
            board[move[0], move[1]] = 0
            
            # Update the best move if needed
            if move_val > best_val:
                best_move = move
                best_val = move_val
                
    else:  # player == -1
        # Maximize for player -1 (O) - this means minimizing the original score
        best_val = float('inf')
        best_move = (-1, -1)
        
        for move in get_empty_cells(board):
            # Make the move
            board[move[0], move[1]] = -1
            
            # Compute score for this move (from player 1 perspective)
            move_val = minimax(board, 0, True)
            
            # Undo the move
            board[move[0], move[1]] = 0
            
            # For player -1, we want the minimum score (best for -1)
            if move_val < best_val:
                best_move = move
                best_val = move_val
    
    return best_move

class MinimaxAgent:
    """
    A perfect Tic Tac Toe agent using the minimax algorithm.
    Follows the same interface as the DQNAgent for easy integration.
    """
    
    def __init__(self, player=None):
        """Initialize the agent
        
        Args:
            player: Which player this agent represents (1 for X, -1 for O, None for auto-detect)
        """
        self.player = player
    
    def choose_action(self, state, valid_moves):
        """
        Choose the best action using the minimax algorithm.
        
        Args:
            state: Current board state (3x3 numpy array)
            valid_moves: List of valid (row, col) tuples
            
        Returns:
            The best (row, col) move
        """
        if not valid_moves:
            return None
        
        # Auto-detect player if not set
        if self.player is None:
            # Count X's and O's to determine whose turn it is
            x_count = np.count_nonzero(state == 1)
            o_count = np.count_nonzero(state == -1)
            
            # If equal counts, it's X's turn; if X has one more, it's O's turn
            if x_count == o_count:
                self.player = 1  # X's turn
            else:
                self.player = -1  # O's turn
        
        return find_best_move(state, self.player)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Dummy method to maintain compatibility with DQNAgent interface.
        The minimax agent doesn't need to learn or remember states.
        """
        pass
    
    def replay(self):
        """
        Dummy method to maintain compatibility with DQNAgent interface.
        The minimax agent doesn't need to learn or remember states.
        """
        pass
    
    def save(self, filename):
        """
        Dummy method to maintain compatibility with DQNAgent interface.
        The minimax agent doesn't need to save its state.
        """
        pass
    
    def load(self, filename):
        """
        Dummy method to maintain compatibility with DQNAgent interface.
        The minimax agent doesn't need to load a model.
        """
        pass