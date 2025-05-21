import pygame
import numpy as np

class TicTacToe:
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
        
    def make_move(self, position, use_advanced_rewards=False):
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
        
        # Check rows
        for r in range(3):
            if sum(self.board[r, :]) == 3 * self.current_player:
                self.game_over = True
                self.winner = self.current_player
                reward = 10 if self.current_player == 1 else -10  # Higher win reward
                done = True
                break
                
        # Check columns
        if not self.game_over:
            for c in range(3):
                if sum(self.board[:, c]) == 3 * self.current_player:
                    self.game_over = True
                    self.winner = self.current_player
                    reward = 10 if self.current_player == 1 else -10  # Higher win reward
                    done = True
                    break
                    
        # Check diagonals
        if not self.game_over:
            if self.board[0, 0] + self.board[1, 1] + self.board[2, 2] == 3 * self.current_player:
                self.game_over = True
                self.winner = self.current_player
                reward = 10 if self.current_player == 1 else -10  # Higher win reward
                done = True
            elif self.board[0, 2] + self.board[1, 1] + self.board[2, 0] == 3 * self.current_player:
                self.game_over = True
                self.winner = self.current_player
                reward = 10 if self.current_player == 1 else -10  # Higher win reward
                done = True
                
        # Advanced reward shaping for strategic moves
        if use_advanced_rewards and not done:
            reward += self._calculate_strategic_reward(old_board, position, self.current_player)
                
        # Check for draw
        if not self.game_over and len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0  # No winner
            reward = 1  # Better draw reward - draws are good defensive play
            done = True
            
        # Switch player
        if not self.game_over:
            self.current_player *= -1
            
        return self.get_state(), reward, done
    
    def _calculate_strategic_reward(self, old_board, position, player):
        """Calculate strategic reward for moves that create opportunities or block threats"""
        reward = 0
        i, j = position
        
        # Reward center control (most strategic position)
        if i == 1 and j == 1:
            reward += 0.5
        
        # Reward corner control
        if (i, j) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            reward += 0.3
            
        # Check if move creates a fork (two ways to win)
        fork_opportunities = self._count_fork_opportunities(self.board, player)
        if fork_opportunities >= 2:
            reward += 2.0  # Strong reward for creating forks
        elif fork_opportunities == 1:
            reward += 0.5
            
        # Check if move blocks opponent's win
        if self._blocks_immediate_win(old_board, position, -player):
            reward += 1.5  # Good reward for defensive play
            
        # Check if move creates immediate win threat
        if self._creates_win_threat(self.board, player):
            reward += 1.0  # Reward aggressive play
            
        return reward
    
    def _count_fork_opportunities(self, board, player):
        """Count number of ways the player can win in one move"""
        win_opportunities = 0
        
        # Check rows
        for r in range(3):
            if list(board[r, :]).count(player) == 2 and list(board[r, :]).count(0) == 1:
                win_opportunities += 1
                
        # Check columns  
        for c in range(3):
            if list(board[:, c]).count(player) == 2 and list(board[:, c]).count(0) == 1:
                win_opportunities += 1
                
        # Check diagonals
        diag1 = [board[i, i] for i in range(3)]
        if diag1.count(player) == 2 and diag1.count(0) == 1:
            win_opportunities += 1
            
        diag2 = [board[i, 2-i] for i in range(3)]
        if diag2.count(player) == 2 and diag2.count(0) == 1:
            win_opportunities += 1
            
        return win_opportunities
    
    def _blocks_immediate_win(self, old_board, position, opponent):
        """Check if the move blocks opponent's immediate win"""
        # Check if opponent would win by playing at this position
        test_board = old_board.copy()
        test_board[position[0], position[1]] = opponent
        
        # Check for immediate win (3 in a row)
        i, j = position
        
        # Check row
        if sum(test_board[i, :]) == 3 * opponent:
            return True
        
        # Check column  
        if sum(test_board[:, j]) == 3 * opponent:
            return True
            
        # Check diagonals
        if i == j and sum([test_board[k, k] for k in range(3)]) == 3 * opponent:
            return True
            
        if i + j == 2 and sum([test_board[k, 2-k] for k in range(3)]) == 3 * opponent:
            return True
            
        return False
    
    def _creates_win_threat(self, board, player):
        """Check if current board position creates a win threat for player"""
        return self._count_fork_opportunities(board, player) > 0
        
    def get_result(self):
        return self.winner


class TicTacToeGUI:
    def __init__(self, width=600, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Tic Tac Toe")
        self.game = TicTacToe()
        self.cell_size = width // 3
        self.line_width = 15
        self.mark_width = 20
        self.margin = 50
        
    def draw_board(self):
        self.screen.fill((255, 255, 255))
        
        # Draw lines
        for i in range(1, 3):
            # Vertical lines
            pygame.draw.line(
                self.screen, 
                (0, 0, 0), 
                (i * self.cell_size, 0), 
                (i * self.cell_size, self.height), 
                self.line_width
            )
            
            # Horizontal lines
            pygame.draw.line(
                self.screen, 
                (0, 0, 0), 
                (0, i * self.cell_size), 
                (self.width, i * self.cell_size), 
                self.line_width
            )
            
        # Draw X and O
        for i in range(3):
            for j in range(3):
                center_x = j * self.cell_size + self.cell_size // 2
                center_y = i * self.cell_size + self.cell_size // 2
                
                if self.game.board[i, j] == 1:  # X
                    pygame.draw.line(
                        self.screen, 
                        (255, 0, 0), 
                        (center_x - self.cell_size // 2 + self.margin, center_y - self.cell_size // 2 + self.margin), 
                        (center_x + self.cell_size // 2 - self.margin, center_y + self.cell_size // 2 - self.margin), 
                        self.mark_width
                    )
                    pygame.draw.line(
                        self.screen, 
                        (255, 0, 0), 
                        (center_x - self.cell_size // 2 + self.margin, center_y + self.cell_size // 2 - self.margin), 
                        (center_x + self.cell_size // 2 - self.margin, center_y - self.cell_size // 2 + self.margin), 
                        self.mark_width
                    )
                elif self.game.board[i, j] == -1:  # O
                    pygame.draw.circle(
                        self.screen, 
                        (0, 0, 255), 
                        (center_x, center_y), 
                        self.cell_size // 2 - self.margin, 
                        self.mark_width
                    )
                    
    def handle_click(self, pos):
        if self.game.game_over:
            self.game.reset()
            return
            
        x, y = pos
        j = x // self.cell_size
        i = y // self.cell_size
        
        if 0 <= i < 3 and 0 <= j < 3:
            self.game.make_move((i, j))
            
    def display_winner(self):
        if not self.game.game_over:
            return
            
        font = pygame.font.Font(None, 30)
        
        if self.game.winner == 1:
            text = font.render("X Wins! Click to play again.", True, (0, 0, 0))
        elif self.game.winner == -1:
            text = font.render("O Wins! Click to play again.", True, (0, 0, 0))
        else:
            text = font.render("It's a draw! Click to play again.", True, (0, 0, 0))
            
        text_rect = text.get_rect(center=(self.width // 2, self.height - 30))
        self.screen.blit(text, text_rect)
        
    def run_human_vs_human(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                    
            self.draw_board()
            self.display_winner()
            pygame.display.flip()
            
        pygame.quit()