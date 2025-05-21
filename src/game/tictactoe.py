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
        
    def make_move(self, position):
        if self.game_over:
            return self.get_state(), 0, True
            
        i, j = position
        if self.board[i, j] != 0:
            return self.get_state(), -10, False  # Invalid move penalty
            
        self.board[i, j] = self.current_player
        
        # Check for win or draw
        reward = 0
        done = False
        
        # Check rows
        for i in range(3):
            if sum(self.board[i, :]) == 3 * self.current_player:
                self.game_over = True
                self.winner = self.current_player
                reward = 1 if self.current_player == 1 else -1
                done = True
                break
                
        # Check columns
        if not self.game_over:
            for j in range(3):
                if sum(self.board[:, j]) == 3 * self.current_player:
                    self.game_over = True
                    self.winner = self.current_player
                    reward = 1 if self.current_player == 1 else -1
                    done = True
                    break
                    
        # Check diagonals
        if not self.game_over:
            if self.board[0, 0] + self.board[1, 1] + self.board[2, 2] == 3 * self.current_player:
                self.game_over = True
                self.winner = self.current_player
                reward = 1 if self.current_player == 1 else -1
                done = True
            elif self.board[0, 2] + self.board[1, 1] + self.board[2, 0] == 3 * self.current_player:
                self.game_over = True
                self.winner = self.current_player
                reward = 1 if self.current_player == 1 else -1
                done = True
                
        # Check for draw
        if not self.game_over and len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0  # No winner
            reward = 0.5  # Small reward for draw
            done = True
            
        # Switch player
        if not self.game_over:
            self.current_player *= -1
            
        return self.get_state(), reward, done
        
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