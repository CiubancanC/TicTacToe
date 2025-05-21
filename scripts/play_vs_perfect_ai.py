"""
Play Tic Tac Toe against a perfect AI opponent using the minimax algorithm.
This AI will never lose and will always win when possible.
"""

import sys
sys.path.append('.')
import pygame
import numpy as np
import time

from src.game.tictactoe import TicTacToe
from src.ai.minimax import MinimaxAgent

def human_vs_perfect_ai():
    """Run a game of Tic Tac Toe with a human player against a minimax AI"""
    pygame.init()
    width, height = 600, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Tic Tac Toe - Human vs Perfect AI")
    
    game = TicTacToe()
    agent = MinimaxAgent()
    
    cell_size = width // 3
    line_width = 15
    mark_width = 20
    margin = 50
    
    def draw_board():
        screen.fill((255, 255, 255))
        
        # Draw grid lines
        for i in range(1, 3):
            # Vertical lines
            pygame.draw.line(screen, (0, 0, 0), (i * cell_size, 0), (i * cell_size, height), line_width)
            # Horizontal lines
            pygame.draw.line(screen, (0, 0, 0), (0, i * cell_size), (width, i * cell_size), line_width)
            
        # Draw X and O
        for i in range(3):
            for j in range(3):
                center_x = j * cell_size + cell_size // 2
                center_y = i * cell_size + cell_size // 2
                
                if game.board[i, j] == 1:  # X (Human)
                    pygame.draw.line(
                        screen, 
                        (255, 0, 0), 
                        (center_x - cell_size // 2 + margin, center_y - cell_size // 2 + margin), 
                        (center_x + cell_size // 2 - margin, center_y + cell_size // 2 - margin), 
                        mark_width
                    )
                    pygame.draw.line(
                        screen, 
                        (255, 0, 0), 
                        (center_x - cell_size // 2 + margin, center_y + cell_size // 2 - margin), 
                        (center_x + cell_size // 2 - margin, center_y - cell_size // 2 + margin), 
                        mark_width
                    )
                elif game.board[i, j] == -1:  # O (AI)
                    pygame.draw.circle(
                        screen, 
                        (0, 0, 255), 
                        (center_x, center_y), 
                        cell_size // 2 - margin, 
                        mark_width
                    )
    
    def display_message(message):
        font = pygame.font.Font(None, 30)
        text = font.render(message, True, (0, 0, 0))
        text_rect = text.get_rect(center=(width // 2, height - 30))
        screen.blit(text, text_rect)
    
    running = True
    game_state = "playing"  # "playing", "game_over"
    
    # Human goes first
    human_turn = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_state == "game_over":
                    game.reset()
                    game_state = "playing"
                    human_turn = True
                elif human_turn:
                    # Human's turn
                    x, y = event.pos
                    j = x // cell_size
                    i = y // cell_size
                    
                    if 0 <= i < 3 and 0 <= j < 3 and game.board[i, j] == 0:
                        next_state, reward, done = game.make_move((i, j))
                        
                        if done:
                            game_state = "game_over"
                        else:
                            human_turn = False
        
        # AI's turn
        if not human_turn and game_state == "playing":
            time.sleep(0.5)  # Add a slight delay for better UX
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            
            if valid_moves:
                ai_action = agent.choose_action(state, valid_moves)
                next_state, reward, done = game.make_move(ai_action)
                
                if done:
                    game_state = "game_over"
                else:
                    human_turn = True
            else:
                game_state = "game_over"
        
        # Drawing
        draw_board()
        
        if game_state == "playing":
            if human_turn:
                display_message("Your turn (X)")
            else:
                display_message("AI thinking... (O)")
        else:
            result = game.get_result()
            if result == 1:
                display_message("You win! Click to play again.")
            elif result == -1:
                display_message("AI wins! Click to play again.")
            else:
                display_message("It's a draw! Click to play again.")
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    print("Starting game against the perfect AI agent...")
    print("You will play as X (first player)")
    print("The AI will play as O (second player)")
    print("This AI uses the minimax algorithm and will never lose!")
    print("Click on the board to make your move")
    
    human_vs_perfect_ai()