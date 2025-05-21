import sys
sys.path.append('.')
from src.main import train_agent, human_vs_ai

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_game.py [mode]")
        print("Modes:")
        print("  1 - Train AI agent")
        print("  2 - Play against DQN AI")
        print("  3 - Human vs Human")
        print("  4 - Play against Perfect AI (Minimax)")
        sys.exit(1)
        
    mode = sys.argv[1]
    
    if mode == "1":
        print("Training AI agent with improved architecture...")
        print("This will train the agent with Double DQN and Prioritized Experience Replay")
        train_agent(num_episodes=10000)  # Using our improved agent
        print("Training complete!")
    elif mode == "2":
        print("Starting Human vs DQN AI game...")
        human_vs_ai("models/dqn_agent_final.pt")  # Using our improved trained AI
    elif mode == "3":
        print("Starting Human vs Human game...")
        # Import TicTacToeGUI and run the human vs human game
        from src.game.tictactoe import TicTacToeGUI
        gui = TicTacToeGUI()
        gui.run_human_vs_human()
    elif mode == "4":
        print("Starting Human vs Perfect AI game (Minimax algorithm)...")
        print("This AI uses the minimax algorithm with alpha-beta pruning.")
        print("It will never lose and will always win when possible!")
        # Import the perfect AI game
        from src.ai.minimax import MinimaxAgent
        import pygame
        import time
        from src.game.tictactoe import TicTacToe
        
        # Setup the game
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
    else:
        print(f"Invalid mode: {mode}")