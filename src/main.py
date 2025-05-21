import pygame
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os
import sys

# Fix imports
from src.game.tictactoe import TicTacToe, TicTacToeGUI
from src.ai.agent import DQNAgent

def train_agent(num_episodes=5000, save_interval=500):
    agent = DQNAgent(epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.997, 
                    gamma=0.99, lr=0.0005, batch_size=128)
    game = TicTacToe()
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    rewards_history = []
    win_history = []
    draw_history = []
    loss_history = []
    archive_interval = 1000  # Archive model every 1000 episodes
    
    for episode in range(num_episodes):
        state = game.reset()
        total_reward = 0
        done = False
        use_historical_opponent = agent.should_use_historical_opponent(episode)
        
        while not done:
            valid_moves = game.get_valid_moves()
            action = agent.choose_action(state, valid_moves)
            
            if action is None:
                break
                
            # Use advanced rewards for better learning
            next_state, reward, done = game.make_move(action, use_advanced_rewards=True)
            total_reward += reward
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            
            # Opponent's turn - use historical opponent or self-play
            if not done:
                valid_moves = game.get_valid_moves()
                
                if use_historical_opponent and agent.historical_opponents:
                    # Play against historical version
                    opponent_action = agent.get_historical_opponent_action(state, valid_moves)
                else:
                    # Self-play
                    opponent_action = agent.choose_action(state, valid_moves)
                
                if opponent_action is None:
                    break
                    
                next_state, reward, done = game.make_move(opponent_action, use_advanced_rewards=True)
                
                # Store opponent experience with negated reward (from agent's perspective)
                if not use_historical_opponent:
                    agent.remember(state, opponent_action, -reward, next_state, done)
                
                state = next_state
                
        # Record game outcome
        rewards_history.append(total_reward)
        result = game.get_result()
        
        if result == 1:
            win_history.append(1)
            draw_history.append(0)
            loss_history.append(0)
        elif result == -1:
            win_history.append(0)
            draw_history.append(0)
            loss_history.append(1)
        else:
            win_history.append(0)
            draw_history.append(1)
            loss_history.append(0)
            
        # Archive model periodically for historical opponents
        if (episode + 1) % archive_interval == 0:
            agent.archive_current_model()
        
        # Print progress
        if (episode + 1) % 100 == 0:
            win_rate = sum(win_history[-100:])
            draw_rate = sum(draw_history[-100:])
            loss_rate = sum(loss_history[-100:])
            historical_info = f", Historical Opponents: {len(agent.historical_opponents)}" if agent.historical_opponents else ""
            print(f"Episode: {episode + 1}, Win: {win_rate}%, Draw: {draw_rate}%, Loss: {loss_rate}%, Epsilon: {agent.epsilon:.4f}{historical_info}")
            
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            agent.save(f"models/dqn_agent_episode_{episode + 1}.pt")
            
    # Save final model
    agent.save("models/dqn_agent_final.pt")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Reward History')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    window_size = 100
    win_rate = [sum(win_history[i:i+window_size])/window_size for i in range(0, len(win_history), window_size)]
    draw_rate = [sum(draw_history[i:i+window_size])/window_size for i in range(0, len(draw_history), window_size)]
    loss_rate = [sum(loss_history[i:i+window_size])/window_size for i in range(0, len(loss_history), window_size)]
    
    plt.plot(range(0, len(win_history), window_size), win_rate, label='Win')
    plt.plot(range(0, len(draw_history), window_size), draw_rate, label='Draw')
    plt.plot(range(0, len(loss_history), window_size), loss_rate, label='Loss')
    plt.title('Game Outcome Rates')
    plt.xlabel('Episode')
    plt.ylabel('Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return agent

def human_vs_ai(model_path="models/dqn_agent_final.pt"):
    pygame.init()
    width, height = 600, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Tic Tac Toe - Human vs AI")
    
    game = TicTacToe()
    agent = DQNAgent(epsilon=0)  # No exploration during gameplay
    
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print(f"Could not load model from {model_path}, using untrained agent")
    
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
            
            if event.type == pygame.MOUSEBUTTONDOWN and human_turn and game_state == "playing":
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
                
            # Wait for click to restart
            if pygame.mouse.get_pressed()[0]:
                game.reset()
                game_state = "playing"
                human_turn = True
                time.sleep(0.3)  # Prevent double-clicks
        
        pygame.display.flip()
    
    pygame.quit()

def main():
    print("Tic Tac Toe with Reinforcement Learning")
    print("1. Train AI agent")
    print("2. Play against AI")
    print("3. Human vs Human")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        print("Training AI agent...")
        agent = train_agent()
        print("Training complete!")
        
        play_now = input("Do you want to play against the trained AI? (y/n): ")
        if play_now.lower() == 'y':
            human_vs_ai()
    
    elif choice == '2':
        model_path = "models/dqn_agent_final.pt"
        if os.path.exists(model_path):
            human_vs_ai(model_path)
        else:
            print("No trained model found. Train the model first or play without training.")
            choice = input("Do you want to train the model first? (y/n): ")
            if choice.lower() == 'y':
                agent = train_agent()
                human_vs_ai()
            else:
                human_vs_ai("no_model")
    
    elif choice == '3':
        gui = TicTacToeGUI()
        gui.run_human_vs_human()
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()