#!/usr/bin/env python3
"""
Investigation script for the incredible DQN vs Minimax victory
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.agent import DQNAgent
from src.ai.minimax import MinimaxAgent
from src.game.tictactoe import TicTacToe
import numpy as np

def detailed_game_analysis(dqn_agent, minimax_agent, game_num, dqn_first=True):
    """Play a single game with detailed logging"""
    print(f"\n🔍 GAME {game_num} ANALYSIS")
    print("=" * 50)
    
    game = TicTacToe()
    move_history = []
    
    print(f"DQN goes first: {dqn_first}")
    print(f"Starting position:")
    print_board(game.board)
    
    move_num = 1
    while not game.game_over:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break
        
        # Determine current player
        current_is_dqn = (game.current_player == 1 and dqn_first) or (game.current_player == -1 and not dqn_first)
        
        if current_is_dqn:
            # DQN's turn
            action = dqn_agent.choose_action(game.get_state(), valid_moves)
            player_name = "DQN"
            symbol = "X" if game.current_player == 1 else "O"
        else:
            # Minimax's turn
            action = minimax_agent.choose_action(game.get_state(), valid_moves)
            player_name = "Minimax"
            symbol = "X" if game.current_player == 1 else "O"
        
        print(f"\nMove {move_num}: {player_name} ({symbol}) plays {action}")
        
        # Make the move
        next_state, reward, done = game.make_move(action)
        
        # Log the move
        move_history.append({
            'move_num': move_num,
            'player': player_name,
            'symbol': symbol,
            'action': action,
            'board_after': game.board.copy(),
            'reward': reward,
            'done': done
        })
        
        print_board(game.board)
        
        if done:
            result = game.get_result()
            if result == 1:
                winner = "X"
            elif result == -1:
                winner = "O"
            else:
                winner = "Draw"
            
            print(f"\n🏆 Game Result: {winner}")
            
            # Determine if DQN won
            dqn_won = False
            if result != 0:  # Not a draw
                if (result == 1 and dqn_first) or (result == -1 and not dqn_first):
                    dqn_won = True
                    print("🎉 DQN WON!")
            
            return dqn_won, move_history
        
        move_num += 1
    
    return False, move_history

def print_board(board):
    """Print the board in a readable format"""
    print("   0   1   2")
    for i in range(3):
        row_str = f"{i} "
        for j in range(3):
            if board[i, j] == 1:
                row_str += " X "
            elif board[i, j] == -1:
                row_str += " O "
            else:
                row_str += "   "
            if j < 2:
                row_str += "|"
        print(row_str)
        if i < 2:
            print("  ---|---|---")

def verify_minimax_perfection():
    """Test if Minimax is truly perfect"""
    print("🔍 VERIFYING MINIMAX PERFECTION")
    print("=" * 50)
    
    minimax = MinimaxAgent()
    
    # Test some known positions where Minimax should never lose
    test_cases = [
        # Test case 1: Minimax goes first (should never lose)
        {
            'initial_board': np.zeros((3, 3)),
            'minimax_first': True,
            'description': "Minimax goes first from empty board"
        },
        # Test case 2: Critical blocking scenario
        {
            'initial_board': np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]]),
            'minimax_first': False,
            'current_player': -1,  # Minimax must block
            'description': "Minimax must block immediate win"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['description']}")
        
        game = TicTacToe()
        game.board = test_case['initial_board'].copy()
        
        if 'current_player' in test_case:
            game.current_player = test_case['current_player']
        
        print("Initial board:")
        print_board(game.board)
        
        if not game.game_over:
            valid_moves = game.get_valid_moves()
            minimax_action = minimax.choose_action(game.get_state(), valid_moves)
            print(f"Minimax chooses: {minimax_action}")
            
            # Verify this is optimal
            if test_case.get('description') == "Minimax must block immediate win":
                expected_block = (0, 2)  # Should block the winning move
                if minimax_action == expected_block:
                    print("✅ Minimax correctly blocks!")
                else:
                    print(f"❌ Minimax failed to block! Should play {expected_block}")

def run_focused_battle():
    """Run a focused battle to try to reproduce the win"""
    print("\n🎯 FOCUSED BATTLE ANALYSIS")
    print("=" * 50)
    
    # Load the original DQN model
    dqn = DQNAgent(epsilon=0)
    try:
        dqn.load("models/original_dqn_5000ep.pt")
        print("✅ Original DQN loaded")
    except:
        print("❌ Could not load Original DQN")
        return
    
    minimax = MinimaxAgent()
    
    # Run games with detailed logging until we find a DQN win
    dqn_wins = 0
    games_played = 0
    max_games = 1000
    
    print(f"Running up to {max_games} games to find DQN victories...")
    
    for game_num in range(max_games):
        games_played += 1
        dqn_first = (game_num % 2 == 0)
        
        # Quick check without detailed logging
        game = TicTacToe()
        while not game.game_over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            current_is_dqn = (game.current_player == 1 and dqn_first) or (game.current_player == -1 and not dqn_first)
            
            if current_is_dqn:
                action = dqn.choose_action(game.get_state(), valid_moves)
            else:
                action = minimax.choose_action(game.get_state(), valid_moves)
            
            game.make_move(action)
        
        # Check if DQN won
        result = game.get_result()
        if result != 0:  # Not a draw
            if (result == 1 and dqn_first) or (result == -1 and not dqn_first):
                dqn_wins += 1
                print(f"\n🎉 FOUND DQN WIN #{dqn_wins} at game {game_num + 1}!")
                
                # Now replay with detailed analysis
                detailed_game_analysis(dqn, minimax, game_num + 1, dqn_first)
                
                # Stop after finding first win to analyze
                break
        
        # Progress indicator
        if (game_num + 1) % 100 == 0:
            print(f"   Games: {game_num + 1}, DQN wins: {dqn_wins}")
    
    print(f"\n📊 Final Results:")
    print(f"   Games played: {games_played}")
    print(f"   DQN wins found: {dqn_wins}")
    print(f"   Win rate: {dqn_wins/games_played*100:.3f}%")

def main():
    """Main investigation function"""
    print("🕵️ INVESTIGATING THE INCREDIBLE MINIMAX LOSS")
    print("=" * 60)
    print("Analyzing how DQN managed to beat 'perfect' Minimax AI...")
    
    # Step 1: Verify Minimax implementation
    verify_minimax_perfection()
    
    # Step 2: Run focused battle to find and analyze wins
    run_focused_battle()
    
    print("\n🔍 INVESTIGATION COMPLETE")
    print("=" * 60)
    print("Possible explanations for DQN victory:")
    print("1. 🐛 Bug in Minimax implementation")
    print("2. 🎲 Computational error or random glitch")
    print("3. 🤔 Edge case in game logic")
    print("4. 🎯 DQN found an exploit in our implementation")
    print("\nAnalyze the detailed game log above to determine the cause!")

if __name__ == "__main__":
    main()