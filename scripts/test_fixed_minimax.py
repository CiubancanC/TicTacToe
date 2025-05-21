#!/usr/bin/env python3
"""
Test the fixed Minimax implementation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.ai.minimax import MinimaxAgent, find_best_move

def print_board(board):
    """Print board in readable format"""
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

def test_fixed_minimax():
    """Test the fixed Minimax implementation"""
    print("🔧 TESTING FIXED MINIMAX")
    print("=" * 40)
    
    # Test 1: O should take the win
    print("\n🧪 Test 1: O should take immediate win")
    board = np.array([
        [1, 1, 0],   # X X _
        [-1, -1, 0], # O O _  <- O should win here
        [0, 0, 0]    # _ _ _
    ])
    print("Board state (O should win at (1,2)):")
    print_board(board)
    
    # Test with explicit player assignment
    best_move = find_best_move(board, player=-1)
    print(f"Fixed Minimax (as O) chooses: {best_move}")
    if best_move == (1, 2):
        print("✅ SUCCESS: O takes the win!")
    else:
        print("❌ STILL BROKEN: O didn't take the win")
    
    # Test 2: X should take the win  
    print("\n🧪 Test 2: X should take immediate win")
    board = np.array([
        [1, 1, 0],   # X X _  <- X should win here
        [-1, -1, 0], # O O _
        [0, 0, 0]    # _ _ _
    ])
    print("Board state (X should win at (0,2)):")
    print_board(board)
    
    best_move = find_best_move(board, player=1)
    print(f"Fixed Minimax (as X) chooses: {best_move}")
    if best_move == (0, 2):
        print("✅ SUCCESS: X takes the win!")
    else:
        print("❌ PROBLEM: X didn't take the win")
        
    # Test 3: O should block X's win
    print("\n🧪 Test 3: O should block X's win")
    board = np.array([
        [1, 1, 0],   # X X _  <- O must block here
        [0, -1, 0],  # _ O _
        [0, 0, 0]    # _ _ _
    ])
    print("Board state (O should block at (0,2)):")
    print_board(board)
    
    best_move = find_best_move(board, player=-1)
    print(f"Fixed Minimax (as O) chooses: {best_move}")
    if best_move == (0, 2):
        print("✅ SUCCESS: O blocks X's win!")
    else:
        print("❌ PROBLEM: O didn't block")

def test_auto_detection():
    """Test the auto-detection feature"""
    print("\n🤖 TESTING AUTO-DETECTION")
    print("=" * 40)
    
    # Test game where it's clearly O's turn
    board = np.array([
        [1, 0, 0],   # X _ _
        [0, 0, 0],   # _ _ _
        [0, 0, 0]    # _ _ _
    ])
    
    print("Board with 1 X, 0 O's (should be O's turn):")
    print_board(board)
    
    agent = MinimaxAgent()  # No player specified, should auto-detect
    valid_moves = [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]
    
    action = agent.choose_action(board, valid_moves)
    print(f"Auto-detected agent chooses: {action}")
    print(f"Agent detected it's playing as: {agent.player}")
    
    if agent.player == -1:
        print("✅ SUCCESS: Correctly detected it's O's turn")
    else:
        print("❌ PROBLEM: Auto-detection failed")

def run_corrected_battle():
    """Test a quick battle with the fixed Minimax"""
    print("\n⚔️ CORRECTED BATTLE TEST")
    print("=" * 40)
    
    from src.ai.agent import DQNAgent
    from src.game.tictactoe import TicTacToe
    
    # Load DQN
    dqn = DQNAgent(epsilon=0)
    try:
        dqn.load("models/original_dqn_5000ep.pt")
        print("✅ DQN loaded")
    except:
        print("❌ Could not load DQN, skipping battle test")
        return
    
    # Test 10 games with fixed Minimax
    minimax = MinimaxAgent()
    wins = draws = losses = 0
    
    for game_num in range(10):
        game = TicTacToe() 
        minimax.player = None  # Reset for auto-detection each game
        dqn_first = (game_num % 2 == 0)
        
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
        
        result = game.get_result()
        if result == 0:
            draws += 1
        elif (result == 1 and dqn_first) or (result == -1 and not dqn_first):
            losses += 1  # DQN won
        else:
            wins += 1  # Minimax won
    
    print(f"Fixed Minimax vs DQN (10 games):")
    print(f"  Minimax wins: {wins}")
    print(f"  Draws: {draws}")  
    print(f"  DQN wins: {losses}")
    
    if losses == 0:
        print("✅ SUCCESS: Fixed Minimax is now truly unbeatable!")
    else:
        print(f"⚠️ DQN still won {losses} games - may need more investigation")

def main():
    """Run all tests"""
    print("🛠️ FIXED MINIMAX VERIFICATION")
    print("=" * 60)
    
    test_fixed_minimax()
    test_auto_detection()
    run_corrected_battle()
    
    print("\n🎯 SUMMARY:")
    print("=" * 40)
    print("The bug was that Minimax always optimized for player 1 (X)")
    print("even when it was supposed to be playing as player -1 (O).")
    print("\nFixed by:")
    print("1. Making find_best_move() player-aware")
    print("2. Adding auto-detection to determine current player")
    print("3. Proper optimization for both X and O")
    
    print(f"\nNow Minimax should be truly unbeatable! 🏆")

if __name__ == "__main__":
    main()