#!/usr/bin/env python3
"""
Quick test to identify the Minimax vulnerability
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.ai.minimax import MinimaxAgent, evaluate_board, find_best_move

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

def test_minimax_basic():
    """Test basic Minimax functionality"""
    print("🔍 BASIC MINIMAX TESTS")
    print("=" * 40)
    
    # Test 1: Minimax should block immediate loss
    print("\n🧪 Test 1: Blocking immediate loss")
    board = np.array([
        [1, 1, 0],  # X X _
        [0, 0, 0],  # _ _ _
        [0, 0, 0]   # _ _ _
    ])
    print("Board state (Minimax is O, should block at (0,2)):")
    print_board(board)
    
    # Minimax should choose (0, 2) to block
    best_move = find_best_move(board)
    print(f"Minimax chooses: {best_move}")
    if best_move == (0, 2):
        print("✅ Correct blocking move!")
    else:
        print("❌ Failed to block - this is the bug!")
    
    # Test 2: Minimax should take immediate win
    print("\n🧪 Test 2: Taking immediate win")
    board = np.array([
        [1, 1, 0],  # X X _
        [-1, -1, 0], # O O _  
        [0, 0, 0]   # _ _ _
    ])
    print("Board state (Minimax is O, should win at (1,2)):")
    print_board(board)
    
    best_move = find_best_move(board)
    print(f"Minimax chooses: {best_move}")
    if best_move == (1, 2):
        print("✅ Correct winning move!")
    else:
        print("❌ Failed to win!")

def test_player_perspective():
    """Test if there's a player perspective issue"""
    print("\n🔍 PLAYER PERSPECTIVE TEST")
    print("=" * 40)
    
    minimax = MinimaxAgent()
    
    # Create a board where it's clear who should win
    board = np.array([
        [1, 0, 0],   # X _ _
        [0, -1, 0],  # _ O _
        [0, 0, 0]    # _ _ _
    ])
    
    print("Current board:")
    print_board(board)
    print("If it's Minimax's turn as O (-1), what should it do?")
    
    # Test what Minimax chooses
    valid_moves = [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]
    action = minimax.choose_action(board, valid_moves)
    print(f"Minimax chooses: {action}")
    
    # The issue might be: Minimax algorithm assumes it's always player 1 (X)
    # But in our game, Minimax might be player -1 (O)
    print("\n🚨 POTENTIAL BUG IDENTIFIED:")
    print("The Minimax implementation always assumes it's playing as X (value 1)")
    print("But in our game system, Minimax might be assigned to play as O (value -1)")
    print("This would cause it to make moves that benefit the opponent!")

def test_game_integration():
    """Test how Minimax integrates with our game system"""
    print("\n🔍 GAME INTEGRATION TEST") 
    print("=" * 40)
    
    from src.game.tictactoe import TicTacToe
    
    game = TicTacToe()
    minimax = MinimaxAgent()
    
    print("Starting new game...")
    print("Game's current_player:", game.current_player)  # Should be 1 (X)
    
    # Let's see what happens when Minimax plays as the first player
    print("\nMinimax plays as X (player 1):")
    action = minimax.choose_action(game.get_state(), game.get_valid_moves())
    print(f"Minimax chooses: {action}")
    game.make_move(action)
    print_board(game.board)
    
    print(f"Next player: {game.current_player}")  # Should be -1 (O)
    
    # Now what if Minimax plays as the second player?
    print("\nNow if Minimax were to play as O (player -1):")
    print("This is where the bug likely occurs!")
    
    # The Minimax algorithm always optimizes for player 1 (value 1)
    # But when it's assigned to be player -1, it's actually helping player 1 win!

def main():
    """Run all tests to identify the bug"""
    print("🕵️ MINIMAX BUG INVESTIGATION")
    print("=" * 60)
    
    test_minimax_basic()
    test_player_perspective() 
    test_game_integration()
    
    print("\n🎯 CONCLUSION:")
    print("=" * 40)
    print("🐛 LIKELY BUG FOUND:")
    print("1. Minimax algorithm always optimizes for player 1 (X)")
    print("2. When Minimax is assigned player -1 (O), it still optimizes for X")
    print("3. This means Minimax as O actually helps DQN as X win!")
    print("4. That's how DQN 'beat' the 'perfect' AI!")
    print("\n💡 FIX: Minimax needs to know which player it's representing")
    print("   and optimize for that player, not always for player 1")

if __name__ == "__main__":
    main()