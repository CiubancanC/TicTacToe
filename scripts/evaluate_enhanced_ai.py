#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.agent import DQNAgent
from src.ai.minimax import MinimaxAgent
from src.game.tictactoe import TicTacToe
import numpy as np

def test_strategic_play():
    """Test if the enhanced AI demonstrates strategic play"""
    print("🧪 Testing Enhanced AI Strategic Capabilities")
    print("=" * 50)
    
    # Load enhanced model
    agent = DQNAgent(epsilon=0)  # No exploration for testing
    try:
        agent.load('models/dqn_agent_final.pt')
        print("✅ Enhanced AI model loaded successfully")
    except:
        print("❌ Could not load enhanced model")
        return
    
    # Test 1: Center preference
    print("\n🎯 Test 1: Center Control Preference")
    game = TicTacToe()
    state = game.get_state()
    valid_moves = game.get_valid_moves()
    
    action = agent.choose_action(state, valid_moves)
    if action == (1, 1):  # Center position
        print("✅ AI correctly chooses center on first move")
    else:
        print(f"⚠️  AI chose {action} instead of center (1,1)")
    
    # Test 2: Blocking behavior
    print("\n🛡️  Test 2: Defensive Blocking")
    game.reset()
    # Set up a scenario where opponent is about to win
    game.board[0, 0] = -1  # O
    game.board[0, 1] = -1  # O
    game.current_player = 1  # X's turn (AI)
    
    state = game.get_state()
    valid_moves = game.get_valid_moves()
    action = agent.choose_action(state, valid_moves)
    
    if action == (0, 2):  # Block the win
        print("✅ AI correctly blocks opponent's winning move")
    else:
        print(f"⚠️  AI chose {action}, should block at (0,2)")
    
    # Test 3: Winning move detection
    print("\n🏆 Test 3: Winning Move Recognition")
    game.reset()
    # Set up a scenario where AI can win
    game.board[1, 1] = 1   # X (center)
    game.board[0, 0] = 1   # X (top-left)
    game.current_player = 1  # X's turn (AI)
    
    state = game.get_state()
    valid_moves = game.get_valid_moves()
    action = agent.choose_action(state, valid_moves)
    
    if action == (2, 2):  # Complete the diagonal
        print("✅ AI correctly identifies winning move")
    else:
        print(f"⚠️  AI chose {action}, should win at (2,2)")
    
    return agent

def benchmark_vs_minimax(num_games=100):
    """Benchmark enhanced AI against perfect Minimax"""
    print(f"\n⚔️  Benchmark: Enhanced DQN vs Perfect Minimax ({num_games} games)")
    print("-" * 60)
    
    dqn_agent = DQNAgent(epsilon=0)
    try:
        dqn_agent.load('models/dqn_agent_final.pt')
    except:
        print("❌ Could not load DQN model")
        return
    
    minimax_agent = MinimaxAgent()
    
    results = {'dqn_wins': 0, 'draws': 0, 'minimax_wins': 0}
    
    for game_num in range(num_games):
        game = TicTacToe()
        
        # Alternate who goes first
        dqn_goes_first = (game_num % 2 == 0)
        
        while not game.game_over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            if (game.current_player == 1 and dqn_goes_first) or (game.current_player == -1 and not dqn_goes_first):
                # DQN's turn
                action = dqn_agent.choose_action(game.get_state(), valid_moves)
            else:
                # Minimax's turn
                action = minimax_agent.choose_action(game.get_state(), valid_moves)
            
            game.make_move(action)
        
        # Record result from DQN's perspective
        result = game.get_result()
        if result == 0:
            results['draws'] += 1
        elif (result == 1 and dqn_goes_first) or (result == -1 and not dqn_goes_first):
            results['dqn_wins'] += 1
        else:
            results['minimax_wins'] += 1
    
    # Display results
    print(f"🤖 DQN Wins: {results['dqn_wins']} ({results['dqn_wins']/num_games*100:.1f}%)")
    print(f"🤝 Draws: {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    print(f"🎯 Minimax Wins: {results['minimax_wins']} ({results['minimax_wins']/num_games*100:.1f}%)")
    
    # Performance assessment
    non_loss_rate = (results['dqn_wins'] + results['draws']) / num_games * 100
    print(f"\n📊 DQN Non-Loss Rate: {non_loss_rate:.1f}%")
    
    if non_loss_rate >= 95:
        print("🏆 EXCELLENT: Near-perfect play against optimal opponent!")
    elif non_loss_rate >= 85:
        print("✅ GOOD: Strong defensive play with occasional wins")
    elif non_loss_rate >= 70:
        print("⚠️  FAIR: Decent play but room for improvement")
    else:
        print("❌ POOR: Significant learning needed")

def main():
    """Run all evaluations"""
    print("🎮 Enhanced DQN AI Evaluation Suite")
    print("=" * 50)
    
    # Test strategic capabilities
    agent = test_strategic_play()
    
    if agent:
        # Benchmark against minimax
        benchmark_vs_minimax(50)  # Quick benchmark
        
        print("\n🎯 Evaluation Complete!")
        print("\n💡 To play against the enhanced AI:")
        print("   python3 scripts/play_vs_ai.py")
        print("   python3 run_game.py (option 4)")

if __name__ == "__main__":
    main()