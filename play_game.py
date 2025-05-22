#!/usr/bin/env python3
"""
Interactive TicTacToe game against trained Q-Learning agents
"""

import os
from tictactoe import TicTacToe, HumanPlayer, QLearningAgent, play

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_welcome():
    """Print welcome message and instructions"""
    clear_screen()
    print("=" * 50)
    print("🎮 TICTACTOE vs AI 🤖")
    print("=" * 50)
    print("\nThe AI has been trained with Q-Learning!")
    print("Can you beat it?\n")
    print("Board positions:")
    print("-------------")
    print("| 0 | 1 | 2 |")
    print("-------------")
    print("| 3 | 4 | 5 |")
    print("-------------")
    print("| 6 | 7 | 8 |")
    print("-------------")
    print("\nPress Enter to continue...")
    input()

def play_game():
    """Main game loop"""
    print_welcome()
    
    while True:
        clear_screen()
        print("=" * 50)
        print("GAME MENU")
        print("=" * 50)
        print("\n1. Play as X (you go first)")
        print("2. Play as O (AI goes first)")
        print("3. Watch AI vs AI")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            clear_screen()
            print("\n🎮 You are X, AI is O\n")
            game = TicTacToe()
            human = HumanPlayer('X')
            ai = QLearningAgent('O', train_mode=False)
            
            # Load the trained model
            try:
                ai.load_model("diverse_q_table_O.pkl")
            except:
                print("⚠️  Warning: Could not load trained model. AI will play randomly.")
                input("Press Enter to continue...")
            
            winner = play(game, human, ai, print_game=True)
            
            print("\n" + "=" * 30)
            if winner == 'X':
                print("🎉 Congratulations! You won!")
            elif winner == 'O':
                print("🤖 AI wins! Better luck next time!")
            else:
                print("🤝 It's a draw!")
            print("=" * 30)
            
        elif choice == '2':
            clear_screen()
            print("\n🤖 AI is X, You are O\n")
            game = TicTacToe()
            ai = QLearningAgent('X', train_mode=False)
            human = HumanPlayer('O')
            
            # Load the trained model
            try:
                ai.load_model("diverse_q_table_X.pkl")
            except:
                print("⚠️  Warning: Could not load trained model. AI will play randomly.")
                input("Press Enter to continue...")
            
            winner = play(game, ai, human, print_game=True)
            
            print("\n" + "=" * 30)
            if winner == 'O':
                print("🎉 Congratulations! You won!")
            elif winner == 'X':
                print("🤖 AI wins! Better luck next time!")
            else:
                print("🤝 It's a draw!")
            print("=" * 30)
            
        elif choice == '3':
            clear_screen()
            print("\n🤖 AI vs AI 🤖\n")
            game = TicTacToe()
            ai_x = QLearningAgent('X', train_mode=False)
            ai_o = QLearningAgent('O', train_mode=False)
            
            # Load the trained models
            try:
                ai_x.load_model("diverse_q_table_X.pkl")
                ai_o.load_model("diverse_q_table_O.pkl")
            except:
                print("⚠️  Warning: Could not load trained models.")
            
            winner = play(game, ai_x, ai_o, print_game=True)
            
            print("\n" + "=" * 30)
            if winner == 'X':
                print("🤖 AI X wins!")
            elif winner == 'O':
                print("🤖 AI O wins!")
            else:
                print("🤝 It's a draw!")
            print("=" * 30)
            
        elif choice == '4':
            clear_screen()
            print("\nThanks for playing! Goodbye! 👋\n")
            break
            
        else:
            print("\n❌ Invalid choice. Please try again.")
            input("Press Enter to continue...")
            continue
        
        input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye! 👋\n")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please make sure the trained models exist (run tictactoe.py first)")