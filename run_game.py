#!/usr/bin/env python3
import sys
import os
import glob
import subprocess
import tempfile
sys.path.append('.')
from src.main import train_agent, human_vs_ai

def display_menu():
    """Display the main menu with all available options"""
    print("\n" + "="*60)
    print("🎮 TIC TAC TOE - AI SHOWCASE")
    print("="*60)
    print()
    print("🤖 TRAINING OPTIONS:")
    print("  1 - Train Original DQN AI (standard self-play)")
    print("  2 - Train Enhanced DQN AI (strategic rewards + archives)")
    print("  3 - Quick test Enhanced features (500 episodes)")
    print()
    print("🎯 PLAY AGAINST AI:")
    print("  4 - Play vs DQN AI (select trained model)")
    print("  5 - Play vs Perfect Minimax AI (unbeatable)")
    print()
    print("👥 HUMAN MODES:")
    print("  6 - Human vs Human")
    print("  7 - Compare AI Models (performance stats)")
    print()
    print("  0 - Exit")
    print("="*60)

def list_available_models():
    """List all available trained models"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    models = []
    # Look for final models first
    if os.path.exists(f"{models_dir}/dqn_agent_final.pt"):
        models.append(("Final Model", f"{models_dir}/dqn_agent_final.pt"))
    
    # Look for episode checkpoints
    episode_models = glob.glob(f"{models_dir}/dqn_agent_episode_*.pt")
    episode_models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    
    for model_path in episode_models[:5]:  # Show top 5 most recent
        episode_num = model_path.split('_')[-1].split('.')[0]
        models.append((f"Episode {episode_num}", model_path))
    
    return models

def select_model():
    """Allow user to select which model to play against"""
    models = list_available_models()
    
    if not models:
        print("❌ No trained models found!")
        print("🔧 Train a model first using options 1, 2, or 3")
        return None
    
    print("\n📁 Available Trained Models:")
    print("-" * 40)
    for i, (name, path) in enumerate(models, 1):
        file_size = os.path.getsize(path) / 1024  # KB
        print(f"  {i}. {name} ({file_size:.1f} KB)")
    
    try:
        choice = int(input("\nSelect model (number): ")) - 1
        if 0 <= choice < len(models):
            return models[choice][1]
        else:
            print("❌ Invalid selection")
            return None
    except ValueError:
        print("❌ Please enter a valid number")
        return None

def train_original_dqn():
    """Train original DQN with standard self-play"""
    print("\n🔬 Training Original DQN AI")
    print("Features: Double DQN, Prioritized Replay, Self-play")
    episodes = int(input("Enter number of episodes (default 5000): ") or "5000")
    
    print(f"\n🚀 Starting original DQN training for {episodes} episodes...")
    agent = train_agent(num_episodes=episodes)
    print("✅ Original DQN training complete!")
    return agent

def train_enhanced_dqn():
    """Train enhanced DQN with strategic rewards and archives"""
    print("\n🎯 Training Enhanced DQN AI")
    print("Features: Strategic rewards, Historical opponents, Advanced self-play")
    episodes = int(input("Enter number of episodes (default 10000): ") or "10000")
    
    print(f"\n🚀 Starting enhanced DQN training for {episodes} episodes...")
    print("📊 This includes strategic move bonuses and historical opponent archives")
    
    # Enhanced training using imported modules
    
    # Create temporary enhanced training script
    enhanced_script = f"""
import sys
sys.path.append('.')
from src.main import train_agent
import time

print("🤖 Enhanced DQN Training Started")
start_time = time.time()
agent = train_agent(num_episodes={episodes}, save_interval=max(500, {episodes}//10))
end_time = time.time()
print(f"✅ Enhanced training completed in {{(end_time-start_time)/60:.1f}} minutes")
print(f"📊 Historical opponents: {{len(agent.historical_opponents)}}")
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(enhanced_script)
        temp_script = f.name
    
    try:
        subprocess.run([sys.executable, temp_script], check=True)
        print("✅ Enhanced DQN training complete!")
    finally:
        os.unlink(temp_script)

def quick_test_enhanced():
    """Quick test of enhanced features"""
    print("\n🧪 Quick Test of Enhanced Features")
    print("Running 500 episodes to demonstrate enhanced training...")
    
    try:
        subprocess.run([sys.executable, "scripts/test_enhanced_features.py"], check=True)
    except FileNotFoundError:
        print("❌ Test script not found. Running inline test...")
        agent = train_agent(num_episodes=500, save_interval=200)
        print(f"✅ Quick test complete! Historical opponents: {len(agent.historical_opponents)}")

def play_vs_dqn():
    """Play against trained DQN AI"""
    model_path = select_model()
    if model_path:
        print(f"\n🎮 Starting game with {os.path.basename(model_path)}...")
        human_vs_ai(model_path)

def play_vs_minimax():
    """Play against perfect Minimax AI"""
    print("\n⚔️  Playing against Perfect Minimax AI")
    print("⚠️  Warning: This AI never loses and always wins when possible!")
    
    try:
        subprocess.run([sys.executable, "scripts/play_vs_perfect_ai.py"], check=True)
    except FileNotFoundError:
        print("❌ Perfect AI script not found")

def human_vs_human():
    """Human vs Human mode"""
    print("\n👥 Starting Human vs Human game...")
    from src.game.tictactoe import TicTacToeGUI
    gui = TicTacToeGUI()
    gui.run_human_vs_human()

def compare_models():
    """Compare performance of different AI models"""
    models = list_available_models()
    if len(models) < 2:
        print("❌ Need at least 2 models to compare")
        return
    
    print("\n📊 AI Model Comparison")
    print("-" * 50)
    print("Available models:")
    for i, (name, path) in enumerate(models, 1):
        file_size = os.path.getsize(path) / 1024
        mod_time = os.path.getmtime(path)
        import datetime
        mod_date = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        print(f"  {i}. {name} - {file_size:.1f}KB - {mod_date}")
    
    print("\n💡 To compare models, play against each one and observe:")
    print("   • Aggressiveness vs defensive play")
    print("   • Strategic positioning (center/corners)")
    print("   • Win/draw/loss patterns")

def main():
    """Main menu loop"""
    while True:
        try:
            display_menu()
            choice = input("Enter your choice (0-7): ").strip()
            
            if choice == '0':
                print("👋 Thanks for playing!")
                break
            elif choice == '1':
                train_original_dqn()
            elif choice == '2':
                train_enhanced_dqn()
            elif choice == '3':
                quick_test_enhanced()
            elif choice == '4':
                play_vs_dqn()
            elif choice == '5':
                play_vs_minimax()
            elif choice == '6':
                human_vs_human()
            elif choice == '7':
                compare_models()
            else:
                print("❌ Invalid choice. Please enter a number from 0-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    # Support both old command line interface and new interactive menu
    if len(sys.argv) >= 2:
        print("🔄 Legacy mode detected. Use interactive menu for full features.")
        mode = sys.argv[1]
        
        if mode == "1":
            train_original_dqn()
        elif mode == "2":
            play_vs_dqn()
        elif mode == "3":
            human_vs_human()
        elif mode == "4":
            play_vs_minimax()
        else:
            print(f"❌ Invalid mode: {mode}")
            print("Use interactive menu for full features:")
            main()
    else:
        main()