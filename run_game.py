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
    print("🤖 AI vs AI BATTLES:")
    print("  6 - DQN vs DQN (model tournament)")
    print("  7 - DQN vs Minimax (enhanced vs perfect)")
    print()
    print("👥 HUMAN MODES:")
    print("  8 - Human vs Human")
    print("  9 - Compare AI Models (performance stats)")
    print()
    print("  0 - Exit")
    print("="*60)

def list_available_models():
    """List all available trained models with training method annotations"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    models = []
    
    # Look for enhanced models first (priority order)
    enhanced_models = glob.glob(f"{models_dir}/enhanced_dqn_*.pt")
    for model_path in sorted(enhanced_models, reverse=True):
        filename = os.path.basename(model_path)
        if "15k" in filename:
            models.append(("🎯 Enhanced DQN (15k episodes)", model_path))
        else:
            episodes = filename.replace("enhanced_dqn_", "").replace(".pt", "")
            models.append((f"🎯 Enhanced DQN ({episodes})", model_path))
    
    # Look for strategic models
    strategic_models = glob.glob(f"{models_dir}/strategic_dqn_*.pt")
    for model_path in sorted(strategic_models, reverse=True):
        filename = os.path.basename(model_path)
        episodes = filename.replace("strategic_dqn_", "").replace(".pt", "")
        models.append((f"🧠 Strategic DQN ({episodes})", model_path))
    
    # Look for original models
    original_models = glob.glob(f"{models_dir}/original_dqn_*.pt")
    for model_path in sorted(original_models, reverse=True):
        filename = os.path.basename(model_path)
        episodes = filename.replace("original_dqn_", "").replace(".pt", "")
        models.append((f"🔬 Original DQN ({episodes})", model_path))
    
    # Legacy models (old naming convention)
    if os.path.exists(f"{models_dir}/dqn_agent_final.pt"):
        models.append(("📦 Legacy Final Model", f"{models_dir}/dqn_agent_final.pt"))
    
    if os.path.exists(f"{models_dir}/strategic_showcase_final.pt"):
        models.append(("🧠 Strategic Showcase (legacy)", f"{models_dir}/strategic_showcase_final.pt"))
    
    # Legacy episode checkpoints (only show recent ones if no newer models)
    if len(models) < 3:
        episode_models = glob.glob(f"{models_dir}/dqn_agent_episode_*.pt")
        episode_models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        
        for model_path in episode_models[:3]:  # Show top 3 most recent
            episode_num = model_path.split('_')[-1].split('.')[0]
            models.append((f"📦 Legacy Episode {episode_num}", model_path))
    
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
    
    # Save with descriptive name
    agent.save(f"models/original_dqn_{episodes}ep.pt")
    print(f"✅ Original DQN training complete!")
    print(f"💾 Model saved as: models/original_dqn_{episodes}ep.pt")
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

# Save with descriptive name
agent.save(f"models/enhanced_dqn_{episodes}ep.pt")
print(f"✅ Enhanced training completed in {{(end_time-start_time)/60:.1f}} minutes")
print(f"📊 Historical opponents: {{len(agent.historical_opponents)}}")
print(f"💾 Model saved as: models/enhanced_dqn_{episodes}ep.pt")
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

def ai_tournament():
    """DQN vs DQN tournament between different models"""
    models = list_available_models()
    if len(models) < 2:
        print("❌ Need at least 2 models for tournament")
        print("🔧 Train more models using options 1, 2, or 3")
        return
    
    print("\n🏆 DQN vs DQN TOURNAMENT")
    print("=" * 50)
    print("Available models:")
    for i, (name, path) in enumerate(models, 1):
        print(f"  {i}. {name}")
    
    try:
        choice1 = int(input("\nSelect Model 1: ")) - 1
        choice2 = int(input("Select Model 2: ")) - 1
        
        if not (0 <= choice1 < len(models) and 0 <= choice2 < len(models)):
            print("❌ Invalid selection")
            return
            
        if choice1 == choice2:
            print("❌ Please select different models")
            return
            
    except ValueError:
        print("❌ Please enter valid numbers")
        return
    
    model1_name, model1_path = models[choice1]
    model2_name, model2_path = models[choice2]
    
    print(f"\n⚔️  BATTLE: {model1_name} vs {model2_name}")
    print("-" * 60)
    
    # Load both models
    from src.ai.agent import DQNAgent
    from src.game.tictactoe import TicTacToe
    
    agent1 = DQNAgent(epsilon=0)
    agent2 = DQNAgent(epsilon=0)
    
    try:
        agent1.load(model1_path)
        agent2.load(model2_path)
        print("✅ Both models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Tournament settings
    num_games = int(input("Number of games (default 100): ") or "100")
    
    results = {
        'model1_wins': 0,
        'model2_wins': 0, 
        'draws': 0,
        'games': []
    }
    
    print(f"\n🎮 Running {num_games} games...")
    
    for game_num in range(num_games):
        game = TicTacToe()
        
        # Alternate who goes first
        model1_goes_first = (game_num % 2 == 0)
        moves_made = 0
        
        while not game.game_over and moves_made < 9:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            # Determine whose turn it is
            if (game.current_player == 1 and model1_goes_first) or (game.current_player == -1 and not model1_goes_first):
                action = agent1.choose_action(game.get_state(), valid_moves)
                current_player = "Model1"
            else:
                action = agent2.choose_action(game.get_state(), valid_moves)
                current_player = "Model2"
            
            game.make_move(action)
            moves_made += 1
        
        # Record result
        result = game.get_result()
        game_result = None
        
        if result == 0:
            results['draws'] += 1
            game_result = "Draw"
        elif (result == 1 and model1_goes_first) or (result == -1 and not model1_goes_first):
            results['model1_wins'] += 1
            game_result = f"{model1_name} wins"
        else:
            results['model2_wins'] += 1
            game_result = f"{model2_name} wins"
        
        results['games'].append({
            'game_num': game_num + 1,
            'first_player': model1_name if model1_goes_first else model2_name,
            'result': game_result,
            'moves': moves_made
        })
        
        # Progress indicator
        if (game_num + 1) % 20 == 0:
            progress = (game_num + 1) / num_games * 100
            print(f"   Progress: {progress:.0f}% ({game_num + 1}/{num_games})")
    
    # Display results
    print(f"\n🏆 TOURNAMENT RESULTS")
    print("=" * 50)
    print(f"🤖 {model1_name}: {results['model1_wins']} wins ({results['model1_wins']/num_games*100:.1f}%)")
    print(f"🤖 {model2_name}: {results['model2_wins']} wins ({results['model2_wins']/num_games*100:.1f}%)")
    print(f"🤝 Draws: {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    
    # Performance analysis
    total_wins = results['model1_wins'] + results['model2_wins']
    if total_wins > 0:
        model1_win_rate = results['model1_wins'] / total_wins * 100
        print(f"\n📊 Head-to-head (excluding draws): {model1_win_rate:.1f}% - {100-model1_win_rate:.1f}%")
    
    # Determine winner
    if results['model1_wins'] > results['model2_wins']:
        print(f"\n🏆 WINNER: {model1_name}")
        advantage = results['model1_wins'] - results['model2_wins']
        print(f"   Advantage: +{advantage} games")
    elif results['model2_wins'] > results['model1_wins']:
        print(f"\n🏆 WINNER: {model2_name}")
        advantage = results['model2_wins'] - results['model1_wins']
        print(f"   Advantage: +{advantage} games")
    else:
        print(f"\n🤝 TIE: Both models equally matched")
    
    # Save detailed results
    save_results = input("\nSave detailed results to file? (y/n): ").lower()
    if save_results == 'y':
        filename = f"visualizations/tournament_{model1_name.replace(' ', '_')}_vs_{model2_name.replace(' ', '_')}.txt"
        with open(filename, 'w') as f:
            f.write(f"Tournament Results: {model1_name} vs {model2_name}\n")
            f.write(f"Games: {num_games}\n")
            f.write(f"Results: {results['model1_wins']}-{results['model2_wins']}-{results['draws']}\n\n")
            f.write("Game Details:\n")
            for game in results['games']:
                f.write(f"Game {game['game_num']}: {game['first_player']} first -> {game['result']} ({game['moves']} moves)\n")
        print(f"📁 Results saved to: {filename}")

def dqn_vs_minimax():
    """DQN vs Perfect Minimax battle"""
    models = list_available_models()
    if not models:
        print("❌ No DQN models found!")
        print("🔧 Train a model first using options 1, 2, or 3")
        return
    
    print("\n⚔️  DQN vs PERFECT MINIMAX")
    print("=" * 50)
    print("📋 Select DQN model to challenge the perfect AI:")
    for i, (name, path) in enumerate(models, 1):
        print(f"  {i}. {name}")
    
    try:
        choice = int(input("\nSelect DQN model: ")) - 1
        if not (0 <= choice < len(models)):
            print("❌ Invalid selection")
            return
    except ValueError:
        print("❌ Please enter a valid number")
        return
    
    model_name, model_path = models[choice]
    
    # Load models
    from src.ai.agent import DQNAgent
    from src.ai.minimax import MinimaxAgent
    from src.game.tictactoe import TicTacToe
    
    dqn_agent = DQNAgent(epsilon=0)
    minimax_agent = MinimaxAgent()
    
    try:
        dqn_agent.load(model_path)
        print(f"✅ {model_name} loaded successfully")
    except Exception as e:
        print(f"❌ Error loading DQN model: {e}")
        return
    
    num_games = int(input("Number of games (default 50): ") or "50")
    
    print(f"\n🎮 {model_name} vs Perfect Minimax AI")
    print(f"🎯 Running {num_games} games...")
    print("⚠️  Note: Perfect AI never loses, best possible is draws")
    
    results = {'dqn_wins': 0, 'draws': 0, 'minimax_wins': 0}
    
    for game_num in range(num_games):
        game = TicTacToe()
        dqn_goes_first = (game_num % 2 == 0)
        
        while not game.game_over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            if (game.current_player == 1 and dqn_goes_first) or (game.current_player == -1 and not dqn_goes_first):
                action = dqn_agent.choose_action(game.get_state(), valid_moves)
            else:
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
        
        if (game_num + 1) % 10 == 0:
            progress = (game_num + 1) / num_games * 100
            print(f"   Progress: {progress:.0f}%")
    
    # Display results
    print(f"\n🏆 BATTLE RESULTS")
    print("=" * 40)
    print(f"🤖 {model_name}: {results['dqn_wins']} wins ({results['dqn_wins']/num_games*100:.1f}%)")
    print(f"🤝 Draws: {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    print(f"🎯 Minimax: {results['minimax_wins']} wins ({results['minimax_wins']/num_games*100:.1f}%)")
    
    # Performance assessment
    non_loss_rate = (results['dqn_wins'] + results['draws']) / num_games * 100
    print(f"\n📊 DQN Non-Loss Rate: {non_loss_rate:.1f}%")
    
    if results['dqn_wins'] > 0:
        print(f"🎉 AMAZING: DQN actually won {results['dqn_wins']} games against perfect AI!")
    elif non_loss_rate >= 90:
        print("🏆 EXCELLENT: Near-perfect defensive play!")
    elif non_loss_rate >= 70:
        print("✅ GOOD: Strong strategic understanding")
    elif non_loss_rate >= 50:
        print("⚠️  FAIR: Decent play but room for improvement")
    else:
        print("❌ NEEDS WORK: Significant learning gaps remain")

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
    
    print("\n💡 To compare models, use options 6-7 for direct battles:")
    print("   • Option 6: DQN vs DQN tournament")
    print("   • Option 7: DQN vs Perfect Minimax")
    print("\n🔍 Or observe gameplay differences:")
    print("   • Aggressiveness vs defensive play")
    print("   • Strategic positioning (center/corners)")
    print("   • Win/draw/loss patterns")

def main():
    """Main menu loop"""
    while True:
        try:
            display_menu()
            choice = input("Enter your choice (0-9): ").strip()
            
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
                ai_tournament()
            elif choice == '7':
                dqn_vs_minimax()
            elif choice == '8':
                human_vs_human()
            elif choice == '9':
                compare_models()
            else:
                print("❌ Invalid choice. Please enter a number from 0-9.")
                
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