#!/usr/bin/env python3
"""
AI Battle Script - Quick tournament between different AI models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.agent import DQNAgent
from src.ai.minimax import MinimaxAgent
from src.game.tictactoe import TicTacToe
import glob

def list_models():
    """List available DQN models with training method annotations"""
    models = []
    models_dir = "models"
    
    # Enhanced models
    enhanced_models = glob.glob(f"{models_dir}/enhanced_dqn_*.pt")
    for model_path in sorted(enhanced_models, reverse=True):
        filename = os.path.basename(model_path)
        if "15k" in filename:
            models.append(("🎯 Enhanced DQN (15k)", model_path))
        else:
            episodes = filename.replace("enhanced_dqn_", "").replace(".pt", "")
            models.append((f"🎯 Enhanced DQN ({episodes})", model_path))
    
    # Strategic models  
    strategic_models = glob.glob(f"{models_dir}/strategic_dqn_*.pt")
    for model_path in sorted(strategic_models, reverse=True):
        filename = os.path.basename(model_path)
        episodes = filename.replace("strategic_dqn_", "").replace(".pt", "")
        models.append((f"🧠 Strategic DQN ({episodes})", model_path))
    
    # Original models
    original_models = glob.glob(f"{models_dir}/original_dqn_*.pt")
    for model_path in sorted(original_models, reverse=True):
        filename = os.path.basename(model_path)
        episodes = filename.replace("original_dqn_", "").replace(".pt", "")
        models.append((f"🔬 Original DQN ({episodes})", model_path))
    
    # Legacy models
    if os.path.exists(f"{models_dir}/dqn_agent_final.pt"):
        models.append(("📦 Legacy Final", f"{models_dir}/dqn_agent_final.pt"))
    
    if os.path.exists(f"{models_dir}/strategic_showcase_final.pt"):
        models.append(("🧠 Strategic (legacy)", f"{models_dir}/strategic_showcase_final.pt"))
    
    # Legacy episodes (if needed)
    if len(models) < 3:
        episode_models = glob.glob(f"{models_dir}/dqn_agent_episode_*.pt")
        episode_models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        
        for model_path in episode_models[:3]:
            episode_num = model_path.split('_')[-1].split('.')[0]
            models.append((f"📦 Legacy Ep{episode_num}", model_path))
    
    return models

def quick_battle(model1_path, model2_path, model1_name, model2_name, games=20):
    """Quick battle between two models"""
    agent1 = DQNAgent(epsilon=0)
    agent2 = DQNAgent(epsilon=0)
    
    try:
        agent1.load(model1_path)
        agent2.load(model2_path)
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None
    
    results = {'model1_wins': 0, 'model2_wins': 0, 'draws': 0}
    
    for game_num in range(games):
        game = TicTacToe()
        model1_goes_first = (game_num % 2 == 0)
        
        while not game.game_over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            if (game.current_player == 1 and model1_goes_first) or (game.current_player == -1 and not model1_goes_first):
                action = agent1.choose_action(game.get_state(), valid_moves)
            else:
                action = agent2.choose_action(game.get_state(), valid_moves)
            
            game.make_move(action)
        
        result = game.get_result()
        if result == 0:
            results['draws'] += 1
        elif (result == 1 and model1_goes_first) or (result == -1 and not model1_goes_first):
            results['model1_wins'] += 1
        else:
            results['model2_wins'] += 1
    
    return results

def battle_vs_minimax(model_path, model_name, games=20):
    """Battle DQN vs Minimax"""
    dqn = DQNAgent(epsilon=0)
    minimax = MinimaxAgent()
    
    try:
        dqn.load(model_path)
    except Exception as e:
        print(f"❌ Error loading DQN model: {e}")
        return None
    
    results = {'dqn_wins': 0, 'draws': 0, 'minimax_wins': 0}
    
    for game_num in range(games):
        game = TicTacToe()
        dqn_goes_first = (game_num % 2 == 0)
        
        while not game.game_over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            if (game.current_player == 1 and dqn_goes_first) or (game.current_player == -1 and not dqn_goes_first):
                action = dqn.choose_action(game.get_state(), valid_moves)
            else:
                action = minimax.choose_action(game.get_state(), valid_moves)
            
            game.make_move(action)
        
        result = game.get_result()
        if result == 0:
            results['draws'] += 1
        elif (result == 1 and dqn_goes_first) or (result == -1 and not dqn_goes_first):
            results['dqn_wins'] += 1
        else:
            results['minimax_wins'] += 1
    
    return results

def main():
    """Run AI battle showcase"""
    print("⚔️  AI BATTLE ARENA")
    print("=" * 50)
    
    models = list_models()
    if len(models) < 2:
        print("❌ Need at least 2 DQN models for battles")
        print("🔧 Train models using the main interface first")
        return
    
    print("📋 Available Models:")
    for i, (name, path) in enumerate(models, 1):
        print(f"  {i}. {name}")
    
    print(f"\n🎮 Running quick tournament ({20} games each)...")
    
    # All vs all tournament
    battle_results = []
    
    # DQN vs DQN battles
    print(f"\n🤖 DQN vs DQN BATTLES:")
    print("-" * 30)
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            name1, path1 = models[i]
            name2, path2 = models[j]
            
            print(f"⚔️  {name1} vs {name2}... ", end="")
            
            results = quick_battle(path1, path2, name1, name2, 20)
            if results:
                w1, w2, draws = results['model1_wins'], results['model2_wins'], results['draws']
                print(f"{w1}-{w2}-{draws}")
                
                if w1 > w2:
                    winner = name1
                    margin = w1 - w2
                elif w2 > w1:
                    winner = name2  
                    margin = w2 - w1
                else:
                    winner = "TIE"
                    margin = 0
                
                battle_results.append({
                    'battle': f"{name1} vs {name2}",
                    'result': f"{w1}-{w2}-{draws}",
                    'winner': winner,
                    'margin': margin
                })
            else:
                print("FAILED")
    
    # DQN vs Minimax battles
    print(f"\n🎯 DQN vs PERFECT MINIMAX:")
    print("-" * 35)
    
    minimax_results = []
    for name, path in models:
        print(f"🤖 {name} vs Minimax... ", end="")
        
        results = battle_vs_minimax(path, name, 20)
        if results:
            wins, draws, losses = results['dqn_wins'], results['draws'], results['minimax_wins']
            non_loss_rate = (wins + draws) / 20 * 100
            print(f"{wins}-{draws}-{losses} ({non_loss_rate:.0f}% non-loss)")
            
            minimax_results.append({
                'model': name,
                'vs_minimax': f"{wins}-{draws}-{losses}",
                'non_loss_rate': non_loss_rate
            })
        else:
            print("FAILED")
    
    # Summary
    print(f"\n🏆 TOURNAMENT SUMMARY")
    print("=" * 40)
    
    if battle_results:
        print("🥊 DQN Head-to-Head Results:")
        for battle in battle_results:
            if battle['winner'] != "TIE":
                print(f"   {battle['battle']}: {battle['winner']} wins (+{battle['margin']})")
            else:
                print(f"   {battle['battle']}: TIE")
    
    if minimax_results:
        print(f"\n🎯 vs Perfect Minimax Performance:")
        # Sort by non-loss rate
        minimax_results.sort(key=lambda x: x['non_loss_rate'], reverse=True)
        
        for i, result in enumerate(minimax_results, 1):
            rate = result['non_loss_rate']
            if rate >= 70:
                emoji = "🏆"
            elif rate >= 50:
                emoji = "✅"
            elif rate >= 30:
                emoji = "⚠️"
            else:
                emoji = "❌"
            
            print(f"   {i}. {emoji} {result['model']}: {result['vs_minimax']} ({rate:.0f}%)")
    
    print(f"\n💡 Use 'python3 run_game.py' for detailed battles and training")

if __name__ == "__main__":
    main()