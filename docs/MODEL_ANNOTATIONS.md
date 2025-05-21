# 🏷️ Model Training Method Annotations

This document explains the enhanced model naming system that clearly identifies which training method was used for each AI model.

---

## 📋 **Model Naming Convention**

### **🎯 Enhanced DQN Models**
- **Filename**: `enhanced_dqn_[episodes]ep.pt`
- **Example**: `enhanced_dqn_15k.pt`, `enhanced_dqn_10000ep.pt`
- **Features**:
  - Strategic reward bonuses (center +0.5, corners +0.3, forks +2.0)
  - Historical opponent archives (up to 5 models)
  - Progressive opponent usage (30% → 70%)
  - Higher win rewards (+10 vs +1)

### **🧠 Strategic DQN Models** 
- **Filename**: `strategic_dqn_[episodes]ep.pt`
- **Example**: `strategic_dqn_1000ep.pt`, `strategic_dqn_2000ep.pt`
- **Features**:
  - Focused strategic training demonstration
  - Real-time strategic behavior tracking
  - Emphasis on center/corner positioning
  - Detailed learning analytics

### **🔬 Original DQN Models**
- **Filename**: `original_dqn_[episodes]ep.pt` 
- **Example**: `original_dqn_5000ep.pt`
- **Features**:
  - Baseline Double DQN implementation
  - Prioritized Experience Replay
  - Standard self-play training
  - Basic reward structure (+1/-1/+0.5)

### **📦 Legacy Models**
- **Filename**: Various old conventions
- **Examples**: `dqn_agent_final.pt`, `dqn_agent_episode_*.pt`
- **Status**: Pre-annotation system models

---

## 🎮 **How to Use Annotated Models**

### **In Interactive Menu**
```bash
python3 run_game.py
```
The enhanced interface automatically detects and categorizes models:
```
📁 Available Trained Models:
  1. 🎯 Enhanced DQN (15k episodes) (417.2 KB)
  2. 🧠 Strategic DQN (1000ep) (417.5 KB)  
  3. 🔬 Original DQN (5000ep) (417.1 KB)
  4. 📦 Legacy Final Model (417.0 KB)
```

### **In AI Battle Arena**
```bash
python3 scripts/ai_battle.py
```
Quick tournaments with clear method identification:
```
⚔️ 🎯 Enhanced DQN (15k) vs 🔬 Original DQN (5000ep)... 8-2-10
⚔️ 🧠 Strategic DQN (1000ep) vs 🔬 Original DQN (5000ep)... 12-7-1
```

---

## 📊 **Performance Comparison by Method**

| Training Method | Typical Performance vs Minimax | Strategic Behavior | Training Time |
|----------------|--------------------------------|-------------------|---------------|
| **🎯 Enhanced DQN** | 35-50% non-loss rate | High strategic awareness | Moderate |
| **🧠 Strategic DQN** | 30-40% non-loss rate | Very high in early episodes | Fast |
| **🔬 Original DQN** | 20-35% non-loss rate | Basic positional play | Fast |
| **📦 Legacy** | Variable | Unknown method | N/A |

---

## 🔄 **Migration from Legacy Models**

### **Automatic Detection**
The system automatically categorizes existing models as "Legacy" while new training uses the annotated naming.

### **Retraining Recommendation**
For best clarity, retrain important models using the new system:

```bash
# Original DQN baseline
python3 run_game.py → Option 1 → 5000 episodes
# Creates: models/original_dqn_5000ep.pt

# Enhanced DQN with all features  
python3 run_game.py → Option 2 → 10000 episodes
# Creates: models/enhanced_dqn_10000ep.pt

# Strategic showcase
python3 scripts/train_strategic_showcase.py → 1000 episodes
# Creates: models/strategic_dqn_1000ep.pt
```

---

## 🛠️ **Technical Implementation**

### **Model Detection Logic**
```python
def list_available_models():
    """Detect and categorize models by filename patterns"""
    
    # Priority order:
    # 1. Enhanced DQN models (enhanced_dqn_*.pt)
    # 2. Strategic DQN models (strategic_dqn_*.pt)  
    # 3. Original DQN models (original_dqn_*.pt)
    # 4. Legacy models (old naming conventions)
```

### **Training Script Integration**
Each training method automatically saves with appropriate naming:
- `train_original_dqn()` → `original_dqn_[episodes]ep.pt`
- `train_enhanced_dqn()` → `enhanced_dqn_[episodes]ep.pt`
- `train_strategic_showcase()` → `strategic_dqn_[episodes]ep.pt`

---

## 🎯 **Benefits of Annotation System**

1. **Clear Method Identification**: Instantly know which training approach was used
2. **Performance Comparison**: Easy to compare different methods head-to-head
3. **Experiment Tracking**: Better organization for AI research and development
4. **User Experience**: Clearer model selection in battles and gameplay
5. **Documentation**: Self-documenting model collection

---

## 💡 **Future Extensions**

The annotation system is designed to be extensible for additional training methods:

- **🧬 Genetic Algorithm DQN**: `genetic_dqn_[episodes]ep.pt`
- **🤝 Multi-Agent DQN**: `multiagent_dqn_[episodes]ep.pt`  
- **🎲 Curriculum DQN**: `curriculum_dqn_[episodes]ep.pt`
- **🔮 Transformer DQN**: `transformer_dqn_[episodes]ep.pt`

Each new method gets its own emoji and clear naming pattern for easy identification and comparison.

---

*Enhanced model organization for better AI development workflow* 🚀