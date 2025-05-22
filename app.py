from flask import Flask, render_template, jsonify, request
import pickle
from tictactoe import TicTacToe, QLearningAgent

app = Flask(__name__)

# Global game state
game_state = {
    'game': None,
    'ai_player': None,
    'human_letter': None,
    'current_turn': None
}

def init_game(human_plays_as):
    """Initialize a new game"""
    game_state['game'] = TicTacToe()
    game_state['human_letter'] = human_plays_as
    ai_letter = 'O' if human_plays_as == 'X' else 'X'
    
    # Create and load AI player
    game_state['ai_player'] = QLearningAgent(ai_letter, train_mode=False)
    try:
        game_state['ai_player'].load_model(f"diverse_q_table_{ai_letter}.pkl")
        print(f"Loaded AI model for {ai_letter}")
    except FileNotFoundError:
        print(f"Warning: Could not load trained model for {ai_letter}")
    
    # Set starting turn
    game_state['current_turn'] = 'X'
    
    # If AI plays X, make its move
    if ai_letter == 'X':
        make_ai_move()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    data = request.json
    human_plays_as = data.get('play_as', 'X')
    init_game(human_plays_as)
    
    return jsonify({
        'board': game_state['game'].board,
        'current_turn': game_state['current_turn'],
        'human_letter': game_state['human_letter'],
        'game_over': False
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    square = data.get('square')
    
    if not game_state['game'] or game_state['game'].current_winner:
        return jsonify({'error': 'Game not initialized or already over'}), 400
    
    # Check if it's human's turn
    if game_state['current_turn'] != game_state['human_letter']:
        return jsonify({'error': 'Not your turn'}), 400
    
    # Make human move
    if not game_state['game'].make_move(square, game_state['human_letter']):
        return jsonify({'error': 'Invalid move'}), 400
    
    # Check for winner or draw
    if game_state['game'].current_winner or not game_state['game'].empty_squares():
        return jsonify({
            'board': game_state['game'].board,
            'current_turn': None,
            'game_over': True,
            'winner': game_state['game'].current_winner,
            'is_draw': game_state['game'].current_winner is None
        })
    
    # Switch turn and make AI move
    game_state['current_turn'] = 'O' if game_state['current_turn'] == 'X' else 'X'
    make_ai_move()
    
    # Check for winner or draw after AI move
    game_over = game_state['game'].current_winner is not None or not game_state['game'].empty_squares()
    
    return jsonify({
        'board': game_state['game'].board,
        'current_turn': game_state['current_turn'],
        'game_over': game_over,
        'winner': game_state['game'].current_winner,
        'is_draw': game_over and game_state['game'].current_winner is None
    })

def make_ai_move():
    """Make AI move and update game state"""
    ai_square = game_state['ai_player'].get_move(game_state['game'])
    if ai_square is not None:
        game_state['game'].make_move(ai_square, game_state['ai_player'].letter)
        game_state['current_turn'] = 'O' if game_state['current_turn'] == 'X' else 'X'

@app.route('/get_state', methods=['GET'])
def get_state():
    if not game_state['game']:
        return jsonify({'error': 'No game in progress'}), 400
    
    game_over = game_state['game'].current_winner is not None or not game_state['game'].empty_squares()
    
    return jsonify({
        'board': game_state['game'].board,
        'current_turn': game_state['current_turn'],
        'human_letter': game_state['human_letter'],
        'game_over': game_over,
        'winner': game_state['game'].current_winner,
        'is_draw': game_over and game_state['game'].current_winner is None
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)