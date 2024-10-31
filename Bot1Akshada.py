import random
import numpy as np

# Helper function: Check if a move is valid (column is not full)
def is_valid_move(board, column):
    return board[0][column] == 0

# Helper function: Get the list of valid moves (non-full columns)
def get_valid_moves(board):
    return [col for col in range(board.shape[1]) if is_valid_move(board, col)]

# Helper function: Apply a move to the board
def make_move(board, column, player):
    for row in range(board.shape[0]-1, -1, -1):
        if board[row][column] == 0:
            board[row][column] = player
            break

# Heuristic evaluation of board state
def evaluate_board(board, player, opponent):
    score = 0
    center_column = board.shape[1] // 2

    # Reward for controlling the center column
    center_count = sum([1 for row in range(board.shape[0]) if board[row][center_column] == player])
    score += center_count * 3
    
    # Add more heuristics (e.g., horizontal, vertical, diagonal control)
    # The idea is to reward board positions where the player is close to forming a winning line
    return score

# Minimax function with n-step lookahead
def minimax(board, depth, maximizing_player, player, opponent, n_steps):
    valid_moves = get_valid_moves(board)
    
    # Base case: If we've reached max depth or there are no more valid moves
    if depth == 0 or len(valid_moves) == 0:
        return evaluate_board(board, player, opponent)
    
    if maximizing_player:
        max_eval = -float('inf')
        for col in valid_moves:
            temp_board = board.copy()
            make_move(temp_board, col, player)
            eval = minimax(temp_board, depth-1, False, player, opponent, n_steps)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for col in valid_moves:
            temp_board = board.copy()
            make_move(temp_board, col, opponent)
            eval = minimax(temp_board, depth-1, True, player, opponent, n_steps)
            min_eval = min(min_eval, eval)
        return min_eval

# Main bot function that selects the best move using n-step lookahead
def n_step_lookahead_bot(board, player, opponent, n_steps):
    valid_moves = get_valid_moves(board)
    best_score = -float('inf')
    best_move = random.choice(valid_moves)  # Fallback to random move
    
    for col in valid_moves:
        temp_board = board.copy()
        make_move(temp_board, col, player)
        score = minimax(temp_board, n_steps-1, False, player, opponent, n_steps)
        
        if score > best_score:
            best_score = score
            best_move = col
    
    return best_move

# Example usage in a game environment:
def act(observation, configuration):
    board = observation['board']
    board = np.array(board).reshape(configuration['rows'], configuration['columns'])
    
    player = observation['mark']  # The bot's mark (1 or 2)
    opponent = 3 - player         # The opponent's mark
    
    n_steps = 3  # You can adjust this value for deeper lookahead
    return n_step_lookahead_bot(board, player, opponent, n_steps)


# Simulate a game observation and configuration
observation = {
    'board': [0] * 42,  # A 7x6 empty board (42 cells)
    'mark': 1  # Player 1's turn
}

configuration = {
    'rows': 6,
    'columns': 7,
    'inarow': 4  # Connect 4 condition
}

# Call the act function to test the bot
move = act(observation, configuration)
print("Selected move:", move)
