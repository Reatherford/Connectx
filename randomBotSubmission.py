import random

# Agent function that meets the submission criteria
def agent(observation, configuration):
    # Number of Columns on the Board
    columns = configuration.columns
    # The current serialized Board (rows x columns)
    board = observation.board
    # Find valid moves (columns with an empty spot)
    valid_moves = [col for col in range(columns) if board[col] == 0]
    
    # Return a random valid column
    return random.choice(valid_moves)
