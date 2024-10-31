import random
import numpy as np
from kaggle_environments import make, evaluate

# Selects random valid column
def agent_random(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)

# Selects middle column
def agent_middle(obs, config):
    return config.columns // 2

# Selects leftmost valid column
def agent_leftmost(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return valid_moves[0]

# New agent function to select a valid move
def agent(observation, configuration):
    # Number of Columns on the Board
    columns = configuration.columns
    # The current serialized Board (rows x columns)
    board = observation.board
    # Find valid moves (columns with an empty spot)
    valid_moves = [col for col in range(columns) if board[col] == 0]
    
    # Return a random valid column
    return random.choice(valid_moves)

def get_win_percentages(agent1, agent2, n_rounds=100):
    # Create an environment
    env = make("connectx")
    config = env.configuration

    # Agent 1 goes first (roughly) half the time
    outcomes = evaluate("connectx", [agent1, agent2], num_episodes=n_rounds//2)
    
    # Agent 2 goes first (roughly) half the time
    outcomes += [[b, a] for [a, b] in evaluate("connectx", [agent2, agent1], num_episodes=n_rounds-n_rounds//2)]

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1, -1]) / len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1, 1]) / len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))

# Create a Connect Four environment
env = make("connectx")

# Agents play one game round
env.run([agent_leftmost, agent_random])

# Show the game
env.render(mode="ipython")

# Calculate win percentages
get_win_percentages(agent1=agent_middle, agent2=agent_random)
get_win_percentages(agent1=agent_leftmost, agent2=agent_random)
