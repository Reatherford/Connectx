import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque

# Constants
GAMMA = 0.99  # Discount factor for future rewards
EPSILON = 1.0  # Exploration rate
EPSILON_DECAY = 0.995  # Decay rate for epsilon
MIN_EPSILON = 0.1
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64

# Initialize the Connect X configuration
ROWS = 6
COLUMNS = 7
WINNING_LENGTH = 4

# Initialize experience replay memory
memory = deque(maxlen=MEMORY_SIZE)


# Define the Q-Network
class QNetwork(tf.keras.Model):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(action_size, activation='linear')

    def call(self, state):
        x = tf.reshape(state, (-1, ROWS, COLUMNS, 1))  # Reshape to match input shape for Conv2D
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)


# Initialize the main and target Q-networks
model = QNetwork(action_size=COLUMNS)
target_model = QNetwork(action_size=COLUMNS)

# Compile both models
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
             loss='mse')
target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss='mse')

target_model.set_weights(model.get_weights())


# Helper functions for board operations
def is_valid_move(board, column):
    return board[0][column] == 0


def get_valid_moves(board):
    return [col for col in range(COLUMNS) if is_valid_move(board, col)]


def make_move(board, column, player):
    for row in range(ROWS - 1, -1, -1):
        if board[row][column] == 0:
            board[row][column] = player
            break


# Helper function for training the Q-network
def train_q_network():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = np.array(states)
    next_states = np.array(next_states)

    # Get current Q values
    current_q_values = model.predict(states)
    # Get next Q values from target network
    next_q_values = target_model.predict(next_states)

    # Update Q values for the actions taken
    for i in range(BATCH_SIZE):
        if dones[i]:
            current_q_values[i][actions[i]] = rewards[i]
        else:
            current_q_values[i][actions[i]] = rewards[i] + GAMMA * np.max(next_q_values[i])

    # Train the model with updated Q values
    model.train_on_batch(states, current_q_values)


# Epsilon-greedy action selection
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(get_valid_moves(state))
    else:
        state_tensor = np.expand_dims(state, axis=0)  # Add batch dimension
        q_values = model(state_tensor)
        valid_moves = get_valid_moves(state)
        valid_q_values = [(q_values[0][col].numpy(), col) for col in valid_moves]
        _, best_move = max(valid_q_values)
        return best_move


# Self-play loop for training the bot
def self_play_episode(epsilon):
    board = np.zeros((ROWS, COLUMNS))
    player = 1
    done = False
    moves = 0
    reward = 0

    while not done:
        # Bot chooses action
        action = select_action(board, epsilon)
        make_move(board, action, player)

        # Check for win/loss/draw
        if check_winner(board, player):
            reward = 1
            done = True
        elif moves == ROWS * COLUMNS - 1:
            reward = 0  # Draw
            done = True
        else:
            reward = -0.1  # Small penalty for each move to encourage quicker wins

        next_state = np.copy(board)
        memory.append((np.copy(board), action, reward, next_state, done))

        # Alternate player
        player = 3 - player
        moves += 1

    return reward


# Main bot function for Connect X competition
def act(observation, configuration):
    board = np.array(observation['board']).reshape(configuration['rows'], configuration['columns'])
    epsilon = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    action = select_action(board, epsilon)
    return action


# Function to check for a winner
def check_winner(board, player):
    # Check horizontal, vertical, and diagonal for a win condition
    for row in range(ROWS):
        for col in range(COLUMNS):
            if (col <= COLUMNS - WINNING_LENGTH and np.all(board[row, col:col + WINNING_LENGTH] == player)) or \
                    (row <= ROWS - WINNING_LENGTH and np.all(board[row:row + WINNING_LENGTH, col] == player)) or \
                    (col <= COLUMNS - WINNING_LENGTH and row <= ROWS - WINNING_LENGTH and
                     all([board[row + i, col + i] == player for i in range(WINNING_LENGTH)])) or \
                    (col >= WINNING_LENGTH - 1 and row <= ROWS - WINNING_LENGTH and
                     all([board[row + i, col - i] == player for i in range(WINNING_LENGTH)])):
                return True
    return False


# Training loop
for episode in range(1000):  # Train over multiple episodes
    epsilon = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    reward = self_play_episode(epsilon)
    train_q_network()

    # Update target network weights periodically
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())

    print(f"Episode {episode}, Reward: {reward}, Epsilon: {epsilon}")

print("Training completed.")