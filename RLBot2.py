import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque

# Constants and Hyperparameters
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
ROWS = 6
COLUMNS = 7
WINNING_LENGTH = 4

# Initialize experience replay memory
memory = deque(maxlen=MEMORY_SIZE)

# Define the Q-Network (using Keras)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(ROWS, COLUMNS, 1)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(COLUMNS, activation='linear')  # Output layer
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)  # Use an optimizer like Adam
model.compile(optimizer=optimizer, loss='mse')  # Mean Squared Error loss

# Target Network (for stability)
target_model = keras.models.clone_model(model)  # Create a copy of the main model
target_model.set_weights(model.get_weights())

MODEL_PATH = ".weights.h5"


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

def check_winner(board, player):  # Corrected diagonal checks
    # Check horizontal
    for row in range(ROWS):
        for col in range(COLUMNS - WINNING_LENGTH + 1):
            if all(board[row, col + i] == player for i in range(WINNING_LENGTH)):
                return True
    # Check vertical
    for row in range(ROWS - WINNING_LENGTH + 1):
        for col in range(COLUMNS):
            if all(board[row + i, col] == player for i in range(WINNING_LENGTH)):
                return True
    # Check diagonals
    for row in range(ROWS - WINNING_LENGTH + 1):
        for col in range(COLUMNS - WINNING_LENGTH + 1):
            if all(board[row + i, col + i] == player for i in range(WINNING_LENGTH)):
                return True
            if all(board[row + i, col + WINNING_LENGTH - 1 - i] == player for i in range(WINNING_LENGTH)):
                return True
    return False

# Training step
# def train_q_network(states, actions, rewards, next_states, dones):
#     q_next = model.predict(next_states, verbose=0).numpy()
#     target_q_values = rewards + GAMMA * np.max(q_next, axis=1) * (1 - dones)

#     masks = tf.one_hot(actions, COLUMNS)

#     with tf.GradientTape() as tape:
#       q_values = model(states)
#       masked_q_values = tf.reduce_sum(masks * q_values, axis=1)
#       loss = tf.reduce_mean(keras.losses.mean_squared_error(target_q_values, masked_q_values))


#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Epsilon-greedy action selection (similar to before, but using the Keras model)
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(get_valid_moves(state))
    else:
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, axis=-1)  # Add channel dimension
        state_tensor = tf.expand_dims(state_tensor, axis=0)  # Add batch dimension
        q_values = model(state_tensor).numpy()[0]
        valid_moves = get_valid_moves(state)
        valid_q_values = [(q_values[col], col) for col in valid_moves]
        _, best_move = max(valid_q_values)
        return best_move


# Self-play episode (similar as before)
def self_play_episode(epsilon):  # Corrected and complete implementation
    board = np.zeros((ROWS, COLUMNS), dtype=np.int8)
    player = 1
    done = False
    moves = 0

    while not done:
        action = select_action(board, epsilon)
        current_state = np.copy(board)
        make_move(board, action, player)

        reward = -0.1 # Small negative reward to encourage faster wins.
        if check_winner(board, player):
            reward = 1
            done = True
        elif moves == ROWS * COLUMNS - 1:
            reward = 0
            done = True

        next_state = np.copy(board)
        memory.append((current_state, action, reward, next_state, done)) # Append current state to memory
        
        player = 3 - player
        moves += 1

    return reward

def train_q_network(states, actions, rewards, next_states, dones):
    q_next = model.predict(next_states, verbose=0)
    target_q_values = rewards + GAMMA * np.max(q_next, axis=1) * (1 - dones)

    masks = tf.one_hot(actions, COLUMNS)

    with tf.GradientTape() as tape:
        q_values = model(states)
        masked_q_values = tf.reduce_sum(masks * q_values, axis=1)

        # Use the MeanSquaredError loss function
        mse_loss = keras.losses.MeanSquaredError()
        loss = mse_loss(target_q_values, masked_q_values)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def train_agent(num_episodes):
    best_reward = float('-inf')
    epsilon = EPSILON  # Initialize epsilon once
    loss = None  # Initialize loss variable

    for episode in range(1, num_episodes + 1):
        # Decay epsilon
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        
        # Play an episode and get the reward
        reward = self_play_episode(epsilon)

        # Train the Q-Network if there are enough samples
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to NumPy arrays and reshape
            states = np.array(states, dtype=np.float32).reshape(-1, ROWS, COLUMNS, 1)
            next_states = np.array(next_states, dtype=np.float32).reshape(-1, ROWS, COLUMNS, 1)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            actions = np.array(actions)

            # Train the Q-network and get the loss
            loss = train_q_network(states, actions, rewards, next_states, dones)

        # Update target network periodically
        if episode % 100 == 0:
            target_model.set_weights(model.get_weights())
            print(f"Target network updated at episode {episode}")

        # Save model periodically and when the best reward is achieved
        if episode % 1000 == 0:
            model.save_weights(MODEL_PATH)
            print(f"Model saved at episode {episode}, Reward: {reward}, Epsilon: {epsilon:.4f}, Loss: {loss.numpy() if loss else 'N/A'}")

        if reward > best_reward:
            model.save_weights(MODEL_PATH)
            best_reward = reward
            print(f"New best model saved at episode {episode} with reward {reward}")

        # Log progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {reward}, Epsilon: {epsilon:.4f}, Loss: {loss.numpy() if loss else 'N/A'}")

    print("Training completed.")


# Main bot function (for Kaggle) - uses epsilon-greedy action selection
def act(observation, configuration): # Kaggle bot function
    board = np.array(observation['board']).reshape(configuration['rows'], configuration['columns'])
    
    model.load_weights(MODEL_PATH)

    epsilon = 0.05 # Exploit learned policy
    action = select_action(board, epsilon)
    return action

if __name__ == "__main__":  # This ensures the training code runs only when you execute the script directly
    num_episodes = 50000  # Or more
    train_agent(num_episodes)
    print("Training completed.")

for episode in range(1000): # Number of training episodes
    epsilon = max(MIN_EPSILON, EPSILON * EPSILON_DECAY**episode) # Correct epsilon decay
    reward = self_play_episode(epsilon)  # Call self_play_episode to populate memory

    if len(memory) > BATCH_SIZE:
        batch = random.sample(memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to NumPy arrays and reshape
        states = np.array(states, dtype=np.float32).reshape(-1, ROWS, COLUMNS, 1)
        next_states = np.array(next_states, dtype=np.float32).reshape(-1, ROWS, COLUMNS, 1)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)  # Convert dones to NumPy array
        actions = np.array(actions)

        train_q_network(states, actions, rewards, next_states, dones)

    if episode % 10 == 0:
        target_model.set_weights(model.get_weights()) # Update target network

    print(f"Episode {episode}, Reward: {reward}, Epsilon: {epsilon}")

best_reward = float('-inf')

if reward > best_reward:
    model.save_weights(MODEL_PATH)
    best_reward = reward
    print(f"New best model saved at episode {episode} with reward {reward}")

print("Training completed.")
print(f"Episode {episode}, Reward: {reward}, Epsilon: {epsilon}, Loss: {loss.numpy()}")

