

from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
env.render()

# This agent random chooses a non-empty column.
def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

env.reset()
# Play as the first agent against default "random" agent.
env.run([my_agent, "random"])
env.render(mode="ipython", width=500, height=450)

# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()
while not env.done:
    my_action = my_agent(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()
My Action 5
My Action 1
My Action 2
My Action 3
My Action 4
My Action 3
My Action 1
My Action 1
My Action 5
My Action 1
My Action 0
My Action 1
My Action 0

def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))

# Run multiple episodes to estimate its performance.
print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
My Agent vs Random Agent: 0.8
My Agent vs Negamax Agent: -0.8

# "None" represents which agent you'll manually play as (first or second player).
env.play([None, "negamax"], width=500, height=450)

# "None" represents which agent you'll manually play as (first or second player).
env.play([None, "negamax"], width=500, height=450)