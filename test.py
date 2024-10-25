from kaggle_environments import make, evaluate

env = make("connectx", debug=True)
print(list(env.agents))  # This will print the list of agents

# Run a game between two random agents
env.run(["random", "random"]) 

# Render the game in an IPython environment (like Jupyter Notebook or Google Colab)
env.render(mode="ipython")