# Stdout replacement is a temporary workaround, as noted in the instructions
import sys
from kaggle_environments import utils, make

# Save the current stdout, because we will replace it temporarily
out = sys.stdout

# Read the submission file you just wrote
submission = utils.read_file("submission.py")

# Retrieve the last callable (function) in the submission file
agent = utils.get_last_callable(submission)

# Restore stdout
sys.stdout = out

# Create the ConnectX environment
env = make("connectx", debug=True)

# Run your agent against itself to check if it is fully functional
env.run([agent, agent])

# Check the status of both players
if env.state[0].status == env.state[1].status == "DONE":
    print("Success!")
else:
    print("Failed...")
