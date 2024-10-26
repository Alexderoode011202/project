from basics import maze_generator, reward_function
from numpy import ndarray
import numpy as np
from typing import Dict, Callable, Tuple
import matplotlib.pyplot as plt

def value_iteration(maze: ndarray = maze_generator(),
                    reward_function: Callable = reward_function,
                    threshold: float = 0.5,  # Convergence threshold for value updates
                    discount_value: float = 0.9,  # Discount factor for future rewards
                    checkpoint_position: Tuple[int, int] = (8, 5),  # Position of checkpoint
                    end_position: Tuple[int, int] = (8, 7),  # Position of the goal
                    verbose: bool = True,  # Verbosity flag for logging
                    images: Dict[int, str] = {1: "images\\arrow-up.png",  
                                               2: "images\\arrow-down.png", 
                                               3: "images\\arrow-left.png", 
                                               4: "images\\arrow-right.png"}  
                    ) -> ndarray:
    # Initialize value function and policy for whether the checkpoint was passed or not
    value_func: Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}
    policy: Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}

    # Lists to store metrics for analysis
    steps_to_goal = []  
    convergence_speeds = []  
    learning_stabilities = []  

    biggest_change = float('inf')  # Initialize the largest change in value function
    passed_checkpoint = (True, False)  # States to evaluate (checkpoint passed or not)
    iteration = 0  # Iteration counter

    # Loop until convergence or maximum iterations reached
    while biggest_change > threshold and iteration < 100:
        iteration += 1
        biggest_change = 0.0  # Reset biggest change for the iteration
        updates = []  # List to track updates in the current iteration

        # Iterate over the states of checkpoint passed
        for checkpoint_value in passed_checkpoint:
            # Iterate through all positions in the maze, skipping walls
            for x in range(1, maze.shape[0] - 1):
                for y in range(1, maze.shape[1] - 1):
                    if maze[x, y] == '1':  # Skip walls represented by '1'
                        continue

                    current_pos = (x, y)  # Current position of the agent
                    best_value = -float("inf")  # Initialize best value for current position

                    # Evaluate all possible moves
                    for move in ['up', 'down', 'left', 'right']:
                        # Calculate new position based on move direction
                        new_pos = (
                            current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],
                            current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move]
                        )
                        # Check if the new position is a wall; if so, revert to current position
                        if maze[new_pos] == '1':
                            new_pos = current_pos

                        # Calculate the reward for the current move
                        reward = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value,
                                                 checkpoint_position=checkpoint_position, end_position=end_position)

                        # Calculate the value of the current action
                        value = reward + discount_value * value_func[checkpoint_value][new_pos]

                        # Update best value if current value is better
                        if value > best_value:
                            best_value = value

                    # Calculate the change in value for the current position
                    old_value = value_func[checkpoint_value][current_pos]
                    change = abs(best_value - old_value)
                    updates.append(change)  # Track update change
                    biggest_change = max(biggest_change, change)  # Update largest change
                    value_func[checkpoint_value][current_pos] = best_value  # Update value function

        # Store convergence metrics
        convergence_speeds.append(biggest_change)
        learning_stabilities.append(np.std(updates))

    # Plot Value Functions
    for checkpoint_value in passed_checkpoint:
        plt.matshow(value_func[checkpoint_value])  # Visualize value function
        plt.title(f"Value Function - Checkpoint Passed: {checkpoint_value}")
        plt.colorbar()  # Show color bar
        plt.show()

    # Derive and plot policies from the computed value function
    for checkpoint_value in passed_checkpoint:
        for x in range(1, maze.shape[0] - 1):
            for y in range(1, maze.shape[1] - 1):
                current_pos = (x, y)

                if maze[x, y] == '1':  # Skip walls
                    continue

                best_move = ""  # Initialize best move
                best_value = -float("inf")  # Initialize best value for move
                for move in ['up', 'down', 'left', 'right']:
                    # Calculate new position based on move direction
                    new_pos = (
                        current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],
                        current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move]
                    )
                    # Check if the new position is a wall; if so, revert to current position
                    if maze[new_pos] == '1':
                        new_pos = current_pos

                    # Calculate the reward for the current move
                    reward = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value,
                                             checkpoint_position=checkpoint_position, end_position=end_position)
                    # Calculate the value of the current action
                    value = reward + discount_value * value_func[checkpoint_value][new_pos]

                    # Update best move if current value is better
                    if value > best_value:
                        best_value = value
                        best_move = move

                # Map best move to corresponding policy value
                policy_value = {"up": 1, "down": 2, "left": 3, "right": 4}.get(best_move, 0)
                policy[checkpoint_value][current_pos] = policy_value  # Update policy

    # Policy Visualization
    for checkpoint_value in passed_checkpoint:
        plt.matshow(policy[checkpoint_value])  # Visualize policy
        plt.title(f"Policy {checkpoint_value} - Best Policy Move")
        plt.colorbar()

        # Overlay different images on policy squares
        for x in range(policy[checkpoint_value].shape[0]):
            for y in range(policy[checkpoint_value].shape[1]):
                policy_value = policy[checkpoint_value][x, y]
                if policy_value in images.keys():  # Check if policy value has an associated image
                    img = plt.imread(images[policy_value])  # Load the corresponding image
                    # Overlay image in the correct grid cell
                    plt.imshow(img, extent=[y - 0.5, y + 0.5, x + 0.5, x - 0.5], alpha=0.5, aspect='auto')
                if policy_value != 0:
                    # Display policy value in the grid cell
                    plt.text(y, x, str(int(policy_value)), ha='center', va='center', color='black')

        plt.xlim(-0.5, policy[True].shape[1] - 0.5)  # Set x-axis limits
        plt.ylim(policy[True].shape[0] - 0.5, -0.5)  # Set y-axis limits
        plt.show()

    # Path Tracing
    def trace_path(policy, start_position, end_position, max_steps=100):
        path = []  # List to store the path taken
        current_pos = start_position  # Starting position of the agent
        steps = 0  # Step counter
        # Dictionary to map policy values to move directions
        move_dict = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        passed_checkpoint = False  # Flag to track if checkpoint was passed

        # Trace the path until reaching the end position or maximum steps
        while current_pos != end_position and steps < max_steps:
            path.append(current_pos)  # Add current position to path
            if current_pos == checkpoint_position:
                passed_checkpoint = True  # Update flag if checkpoint reached
            move = int(policy[passed_checkpoint][current_pos])  # Get the move from the policy
            # Calculate next position based on the move
            next_pos = (current_pos[0] + move_dict.get(move, (0, 0))[0],
                        current_pos[1] + move_dict.get(move, (0, 0))[1])
            # Break if the next position is a wall
            if maze[next_pos] == '1':
                break
            current_pos = next_pos  # Update current position
            steps += 1  # Increment step counter

        path.append(end_position)  # Add end position to path
        return path  # Return the traced path

    start_position = (1, 1)  # Starting position of the agent
    optimal_path = trace_path(policy, start_position, checkpoint_position)  # Path to checkpoint
    optimal_path2 = trace_path(policy, checkpoint_position, end_position)  # Path to end

    # Prepare data for plotting the traced paths
    path_x = [p[1] for p in optimal_path]
    path_y = [p[0] for p in optimal_path]
    path_x2 = [p[1] for p in optimal_path2]
    path_y2 = [p[0] for p in optimal_path2]

    # Visualize the policy with the traced paths
    plt.matshow(policy[False])  # Show policy for when the checkpoint is not passed
    plt.plot(path_x, path_y, 'r-', linewidth=2)  # Plot path to checkpoint in red
    plt.plot(path_x2, path_y2, 'b-', linewidth=1.5)  # Plot path to end in blue
    plt.title("Optimal Path Traced on Policy")
    plt.show()

    # Efficiency and Convergence Plots
    efficiency = [1 / step if step > 0 else 0 for step in steps_to_goal]  # Calculate efficiency

    plt.figure(figsize=(12, 10))  # Set figure size for plots

    # Plot Steps to Goal
    plt.subplot(4, 1, 1)
    plt.plot(range(1, len(steps_to_goal) + 1), steps_to_goal, marker='o')
    plt.title('Steps to Goal Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Steps to Goal')
    plt.grid()

    # Plot Efficiency
    plt.subplot(4, 1, 2)
    plt.plot(range(1, len(efficiency) + 1), efficiency, marker='o', color='orange')
    plt.title('Efficiency Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Efficiency (1/Steps)')
    plt.grid()

    # Plot Convergence Speed
    plt.subplot(4, 1, 3)
    plt.plot(range(1, len(convergence_speeds) + 1), convergence_speeds, marker='o', color='green')
    plt.title('Convergence Speed Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Max Value Change (Delta)')
    plt.grid()

    # Plot Learning Stability
    plt.subplot(4, 1, 4)
    plt.plot(range(1, len(learning_stabilities) + 1), learning_stabilities, marker='o', color='purple')
    plt.title('Learning Stability Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Standard Deviation of Updates')
    plt.grid()

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

    return policy  # Return the computed policy

# Run value_iteration with path tracing and print the resulting policy
print(value_iteration())
