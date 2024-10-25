from basics import maze_generator, reward_function
from numpy import ndarray
import numpy as np
from typing import Dict, Callable, Tuple
import matplotlib.pyplot as plt
import os

print("Current working directory:", os.getcwd())


def policy_iteration(maze: ndarray = maze_generator(),
                     reward_function: Callable = reward_function,
                     threshold: float = 0.5,
                     discount_value: float = 0.9,
                     checkpoint_position: Tuple[int, int] = (5, 3),
                     end_position: Tuple[int, int] = (8, 7),
                     verbose: bool = True,
                     images: Dict[int, str] = {1: "images\\arrow-up.png",
                                               2: "images\\arrow-down.png",
                                               3: "images\\arrow-left.png",
                                               4: "images\\arrow-right.png"}
                     ) -> Tuple[ndarray, Dict[str, float]]:
    # Initialize the policy and value functions
    policy = np.random.choice(['up', 'down', 'left', 'right'], size=maze.shape)
    value_func = np.zeros(shape=maze.shape)

    # Performance evaluation metrics
    steps_to_goal = []
    efficiencies = []
    convergence_speeds = []
    learning_stabilities = []

    def policy_evaluation():
        nonlocal steps_to_goal, learning_stabilities
        steps = 0
        while True:
            delta = 0
            updates = []  # Track updates for learning stability
            for x in range(1, maze.shape[0] - 1):
                for y in range(1, maze.shape[1] - 1):
                    if maze[x, y] == '1':
                        continue  # Skip walls

                    current_pos = (x, y)
                    old_value = value_func[current_pos]

                    # Get the next position based on the current policy
                    move = policy[current_pos]
                    new_pos = (current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],
                               current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move])

                    if maze[new_pos[0], new_pos[1]] == '1':
                        new_pos = current_pos  # If the new position is a wall, stay in the same position

                    # Compute the new value
                    reward = reward_function(agent_position=current_pos, passed_checkpoint=False,
                                             checkpoint_position=checkpoint_position, end_position=end_position)
                    new_value = reward + discount_value * value_func[new_pos]

                    # Update value function and track the maximum change
                    value_func[current_pos] = new_value
                    delta = max(delta, abs(old_value - new_value))
                    updates.append(abs(old_value - new_value))  # Store update for learning stability

            steps += 1
            steps_to_goal.append(steps)  # Record steps taken
            if delta < threshold:
                break
            convergence_speeds.append(delta)  # Record convergence speed

            # Calculate the standard deviation of updates for learning stability
            if updates:
                learning_stability = np.std(updates)  # Standard deviation of updates
                learning_stabilities.append(learning_stability)

    def policy_improvement():
        policy_stable = True
        for x in range(1, maze.shape[0] - 1):
            for y in range(1, maze.shape[1] - 1):
                if maze[x, y] == '1':
                    continue  # Skip walls

                current_pos = (x, y)
                old_action = policy[current_pos]

                # Choose the best action based on the current value function
                best_value = -float("inf")
                best_action = None
                for move in ['up', 'down', 'left', 'right']:
                    new_pos = (current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],
                               current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move])

                    if maze[new_pos[0], new_pos[1]] == '1':
                        new_pos = current_pos  # If the new position is a wall, stay in the same position

                    reward = reward_function(agent_position=current_pos, passed_checkpoint=False,
                                             checkpoint_position=checkpoint_position, end_position=end_position)
                    value = reward + discount_value * value_func[new_pos]

                    if value > best_value:
                        best_value = value
                        best_action = move

                # Update the policy if the best action changes
                policy[current_pos] = best_action
                if old_action != best_action:
                    policy_stable = False

        return policy_stable

    # Policy iteration loop
    while True:
        policy_evaluation()
        if policy_improvement():
            break

    # Plot the value function heatmap with arrows overlayed
    plt.matshow(value_func, cmap='viridis')  # Plot the heatmap using the value function
    plt.colorbar()

    for x in range(policy.shape[0]):
        for y in range(policy.shape[1]):
            if maze[x, y] == '1':
                continue  # Skip walls

            move = policy[x, y]
            if move == 'up':
                img = plt.imread(images[1])
            elif move == 'down':
                img = plt.imread(images[2])
            elif move == 'left':
                img = plt.imread(images[3])
            elif move == 'right':
                img = plt.imread(images[4])
            else:
                continue

            # Overlay the image with a reduced alpha for better clarity
            plt.imshow(img, extent=[y - 0.4, y + 0.4, x + 0.4, x - 0.4], alpha=0.8, aspect='auto')

    plt.xlim(-0.5, policy.shape[1] - 0.5)
    plt.ylim(policy.shape[0] - 0.5, -0.5)
    plt.title("Policy Iteration - Optimal Policy with Value Function Heatmap")
    plt.show()

    # Performance evaluation plots
    efficiency = [1 / step if step > 0 else 0 for step in steps_to_goal]

    plt.figure(figsize=(12, 10))

    # Plotting Steps to Goal
    plt.subplot(4, 1, 1)
    plt.plot(steps_to_goal, marker='o')
    plt.title('Steps to Goal Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Steps to Goal')
    plt.grid()

    # Plotting Efficiency
    plt.subplot(4, 1, 2)
    plt.plot(efficiency, marker='o', color='orange')
    plt.title('Efficiency Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Efficiency (1/Steps)')
    plt.grid()

    # Plotting Convergence Speed
    plt.subplot(4, 1, 3)
    plt.plot(convergence_speeds, marker='o', color='green')
    plt.title('Convergence Speed Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Max Value Change (Delta)')
    plt.grid()

    # Plotting Learning Stability
    plt.subplot(4, 1, 4)
    plt.plot(learning_stabilities, marker='o', color='purple')
    plt.title('Learning Stability Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Standard Deviation of Updates')
    plt.grid()

    plt.tight_layout()
    plt.show()

    return policy


print(policy_iteration())
