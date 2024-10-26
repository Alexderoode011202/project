
# from basics import maze_generator, reward_function
# from numpy import ndarray
# import numpy as np
# from typing import Dict, Callable, Tuple
# import matplotlib.pyplot as plt

# def value_iteration(maze                : ndarray           = maze_generator(),
#                     reward_function     : Callable          = reward_function,
#                     threshold           : float             = 0.5,
#                     discount_value      : float             = 0.9,
#                     checkpoint_position : Tuple[int, int]   = (5, 3),
#                     end_position        : Tuple[int, int]   = (8, 7),
#                     verbose             : bool              = True,
#                     images              : Dict[int, str]    = {1: "images\\arrow-up.png",  # Image for policy 1
#                                                                   2: "images\\arrow-down.png",  # Image for policy 2
#                                                                   3: "images\\arrow-left.png",  # Image for policy 3
#                                                                   4: "images\\arrow-right.png"}  # Image for policy 4
#                     ) -> ndarray:

#     value_func  : Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}
#     policy      : Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}

#     biggest_change      : float             = float('Inf')
#     passed_checkpoint   : Tuple[bool, bool] = (True, False)
#     iteration           : int               = 0
#     while iteration < 25:
#         iteration += 1
#         biggest_change: float = 0.0

#         for checkpoint_value in passed_checkpoint:      #   \
#             for x in range(1, maze.shape[0] - 1):      #   |---- Go over all possible states
#                 for y in range(1, maze.shape[1] - 1):  #   /

#                     if verbose:
#                         print(f"Iteration: {iteration} at position ({x}, {y})")

#                     current_pos : Tuple[int,int]    = (x,y)

#                     if maze[x,y] == '1':              # walls cannot be reached so there is no point in going over those
#                         continue

#                     if maze[current_pos] == "E":
#                         value_func[checkpoint_value][current_pos] = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position)
#                         continue

#                     best_value  : float = -float("inf")
#                     for move in ['up', 'down', 'left', 'right']:

#                         new_pos = (current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],     #   \ Go over all moves
#                                     current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move])    #   / that are feasible in a state

#                         if maze[new_pos[0], new_pos[1]] == '1':
#                             new_pos = current_pos

#                         value = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position) \
#                             + discount_value * value_func[checkpoint_value][new_pos]

#                         if value > best_value:
#                             best_value = value

#                     old_value   : float = value_func[checkpoint_value][new_pos]
#                     change      : float = best_value - old_value

#                     if change > biggest_change:
#                         biggest_change = change

#                     value_func[checkpoint_value][current_pos] = best_value

#                     if verbose:
#                         print(f"update made at position: {x}, {y} with value: {best_value} at iteration: {iteration} with change: {change}")



#     # Plot the heatmap with image overlay for value_func[True]
#     plt.matshow(value_func[True])
#     plt.colorbar()
#     plt.show()

#     # Plot the heatmap with image overlay for value_func[False]
#     plt.matshow(value_func[False])
#     plt.colorbar()
#     plt.show()

#     # once we are done with iterating over the values:
#     for checkpoint_value in passed_checkpoint:
#             for x in range(1, maze.shape[0] - 1):
#                 for y in range(1, maze.shape[1] - 1):
#                     current_pos: Tuple[int, int] = (x,y)


#                     if maze[current_pos] == '1':
#                         continue

#                     elif maze[current_pos] == "E" and checkpoint_value:
#                         policy[checkpoint_value][current_pos] = 5
#                         continue


#                     best_move   : str = ""
#                     best_value  : int = -float("inf")

#                     for move in ['up', 'down', 'left', 'right']:
#                         new_pos = (current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],
#                                     current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move])

#                         if maze[new_pos] == "1":
#                             new_pos = current_pos

#                         value = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position) \
#                         + discount_value * value_func[checkpoint_value][new_pos]

#                         if value > best_value:
#                             best_value = value
#                             best_move = move

#                     if best_move == "up":
#                         policy[checkpoint_value][current_pos] = 1
#                     elif best_move == "down":
#                         policy[checkpoint_value][current_pos] = 2
#                     elif best_move == "left":
#                         policy[checkpoint_value][current_pos] = 3
#                     elif best_move == "right":
#                         policy[checkpoint_value][current_pos] = 4




#     # Create a color-coded plot for policy[True] using default colors
#     plt.matshow(policy[True])  # Automatically color-coded based on policy values
#     plt.colorbar()  # Show colorbar to indicate what each color means

#     # Overlay different images on policy squares for policy[True]
#     for x in range(policy[True].shape[0]):
#         for y in range(policy[True].shape[1]):
#             policy_value = policy[True][x, y]
#             if policy_value in images.keys():  # Check if policy value has an associated image
#                 img = plt.imread(images[policy_value])  # Load the corresponding image
#                 plt.imshow(img, extent=[y - 0.5, y + 0.5, x + 0.5, x - 0.5], alpha=0.5, aspect='auto')  # Overlay image in the correct grid cell
#             if policy_value != 0:
#                 plt.text(y, x, str(int(policy_value)), ha='center', va='center', color='black')  # Display policy value

#     plt.xlim(-0.5, policy[True].shape[1] - 0.5)  # Set x-axis limits
#     plt.ylim(policy[True].shape[0] - 0.5, -0.5)
#     plt.show()

#     # Create a color-coded plot for policy[False] using default colors
#     plt.matshow(policy[False])  # Automatically color-coded based on policy values
#     plt.colorbar()  # Show colorbar

#     # Overlay different images on policy squares for policy[False]
#     for x in range(policy[False].shape[0]):
#         for y in range(policy[False].shape[1]):
#             policy_value = policy[False][x, y]
#             if policy_value in images.keys():  # Check if policy value has an associated image
#                 img = plt.imread(images[policy_value])  # Load the corresponding image
#                 plt.imshow(img, extent=[y - 0.5, y + 0.5, x + 0.5, x - 0.5], alpha=0.5, aspect='auto')  # Overlay image in the correct grid cell
#             if policy_value != 0:
#                 plt.text(y, x, str(int(policy_value)), ha='center', va='center', color='black')  # Display policy value

#     plt.xlim(-0.5, policy[False].shape[1] - 0.5)  # Adjust axis limits for consistency
#     plt.ylim(policy[False].shape[0] - 0.5, -0.5)
#     plt.show()
#     return policy

# print(value_iteration())
























# from basics import maze_generator, reward_function
# from numpy import ndarray
# import numpy as np
# from typing import Dict, Callable, Tuple
# import matplotlib.pyplot as plt
# import os

# def value_iteration(maze: ndarray = maze_generator(),
#                     reward_function: Callable = reward_function,
#                     threshold: float = 0.5,
#                     discount_value: float = 0.9,
#                     checkpoint_position: Tuple[int, int] = (5, 3),
#                     end_position: Tuple[int, int] = (8, 7),
#                     verbose: bool = True,
#                     images: Dict[int, str] = {1: os.path.join("images", "arrow-up.png"),
#                                               2: os.path.join("images", "arrow-down.png"),
#                                               3: os.path.join("images", "arrow-left.png"),
#                                               4: os.path.join("images", "arrow-right.png")}
#                     ) -> Tuple[ndarray, Dict[str, float]]:

#     move_dict = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
#     policy_to_move = {1: "up", 2: "down", 3: "left", 4: "right"}

#     value_func: Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}
#     policy: Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}

#     biggest_change = float('Inf')
#     passed_checkpoint = (True, False)
#     iteration = 0

#     # Lists to track convergence speed and learning stability
#     convergence_speed = []
#     stability_metrics = []

#     while iteration < 25:
#         iteration += 1
#         biggest_change = 0.0
#         changes = []  # Store changes for stability metrics

#         for checkpoint_value in passed_checkpoint:
#             for x in range(1, maze.shape[0] - 1):
#                 for y in range(1, maze.shape[1] - 1):
#                     if maze[x, y] == '1':
#                         continue
#                     current_pos = (x, y)
#                     if maze[current_pos] == "E":
#                         value_func[checkpoint_value][current_pos] = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position)
#                         continue

#                     best_value = -float("inf")
#                     for move in ['up', 'down', 'left', 'right']:
#                         new_pos = (current_pos[0] + move_dict[move][0], current_pos[1] + move_dict[move][1])
#                         if maze[new_pos[0], new_pos[1]] == '1':
#                             new_pos = current_pos
#                         value = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position) \
#                                 + discount_value * value_func[checkpoint_value][new_pos]
#                         if value > best_value:
#                             best_value = value

#                     old_value = value_func[checkpoint_value][current_pos]
#                     change = best_value - old_value
#                     changes.append(change)  # Record change for stability analysis

#                     if abs(change) > biggest_change:
#                         biggest_change = abs(change)
#                     value_func[checkpoint_value][current_pos] = best_value

#         # Record the maximum change for convergence speed analysis
#         convergence_speed.append(biggest_change)

#         # Calculate stability as standard deviation of changes
#         stability_metrics.append(np.std(changes))

#         # Check for convergence
#         if biggest_change < threshold:
#             print(f"Converged at iteration {iteration}")
#             break

#     # Plot the heatmap with image overlay for value_func[True]
#     plt.matshow(value_func[True])
#     plt.colorbar()
#     plt.show()

#     # Plot the heatmap with image overlay for value_func[False]
#     plt.matshow(value_func[False])
#     plt.colorbar()
#     plt.show()

#     for checkpoint_value in passed_checkpoint:
#         for x in range(1, maze.shape[0] - 1):
#             for y in range(1, maze.shape[1] - 1):
#                 current_pos = (x, y)
#                 if maze[current_pos] == '1':
#                     continue
#                 elif maze[current_pos] == "E" and checkpoint_value:
#                     policy[checkpoint_value][current_pos] = 5
#                     continue
#                 best_move = ""
#                 best_value = -float("inf")
#                 for move in ['up', 'down', 'left', 'right']:
#                     new_pos = (current_pos[0] + move_dict[move][0], current_pos[1] + move_dict[move][1])
#                     if maze[new_pos] == "1":
#                         new_pos = current_pos
#                     value = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position) \
#                             + discount_value * value_func[checkpoint_value][new_pos]
#                     if value > best_value:
#                         best_value = value
#                         best_move = move
#                 if best_move == "up":
#                     policy[checkpoint_value][current_pos] = 1
#                 elif best_move == "down":
#                     policy[checkpoint_value][current_pos] = 2
#                 elif best_move == "left":
#                     policy[checkpoint_value][current_pos] = 3
#                 elif best_move == "right":
#                     policy[checkpoint_value][current_pos] = 4

#     # Plotting policy maps
#     for checkpoint in [True, False]:
#         plt.matshow(policy[checkpoint])
#         plt.colorbar()
#         for x in range(policy[checkpoint].shape[0]):
#             for y in range(policy[checkpoint].shape[1]):
#                 policy_value = policy[checkpoint][x, y]
#                 if policy_value in images.keys():
#                     img = plt.imread(images[policy_value])
#                     plt.imshow(img, extent=[y - 0.5, y + 0.5, x + 0.5, x - 0.5], alpha=0.5, aspect='auto')
#                 if policy_value != 0:
#                     plt.text(y, x, str(int(policy_value)), ha='center', va='center', color='black')
#         plt.xlim(-0.5, policy[checkpoint].shape[1] - 0.5)
#         plt.ylim(policy[checkpoint].shape[0] - 0.5, -0.5)
#         plt.show()

#     # Function to evaluate performance and return metrics
#     def evaluate_performance(policy, checkpoint_position, end_position, max_steps=100):
#         agent_position = (1, 1)
#         reached_checkpoint = False
#         steps = 0
#         steps_list = []
#         efficiency_list = []

#         while agent_position != end_position and steps < max_steps:
#             move = int(policy[reached_checkpoint][agent_position])
#             move_direction = policy_to_move[move]
#             agent_position = \
#             (agent_position[0] + move_dict[move_direction][0], agent_position[1] + move_dict[move_direction][1])
#             steps += 1
#             steps_list.append(steps)
#             efficiency_list.append(1 / steps if steps > 0 else 0)
#             if agent_position == checkpoint_position:
#                 reached_checkpoint = True

#         reached_goal = agent_position == end_position
#         return {
#             "steps_to_goal": steps_list,
#             "efficiency": efficiency_list,
#             "reached_checkpoint": reached_checkpoint,
#             "reached_goal": reached_goal
#         }

#     # Evaluate performance metrics
#     performance_metrics = evaluate_performance(policy, checkpoint_position, end_position)

#     # Plotting performance metrics
#     plt.plot(performance_metrics["steps_to_goal"], label="Steps to Goal")
#     plt.xlabel("Steps")
#     plt.ylabel("Count")
#     plt.title("Steps to Reach Goal Over Time")
#     plt.legend()
#     plt.show()

#     plt.plot(performance_metrics["efficiency"], label="Efficiency")
#     plt.xlabel("Steps")
#     plt.ylabel("Efficiency (1 / steps)")
#     plt.title("Efficiency Over Time")
#     plt.legend()
#     plt.show()

#     # Plot convergence speed
#     plt.plot(convergence_speed, label="Convergence Speed")
#     plt.xlabel("Iteration")
#     plt.ylabel("Max Change in Value Function")
#     plt.title("Convergence Speed Over Iterations")
#     plt.axhline(y=threshold, color='r', linestyle='--', label='Convergence Threshold')
#     plt.legend()
#     plt.show()

#     # Plot learning stability
#     plt.plot(stability_metrics, label="Learning Stability (Std Dev of Changes)")
#     plt.xlabel("Iteration")
#     plt.ylabel("Standard Deviation of Changes")
#     plt.title("Learning Stability Over Iterations")
#     plt.axhline(y=np.mean(stability_metrics) + 0.1 * np.mean(stability_metrics), color='g', linestyle='--',
#                 label='Average Stability')
#     plt.legend()
#     plt.show()

#     print("Performance Metrics:", performance_metrics)
#     return policy, performance_metrics


# # Run value_iteration function
# print(value_iteration())



from basics import maze_generator, reward_function
from numpy import ndarray
import numpy as np
from typing import Dict, Callable, Tuple
import matplotlib.pyplot as plt

def value_iteration(maze: ndarray = maze_generator(),
                    reward_function: Callable = reward_function,
                    threshold: float = 0.5,
                    discount_value: float = 0.9,
                    checkpoint_position: Tuple[int, int] = (8, 5),
                    end_position: Tuple[int, int] = (8, 7),
                    verbose: bool = True,
                    images: Dict[int, str] = {1: "images\\arrow-up.png",  # Image for policy 1
                                               2: "images\\arrow-down.png",  # Image for policy 2
                                               3: "images\\arrow-left.png",  # Image for policy 3
                                               4: "images\\arrow-right.png"}  # Image for policy 4
                    ) -> ndarray:
    value_func: Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}
    policy: Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}

    steps_to_goal = []
    efficiencies = []
    convergence_speeds = []
    learning_stabilities = []

    biggest_change = float('inf')
    passed_checkpoint = (True, False)
    iteration = 0

    while biggest_change > threshold and iteration < 100:
        iteration += 1
        biggest_change = 0.0
        updates = []

        for checkpoint_value in passed_checkpoint:
            for x in range(1, maze.shape[0] - 1):
                for y in range(1, maze.shape[1] - 1):
                    if maze[x, y] == '1':  # Skip walls
                        continue

                    current_pos = (x, y)
                    best_value = -float("inf")
                    for move in ['up', 'down', 'left', 'right']:
                        new_pos = (
                            current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],
                            current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move]
                        )
                        if maze[new_pos] == '1':
                            new_pos = current_pos

                        reward = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value,
                                                 checkpoint_position=checkpoint_position, end_position=end_position)
                        value = reward + discount_value * value_func[checkpoint_value][new_pos]

                        if value > best_value:
                            best_value = value

                    old_value = value_func[checkpoint_value][current_pos]
                    change = abs(best_value - old_value)
                    updates.append(change)
                    biggest_change = max(biggest_change, change)
                    value_func[checkpoint_value][current_pos] = best_value

        convergence_speeds.append(biggest_change)
        learning_stabilities.append(np.std(updates))

    # Plot Value Functions
    for checkpoint_value in passed_checkpoint:
        plt.matshow(value_func[checkpoint_value])
        plt.title(f"Value Function - Checkpoint Passed: {checkpoint_value}")
        plt.colorbar()
        plt.show()

    # Derive and plot policies from the computed value function
    for checkpoint_value in passed_checkpoint:
        for x in range(1, maze.shape[0] - 1):
            for y in range(1, maze.shape[1] - 1):
                current_pos = (x, y)

                if maze[x, y] == '1':  # Skip walls
                    continue

                best_move = ""
                best_value = -float("inf")
                for move in ['up', 'down', 'left', 'right']:
                    new_pos = (
                        current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],
                        current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move]
                    )
                    if maze[new_pos] == '1':
                        new_pos = current_pos

                    reward = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value,
                                             checkpoint_position=checkpoint_position, end_position=end_position)
                    value = reward + discount_value * value_func[checkpoint_value][new_pos]

                    if value > best_value:
                        best_value = value
                        best_move = move

                policy_value = {"up": 1, "down": 2, "left": 3, "right": 4}.get(best_move, 0)
                policy[checkpoint_value][current_pos] = policy_value

    # Policy Visualization
    for checkpoint_value in passed_checkpoint:
        plt.matshow(policy[checkpoint_value])
        plt.title(f"Policy {checkpoint_value} - Best Policy Move")
        plt.colorbar()

        # Overlay different images on policy squares
        for x in range(policy[checkpoint_value].shape[0]):
            for y in range(policy[checkpoint_value].shape[1]):
                policy_value = policy[checkpoint_value][x, y]
                if policy_value in images.keys():  # Check if policy value has an associated image
                    img = plt.imread(images[policy_value])  # Load the corresponding image
                    plt.imshow(img, extent=[y - 0.5, y + 0.5, x + 0.5, x - 0.5], alpha=0.5, aspect='auto')  # Overlay image in the correct grid cell
                if policy_value != 0:
                    plt.text(y, x, str(int(policy_value)), ha='center', va='center', color='black')  # Display policy value

        plt.xlim(-0.5, policy[True].shape[1] - 0.5)  # Set x-axis limits
        plt.ylim(policy[True].shape[0] - 0.5, -0.5)
        plt.show()


        plt.show()

    # Path Tracing
    def trace_path(policy, start_position, end_position, max_steps=100):
        path = []
        current_pos = start_position
        steps = 0
        move_dict = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        passed_checkpoint = False
        while current_pos != end_position and steps < max_steps:
            path.append(current_pos)
            if current_pos == checkpoint_position:
                passed_checkpoint = True
            move = int(policy[passed_checkpoint][current_pos])
            next_pos = (current_pos[0] + move_dict.get(move, (0, 0))[0],
                        current_pos[1] + move_dict.get(move, (0, 0))[1])
            if maze[next_pos] == '1':
                break
            current_pos = next_pos
            steps += 1

        path.append(end_position)
        return path

    start_position = (1, 1)
    optimal_path = trace_path(policy, start_position, checkpoint_position)
    optimal_path2 = trace_path(policy, checkpoint_position, end_position)

    path_x = [p[1] for p in optimal_path]
    path_y = [p[0] for p in optimal_path]
    path_x2 = [p[1] for p in optimal_path2]
    path_y2 = [p[0] for p in optimal_path2]

    plt.matshow(policy[False])
    plt.plot(path_x, path_y, 'r-', linewidth=2)
    plt.plot(path_x2, path_y2, 'b-', linewidth=1.5)
    plt.title("Optimal Path Traced on Policy")
    plt.show()

    # Efficiency and Convergence Plots
    efficiency = [1 / step if step > 0 else 0 for step in steps_to_goal]

    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(range(1, len(steps_to_goal) + 1), steps_to_goal, marker='o')
    plt.title('Steps to Goal Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Steps to Goal')
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(range(1, len(efficiency) + 1), efficiency, marker='o', color='orange')
    plt.title('Efficiency Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Efficiency (1/Steps)')
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(range(1, len(convergence_speeds) + 1), convergence_speeds, marker='o', color='green')
    plt.title('Convergence Speed Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Max Value Change (Delta)')
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(range(1, len(learning_stabilities) + 1), learning_stabilities, marker='o', color='purple')
    plt.title('Learning Stability Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Standard Deviation of Updates')
    plt.grid()

    plt.tight_layout()
    plt.show()

    return policy

# Run value_iteration with path tracing
print(value_iteration())
