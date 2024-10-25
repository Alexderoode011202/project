from basics import maze_generator, reward_function
from numpy import ndarray
import numpy as np
from typing import Dict, Callable, Tuple
import matplotlib.pyplot as plt
import os

def get_possible_actions(maze: ndarray, current_pos: Tuple[int, int]) -> list:
    possible_actions = []

    directions = [(-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4)]
    for dx, dy, action in directions:
        new_x, new_y = current_pos[0] + dx, current_pos[1] + dy
        if maze[new_x, new_y] != '1':
            possible_actions.append(action)

    return possible_actions

def get_new_state(maze: ndarray, current_pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    move_map = {
        1: (-1, 0),  # Up
        2: (1, 0),   # Down
        3: (0, -1),  # Left
        4: (0, 1)    # Right
    }
    move = move_map.get(action, (0, 0))
    new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
    return new_pos

def policy_iteration(maze                : ndarray           = maze_generator(), 
                    reward_function     : Callable          = reward_function, 
                    threshold           : float             = 0.5, 
                    discount_value      : float             = 0.9,
                    checkpoint_position : Tuple[int, int]   = (5, 3), 
                    end_position        : Tuple[int, int]   = (8, 7),
                    verbose             : bool              = True,
                    images              : Dict[int, str]    = {1: os.path.join("images", "arrow-up.png"),  # Image for policy 1
                                                                  2: os.path.join("images", "arrow-down.png"),  # Image for policy 2
                                                                  3: os.path.join("images", "arrow-left.png"),  # Image for policy 3
                                                                  4: os.path.join("images", "arrow-right.png")}  # Image for policy 4
                    ) -> ndarray:
    value_func  : Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}
    policy      : Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}

    for key in policy.keys():
        for x in range(1, maze.shape[0] - 1):
            for y in range(1, maze.shape[1] - 1):
                if maze[x,y] == '1':
                    continue
                policy[key][x,y] = np.random.choice(get_possible_actions(maze, (x,y)))
                # 1: up, 2: down, 3: left, 4: right

    passed_checkpoint   : Tuple[bool, bool] = (True, False)
    iteration           : int               = 0

    while True: # Policy iteration
        iteration += 1
        while True: # Policy evaluation
            delta = 0.0
            for checkpoint_value in passed_checkpoint:
                for x in range(1, maze.shape[0] - 1):
                    for y in range(1, maze.shape[1] - 1):
                        current_pos = (x,y)
                        if maze[x,y] == '1':
                            continue

                        temp = value_func[checkpoint_value][current_pos]
                        reward = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position)
                        action = policy[checkpoint_value][current_pos]
                        
                        new_pos = get_new_state(maze, current_pos, action)

                        value_func[checkpoint_value][current_pos] = reward + discount_value * value_func[checkpoint_value][new_pos]
                        delta = max(delta, abs(temp - value_func[checkpoint_value][current_pos]))
            if delta < threshold:
                break
        
        # Policy improvement
        policy_stable = True
        for checkpoint_value in passed_checkpoint:
            for x in range(1, maze.shape[0] - 1):
                for y in range(1, maze.shape[1] - 1):
                    current_pos = (x,y)
                    if maze[x,y] == '1':
                        continue

                    old_action = policy[checkpoint_value][current_pos]
                    best_action = 0
                    best_value = -float("inf")
                    for action in get_possible_actions(maze, current_pos):
                        new_pos = get_new_state(maze, current_pos, action)

                        value = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position) + discount_value * value_func[checkpoint_value][new_pos]
                        if value > best_value:
                            best_value = value
                            best_action = action

                    policy[checkpoint_value][current_pos] = best_action
                    if old_action != best_action:
                        policy_stable = False

        if policy_stable:
            break

    if verbose:
        # generate plots for the value function and policy
        for checkpoint_value in passed_checkpoint:
            plt.matshow(value_func[checkpoint_value])
            plt.title(f"Value Function - Checkpoint Passed: {checkpoint_value}")
            plt.colorbar()
            plt.show()

        # overlay images on the policy
            plt.matshow(policy[checkpoint_value])
            plt.colorbar()
            plt.title(f"Policy - Checkpoint Passed: {checkpoint_value}")
            for x in range(1, maze.shape[0] - 1):
                for y in range(1, maze.shape[1] - 1):
                    policy_value = policy[checkpoint_value][x, y]
                    if policy_value in images.keys():  # Check if policy value has an associated image
                        img = plt.imread(images[policy_value])  # Load the corresponding image
                        plt.imshow(img, extent=[y - 0.5, y + 0.5, x + 0.5, x - 0.5], alpha=0.5, aspect='auto')  # Overlay image in the correct grid cell
                    if policy_value != 0:
                        plt.text(y, x, str(int(policy_value)), ha='center', va='center', color='black')  # Display policy value

            plt.xlim(-0.5, policy[checkpoint_value].shape[1] - 0.5)  # Adjust axis limits for consistency
            plt.ylim(policy[checkpoint_value].shape[0] - 0.5, -0.5)
            plt.show()
        print(f"Policy Iteration converged after {iteration} iterations")

    return policy


policy_iteration()
