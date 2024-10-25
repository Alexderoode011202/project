


from basics import maze_generator, reward_function
from numpy import ndarray
import numpy as np
from typing import Dict, Callable, Tuple
import matplotlib.pyplot as plt

def value_iteration(maze                : ndarray           = maze_generator(), 
                    reward_function     : Callable          = reward_function, 
                    threshold           : float             = 0.5, 
                    discount_value      : float             = 0.9,
                    checkpoint_position : Tuple[int, int]   = (5, 3), 
                    end_position        : Tuple[int, int]   = (8, 7),
                    verbose             : bool              = True,
                    images              : Dict[int, str]    = {1: "images\\arrow-up.png",  # Image for policy 1
                                                                  2: "images\\arrow-down.png",  # Image for policy 2
                                                                  3: "images\\arrow-left.png",  # Image for policy 3
                                                                  4: "images\\arrow-right.png"}  # Image for policy 4
                    ) -> ndarray:
    
    value_func  : Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}
    policy      : Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}
    
    biggest_change      : float             = float('Inf')
    passed_checkpoint   : Tuple[bool, bool] = (True, False)
    iteration           : int               = 0
    while iteration < 25:
        iteration += 1
        biggest_change: float = 0.0
        
        for checkpoint_value in passed_checkpoint:      #   \
            for x in range(1, maze.shape[0] - 1):      #   |---- Go over all possible states
                for y in range(1, maze.shape[1] - 1):  #   /
                    
                    if verbose:
                        print(f"Iteration: {iteration} at position ({x}, {y})")
                        
                    current_pos : Tuple[int,int]    = (x,y)
                    
                    if maze[x,y] == '1':              # walls cannot be reached so there is no point in going over those
                        continue
                    
                    if maze[current_pos] == "E":
                        value_func[checkpoint_value][current_pos] = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position)
                        continue
                    
                    best_value  : float = -float("inf")
                    for move in ['up', 'down', 'left', 'right']:
                        
                        new_pos = (current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],     #   \ Go over all moves
                                    current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move])    #   / that are feasible in a state
                        
                        if maze[new_pos[0], new_pos[1]] == '1':
                            new_pos = current_pos
                            
                        value = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position) \
                            + discount_value * value_func[checkpoint_value][new_pos]
                    
                        if value > best_value:
                            best_value = value
                        
                    old_value   : float = value_func[checkpoint_value][new_pos]
                    change      : float = best_value - old_value
                    
                    if change > biggest_change:
                        biggest_change = change

                    value_func[checkpoint_value][current_pos] = best_value
                    
                    if verbose:
                        print(f"update made at position: {x}, {y} with value: {best_value} at iteration: {iteration} with change: {change}")
                        
                         
    
    # Plot the heatmap with image overlay for value_func[True]
    plt.matshow(value_func[True])
    plt.colorbar()
    plt.show()
    
    # Plot the heatmap with image overlay for value_func[False]
    plt.matshow(value_func[False])
    plt.colorbar()
    plt.show()
                        
    # once we are done with iterating over the values:
    for checkpoint_value in passed_checkpoint:
            for x in range(1, maze.shape[0] - 1):
                for y in range(1, maze.shape[1] - 1):
                    current_pos: Tuple[int, int] = (x,y)
                    
                    
                    if maze[current_pos] == '1':
                        continue
                    
                    elif maze[current_pos] == "E" and checkpoint_value:
                        policy[checkpoint_value][current_pos] = 5
                        continue
                        
                    
                    best_move   : str = ""
                    best_value  : int = -float("inf")
                    
                    for move in ['up', 'down', 'left', 'right']:
                        new_pos = (current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],
                                    current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move])
                        
                        if maze[new_pos] == "1":  
                            new_pos = current_pos
                            
                        value = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position) \
                        + discount_value * value_func[checkpoint_value][new_pos]
                        
                        if value > best_value:
                            best_value = value
                            best_move = move
                            
                    if best_move == "up":
                        policy[checkpoint_value][current_pos] = 1
                    elif best_move == "down":
                        policy[checkpoint_value][current_pos] = 2
                    elif best_move == "left":
                        policy[checkpoint_value][current_pos] = 3
                    elif best_move == "right":
                        policy[checkpoint_value][current_pos] = 4
                        



    # Create a color-coded plot for policy[True] using default colors
    plt.matshow(policy[True])  # Automatically color-coded based on policy values
    plt.colorbar()  # Show colorbar to indicate what each color means

    # Overlay different images on policy squares for policy[True]
    for x in range(policy[True].shape[0]):
        for y in range(policy[True].shape[1]):
            policy_value = policy[True][x, y]
            if policy_value in images.keys():  # Check if policy value has an associated image
                img = plt.imread(images[policy_value])  # Load the corresponding image
                plt.imshow(img, extent=[y - 0.5, y + 0.5, x + 0.5, x - 0.5], alpha=0.5, aspect='auto')  # Overlay image in the correct grid cell
            if policy_value != 0:
                plt.text(y, x, str(int(policy_value)), ha='center', va='center', color='black')  # Display policy value

    plt.xlim(-0.5, policy[True].shape[1] - 0.5)  # Set x-axis limits
    plt.ylim(policy[True].shape[0] - 0.5, -0.5)
    plt.show()

    # Create a color-coded plot for policy[False] using default colors
    plt.matshow(policy[False])  # Automatically color-coded based on policy values
    plt.colorbar()  # Show colorbar

    # Overlay different images on policy squares for policy[False]
    for x in range(policy[False].shape[0]):
        for y in range(policy[False].shape[1]):
            policy_value = policy[False][x, y]
            if policy_value in images.keys():  # Check if policy value has an associated image
                img = plt.imread(images[policy_value])  # Load the corresponding image
                plt.imshow(img, extent=[y - 0.5, y + 0.5, x + 0.5, x - 0.5], alpha=0.5, aspect='auto')  # Overlay image in the correct grid cell
            if policy_value != 0:
                plt.text(y, x, str(int(policy_value)), ha='center', va='center', color='black')  # Display policy value

    plt.xlim(-0.5, policy[False].shape[1] - 0.5)  # Adjust axis limits for consistency
    plt.ylim(policy[False].shape[0] - 0.5, -0.5)
    plt.show()
    return policy

print(value_iteration())
