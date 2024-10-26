from basics import maze_generator, reward_function
from numpy import ndarray
import numpy as np
from typing import Dict, Callable, Tuple
import matplotlib.pyplot as plt
import os

def value_iteration(maze: ndarray = maze_generator(),
                    reward_function: Callable = reward_function,
                    threshold: float = 0.5,  
                    discount_value: float = 0.9, 
                    checkpoint_position: Tuple[int, int] = (8, 5),  
                    end_position: Tuple[int, int] = (8, 7), 
                    verbose: bool = True, 
                    images: Dict[int, str] = {
                        1: os.path.join("images", "arrow-up.png"),     
                        2: os.path.join("images", "arrow-down.png"),    
                        3: os.path.join("images", "arrow-left.png"),    
                        4: os.path.join("images", "arrow-right.png")}   
                    ) -> ndarray:
    # Initialize value function and policy for whether the checkpoint was passed or not
    value_func: Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}
    policy: Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}

    # Lists to store metrics for analysis
    steps_to_goal = []  
    efficiencies = [] 
    convergence_speeds = []  
    learning_stabilities = []  

    biggest_change = float('inf')  
    passed_checkpoint = (True, False)  
    iteration = 0  

    # Loop until convergence or maximum iterations reached and reset biggest change for the iteration
    while biggest_change > threshold and iteration < 100:
        iteration += 1
        biggest_change = 0.0  
        updates = [] 

        # Iterate over the states of checkpoint passed
        for checkpoint_value in passed_checkpoint:
            # Iterate through all positions in the maze, skipping walls
            for x in range(1, maze.shape[0] - 1):
                for y in range(1, maze.shape[1] - 1):
                    if maze[x, y] == '1':  
                        continue

                    current_pos = (x, y) 
                    best_value = -float("inf")  

                    # Evaluate all possible moves and calculate new possition based on the move direction
                    for move in ['up', 'down', 'left', 'right']:
                        new_pos = (
                            current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],
                            current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move]
                        )
                        # Check if the new position is a wall and revert it to the current possition if it is.
                        if maze[new_pos] == '1':
                            new_pos = current_pos

                        # Calculate the reward for the current move
                        reward = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value,
                                                 checkpoint_position=checkpoint_position, end_position=end_position)

                        value = reward + discount_value * value_func[checkpoint_value][new_pos]

                        # Update best value if current value is better
                        if value > best_value:
                            best_value = value

                    # Calculate the change in value for the current position
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

    # Pplot policies from the computed value function
    for checkpoint_value in passed_checkpoint:
        for x in range(1, maze.shape[0] - 1):
            for y in range(1, maze.shape[1] - 1):
                current_pos = (x, y)

                if maze[x, y] == '1': 
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
                if policy_value in images.keys(): 
                    img = plt.imread(images[policy_value])  
                    
                    plt.imshow(img, extent=[y - 0.5, y + 0.5, x + 0.5, x - 0.5], alpha=0.5, aspect='auto')
                if policy_value != 0:
                    
                    plt.text(y, x, str(int(policy_value)), ha='center', va='center', color='black')

        plt.xlim(-0.5, policy[True].shape[1] - 0.5)  
        plt.ylim(policy[True].shape[0] - 0.5, -0.5)  
        plt.show()

    # Path Tracing
    def trace_path(policy, start_position, end_position, max_steps=100):
        path = []  
        current_pos = start_position  
        steps = 0  
        # Dictionary to map policy values to move directions
        move_dict = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        passed_checkpoint = False  

        # Trace the path until reaching the end position or maximum steps
        while current_pos != end_position and steps < max_steps:
            path.append(current_pos) 
            if current_pos == checkpoint_position:
                passed_checkpoint = True  
            move = int(policy[passed_checkpoint][current_pos])  
            
            # Calculate next position based on the move
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

    # Prepare data for plotting the traced paths
    path_x = [p[1] for p in optimal_path]
    path_y = [p[0] for p in optimal_path]
    path_x2 = [p[1] for p in optimal_path2]
    path_y2 = [p[0] for p in optimal_path2]

    # Visualize the policy with the traced paths
    plt.matshow(policy[False])  
    plt.plot(path_x, path_y, 'r-', linewidth=2)  
    plt.plot(path_x2, path_y2, 'b-', linewidth=1.5)  
    plt.title("Optimal Path Traced on Policy")
    plt.show()

    # Efficiency and Convergence Plots
    efficiency = [1 / step if step > 0 else 0 for step in steps_to_goal]  

    plt.figure(figsize=(12, 10))  

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

    plt.tight_layout() 
    plt.show()

    return policy  


print(value_iteration())
