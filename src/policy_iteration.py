from basics import maze_generator, reward_function
from numpy import ndarray
import numpy as np
from typing import Dict, Callable, Tuple
import matplotlib.pyplot as plt
import os

# Function to determine possible actions based on current position in the maze
def get_possible_actions(current_pos: Tuple[int, int]) -> list:
    possible_actions = []
    directions = [(-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4)]
    for dx, dy, action in directions:
        new_x, new_y = current_pos[0] + dx, current_pos[1] + dy
        if MAZE[new_x, new_y] != '1':
            possible_actions.append(action)
    return possible_actions

# Calculate new state after taking an action from the current position
def get_new_state(current_pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    move_map = {
        1: (-1, 0),  
        2: (1, 0),  
        3: (0, -1), 
        4: (0, 1)    
    }
    move = move_map.get(action, (0, 0))
    # Calculate new position based on the action
    new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
    return new_pos

# Policy Iteration function to compute the optimal policy
def policy_iteration(verbose: bool = True,) -> Dict[bool, ndarray]:
    value_func: Dict[bool, ndarray] = {True: np.zeros(shape=MAZE.shape), False: np.zeros(shape=MAZE.shape)}
    policy: Dict[bool, ndarray] = {True: np.zeros(shape=MAZE.shape), False: np.zeros(shape=MAZE.shape)}

    steps_to_goal = []         
    efficiencies = []          
    convergence_speeds = []     
    learning_stabilities = []   

    # Randomly initialize policy for each position and checkpoint state
    for key in policy.keys():
        for x in range(1, MAZE.shape[0] - 1):
            for y in range(1, MAZE.shape[1] - 1):
                if MAZE[x, y] == '1':
                    continue
                policy[key][x, y] = np.random.choice(get_possible_actions((x, y)))

    passed_checkpoint: Tuple[bool, bool] = (True, False)  
    iteration: int = 0

    # Outer loop: Policy iteration loop
    while True:
        iteration += 1
        steps = 0
        delta = 0.0

        # Policy Evaluation that updates the value function based on current policy
        while True:
            delta = 0.0
            updates = []  
            for checkpoint_value in passed_checkpoint:
                for x in range(1, MAZE.shape[0] - 1):
                    for y in range(1, MAZE.shape[1] - 1):
                        current_pos = (x, y)
                        if MAZE[x, y] == '1':
                            continue

                        # Save old value for comparison after update and calculate reward for the current state based on the position and checkpoint status
                        temp = value_func[checkpoint_value][current_pos]
                        reward = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value,
                                                 checkpoint_position=CHECKPOINT, end_position=END)
                        action = policy[checkpoint_value][current_pos]
                        new_pos = get_new_state(current_pos, action)

                        value_func[checkpoint_value][current_pos] = reward + GAMMA * value_func[checkpoint_value][new_pos]
                        delta = max(delta, abs(temp - value_func[checkpoint_value][current_pos]))
                        updates.append(abs(temp - value_func[checkpoint_value][current_pos]))

            if delta < THRESHOLD: 
                break
            
        # Record convergence speed for current iteration
        convergence_speeds.append(delta)

        # Track learning stability via standard deviation of updates
        if updates:
            learning_stability = np.std(updates)
            learning_stabilities.append(learning_stability)

        # Policy Improvement: Update policy based on current value function
        policy_stable = True 
        for checkpoint_value in passed_checkpoint:
            for x in range(1, MAZE.shape[0] - 1):
                for y in range(1, MAZE.shape[1] - 1):
                    current_pos = (x, y)
                    if MAZE[x, y] == '1':
                        continue

                    old_action = policy[checkpoint_value][current_pos]
                    best_action = 0
                    best_value = -float("inf")

                    # Evaluate each possible action to find the one with the highest value
                    for action in get_possible_actions(current_pos):
                        new_pos = get_new_state(current_pos, action)
                        value = reward_function(agent_position=current_pos, passed_checkpoint=checkpoint_value,
                                                checkpoint_position=CHECKPOINT, end_position=END) + \
                                GAMMA * value_func[checkpoint_value][new_pos]
                        if value > best_value:
                            best_value = value
                            best_action = action

                    # Update the policy with the best action
                    policy[checkpoint_value][current_pos] = best_action
                    if old_action != best_action:
                        policy_stable = False

        if policy_stable:  
            steps_to_goal.append(iteration) 
            break

    if verbose:
        plot_results(
            steps_to_goal, 
            convergence_speeds, 
            learning_stabilities, 
            value_func, 
            policy
        )

    return policy

# Function to plot results after policy iteration
def plot_results(steps_to_goal, convergence_speeds, learning_stabilities, value_func, policy):
    IMAGES: Dict[int, str] = {
        1: os.path.join("images", "arrow-up.png"),  
        2: os.path.join("images", "arrow-down.png"),  
        3: os.path.join("images", "arrow-left.png"),  
        4: os.path.join("images", "arrow-right.png")
    }

    # Plot value function and policy for each checkpoint status
    for checkpoint_value in policy.keys():
        plt.matshow(value_func[checkpoint_value])
        plt.title(f"Value Function - Checkpoint Passed: {checkpoint_value}")
        plt.colorbar()
        plt.show()

        plt.matshow(policy[checkpoint_value])
        plt.colorbar()
        plt.title(f"Policy - Checkpoint Passed: {checkpoint_value}")

        for x in range(1, MAZE.shape[0] - 1):
            for y in range(1, MAZE.shape[1] - 1):
                policy_value = policy[checkpoint_value][x, y]
                if policy_value in IMAGES.keys():
                    img = plt.imread(IMAGES[policy_value])
                    plt.imshow(img, extent=[y - 0.5, y + 0.5, x + 0.5, x - 0.5], alpha=0.5, aspect='auto')
                if policy_value != 0:
                    plt.text(y, x, str(int(policy_value)), ha='center', va='center', color='black')

        plt.xlim(-0.5, policy[checkpoint_value].shape[1] - 0.5)
        plt.ylim(policy[checkpoint_value].shape[0] - 0.5, -0.5)
        plt.show()

    # Plot metrics over iterations
    efficiency = [1 / step if step > 0 else 0 for step in steps_to_goal]

    plt.figure(figsize=(12, 10))

    # Steps to Goal
    plt.subplot(4, 1, 1)
    plt.plot(range(1, len(steps_to_goal) + 1), steps_to_goal, marker='o')
    plt.title('Steps to Goal Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Steps to Goal')
    plt.grid()

    # Efficiency
    plt.subplot(4, 1, 2)
    plt.plot(range(1, len(efficiency) + 1), efficiency, marker='o', color='orange')
    plt.title('Efficiency Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Efficiency (1/Steps)')
    plt.grid()

    # Convergence Speed
    plt.subplot(4, 1, 3)
    plt.plot(range(1, len(convergence_speeds) + 1), convergence_speeds, marker='o', color='green')
    plt.title('Convergence Speed Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Max Value Change (Delta)')
    plt.grid()

    # Learning Stability
    plt.subplot(4, 1, 4)
    plt.plot(range(1, len(learning_stabilities) + 1), learning_stabilities, marker='o', color='purple')
    plt.title('Learning Stability Over Iterations')
    plt.xlabel('Iteration Count')
    plt.ylabel('Standard Deviation of Updates')
    plt.grid()

    plt.tight_layout()
    plt.show()

# Function to display optimal path from start to end position using optimal policy
def display_path(policy):
    current_pos = (1, 1)  
    passed_checkpoint = False
    path = [current_pos]

    
    while current_pos != END:
        action = policy[passed_checkpoint][current_pos]
        new_pos = get_new_state(current_pos, action)
        path.append(new_pos)

       
        if new_pos == CHECKPOINT and not passed_checkpoint:
            passed_checkpoint = True

        current_pos = new_pos
    
    # Plot optimal path on policy grid
    plt.matshow(policy[True])
    plt.title("Optimal Path")
    plt.plot([pos[1] for pos in path], [pos[0] for pos in path], color='red')
    plt.show()

# Main script to initialize maze, run policy iteration, and display results
if __name__ == "__main__":
    MAZE: ndarray = maze_generator()
    THRESHOLD: float = 0.5 
    GAMMA: float = 0.9
    CHECKPOINT: Tuple[int, int] = (5, 3)
    END: Tuple[int, int] = (8, 7)

    policy = policy_iteration()
    display_path(policy)
