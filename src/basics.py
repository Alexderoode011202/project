import numpy as np
from numpy import ndarray
from typing import Tuple, Callable

def maze_generator() -> ndarray:
    matrix: ndarray = np.array([
        [1]*10,
        [1,"S",1,0,0,0,1,0,0,1],
        [1,0,1,0,1,0,1,0,1,1],
        [1,0,0,0,1,0,0,0,0,1],
        [1,1,1,0,1,1,1,1,0,1],
        [1,0,1,"G",0,0,0,1,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,1,1,0,1,0,1,1,0,1],
        [1,0,0,0,0,0,1,"E",0,1],
        [1]*10
    ])
    
    return matrix

def reward_function(agent_position: Tuple[int, int],
                    passed_checkpoint: bool,
                    checkpoint_position: Tuple[int, int],
                    end_position: Tuple[int, int]
                    ) -> int:

    if agent_position == end_position and passed_checkpoint:
        return 100
    elif agent_position == checkpoint_position and not passed_checkpoint:
        return 50
    elif agent_position == end_position and not passed_checkpoint:
        return -50
    return -20


    
"""
class State:
    def __init__(self, pos_x: int = 1, pos_y: int = 1, passed_sub_goal: bool = False):
        self.pos_x: int = pos_x
        self.pos_y: int = pos_y
        self.passed_sub_goal: bool = passed_sub_goal
        self.done: bool = False
        
    def update_pos(self, position: Tuple[int, int]):
        self.pos_x, self.pos_y = position
"""    
        

    

"""
class Environment:
    def __init__(self, maze: ndarray = maze_generator(), reward_function = reward_function()):
        self.maze               : ndarray                   = maze
        self.reward_function    : Callable                  =  reward_function
        self.start_position     : Tuple[int, int]           = (1,1)
        self.checkpoint_position: Tuple[int, int]           = (3,5)
        self.end_position       : Tuple[int,int]            = (7,8)
        
        
class Agent:
    def __init__(self):
        self.state      : State             = State()
        self.position   : Tuple[int, int]   = (self.state.pos_x, self.state.pos_y)
        self.points     : int               = 0
"""
