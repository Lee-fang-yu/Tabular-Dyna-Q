"""
Description:
    The environment we'll use in this assignment. Please DO NOT modeify this file.
    Code adapted from ShangtongZhang!
Author:
    Chan-Wei Hu

"""
import numpy as np
from copy import deepcopy

class Maze(object):
    def __init__(self):
        self.WORLD_WIDTH = 9
        self.WORLD_HEIGHT = 6
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        self.START_STATE = [2, 0]
        self.GOAL_STATES = [[0, 8]]
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))
        self.max_steps = float('inf')

    def step(self, state, action):
        x, y = state
        
        # take action 
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        
        if [x, y] in self.obstacles:
            x, y = state
        
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
        else:
            reward = 0.0
        
        return [x, y], reward

