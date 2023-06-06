import gym
from gym import spaces
import numpy as np


class GridworldEnv(gym.Env):
    def __init__(self, grid_sz=3):
        self.height = grid_sz
        self.width = grid_sz
        self.action_space = spaces.Discrete(4)
        print("self.action_space: ", self.action_space)
        self.observation_space = spaces.Box(low=-grid_sz, high=grid_sz, shape=(2,), dtype=np.int64)

        print("self.observation_space: ", self.observation_space)
        
        self.moves = {
                0: [-1, 0],  # up
                1: [0, 1],   # right
                2: [1, 0],   # down
                3: [0, -1],  # left
            }

        # begin in start state
        self.reset()
        self.max_steps = 2 * (self.height + self.width)
        self.current_step = 0

    def step(self, action):
        """Simple deterministic dynamics. -1 reward for each step.
           1 reward for reaching goal state."""
        assert self.action_space.contains(action)
        self.S = (self.S[0] + self.moves[action][0], self.S[1] + self.moves[action][1])
        self.S = [np.clip(self.S[0], 0, self.height-1), np.clip(self.S[1], 0, self.width-1)]
        reward = 0
        done = False
        self.current_step += 1
        if self.S == self.G or self.current_step >= self.max_steps:
            reward = 1 if self.S == self.G else 0
            done = True
        return np.array([self.grid_pos_to_idx(self.S), self.grid_pos_to_idx(self.G)]), reward, done, {}

    def reset(self):
        gen_pos = lambda: [np.random.randint(0, self.height), np.random.randint(0, self.width)]
        self.S = gen_pos()
        self.G = gen_pos()
        while self.S == self.G:
            self.G = gen_pos()
        return np.array([self.grid_pos_to_idx(self.S), self.grid_pos_to_idx(self.G)])
    
    def grid_pos_to_idx(self, pos):
        return pos[0] * self.width + pos[1]
    
    def idx_to_grid_pos(self, idx):
        return (idx // self.width, idx % self.width)