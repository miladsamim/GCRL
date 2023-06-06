import numpy as np
import gym
from gym import spaces

class SimpleGridWorld(gym.Env):
    def __init__(self, grid_size):
        super(SimpleGridWorld, self).__init__()

        self.grid_size = grid_size

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, down, left, right
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size),
            spaces.Discrete(self.grid_size),
            spaces.Discrete(self.grid_size),
            spaces.Discrete(self.grid_size)
        ))

        self.max_steps = 2 * self.grid_size  # A meaningful limit depending on the grid size

    def reset(self):
        self.current_step = 0

        # Random start and end positions
        self.start_pos = np.random.randint(0, self.grid_size, size=2)
        self.end_pos = np.random.randint(0, self.grid_size, size=2)

        # Ensure start and end positions are different
        while np.all(self.start_pos == self.end_pos):
            self.end_pos = np.random.randint(0, self.grid_size, size=2)

        self.agent_pos = np.copy(self.start_pos)

        return self._get_obs()

    def _get_obs(self):
        return (self.agent_pos[0], self.agent_pos[1], self.end_pos[0], self.end_pos[1])

    def step(self, action):
        self.current_step += 1

        # Update agent position based on action
        if action == 0:  # Up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # Right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)

        done = False
        reward = 0

        # Check if the agent has reached the end position
        if np.all(self.agent_pos == self.end_pos):
            reward = 1
            done = True
        elif self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '-'
        grid[tuple(self.start_pos)] = 'S'
        grid[tuple(self.end_pos)] = 'E'
        grid[tuple(self.agent_pos)] = 'A'
        print(grid)


    