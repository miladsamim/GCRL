import torch 
import torch.nn as nn
from torch.distributions.categorical import Categorical


import gym
import custom_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box

class TupleToBoxWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=self.env.grid_size - 1, shape=(4,), dtype=np.int32)

    def observation(self, observation):
        return np.array(observation, dtype=np.int32)


class Agent(nn.Module):
    def __init__(self, grid_size, dim=64):
        super().__init__()
        self.state_embedding = nn.Embedding(grid_size**2, dim)


        self.grid_size = grid_size
        self.dim = dim
        self.state_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(grid_size*grid_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
        )
        self.state_encoder.load_state_dict(torch.load('state_encoder.pt'))
        self.critic = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, 4),
        )



    def get_action_and_value(self, x, action=None):
        # with torch.no_grad():
            # x = self.get_state_enc_tokens(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

grid_size = 10
env = gym.make('SimpleGridWorld-v0', grid_size=grid_size)

agent = Agent(grid_size)
agent.load_state_dict(torch.load('agent_simple.pt'))

state_predictor = nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, grid_size+grid_size),
    )

state_predictor.load_state_dict(torch.load('state_predictor_simple.pt'))

obs = env.reset()
env.render()

data_size = 15
obs_data = np.zeros((data_size, 4), dtype=np.int32)
next_obs_data = np.zeros((data_size, 4), dtype=np.int32)
current_added = 0

prev_state = np.asarray(obs, dtype=np.int32)
store_data = False

for i in range(data_size*2):
    action = agent.get_action_and_value(torch.tensor(obs, dtype=torch.float32))[0]
    predicted_state = state_predictor(torch.tensor(obs, dtype=torch.float32)/10)
    pred_x = predicted_state[:grid_size].argmax(dim=0)
    pred_y = predicted_state[grid_size:].argmax(dim=0)
    obs, reward, done, info = env.step(action)
    next_state = np.asarray(obs, dtype=np.int32)
    print('Prev state', prev_state[0], prev_state[1])
    print('Predicted state', pred_x, pred_y)
    env.render()
    if store_data: 
        if (i+1) % 1000 == 0:
            print(f'{i+1} steps')
        obs_data[current_added] = prev_state
        next_obs_data[current_added] = next_state
        current_added += 1
        if current_added+1 == data_size:
            break
    prev_state = next_state
    if done:
        obs = env.reset()

# store data
if store_data:
    np.save('obs_data.npy', obs_data)
    np.save('next_obs_data.npy', next_obs_data)

