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
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


grid_size = 10
env = gym.make('CartPole-v1')

agent = Agent(env)
agent.load_state_dict(torch.load('agent_cart.pt'))

state_predictor = nn.Sequential(
        nn.Linear(np.array(env.observation_space.shape).prod(), 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, np.array(env.observation_space.shape).prod()),
    )

# state_predictor.load_state_dict(torch.load('state_predictor_cart.pt'))

obs = env.reset()
env.render()

data_size = 250_000
# obs_data = np.zeros((data_size, 4), dtype=np.int32)
# next_obs_data = np.zeros((data_size, 4), dtype=np.int32)
obs_dtype = np.float32
obs_data = np.zeros((data_size, 4), dtype=obs_dtype)
next_obs_data = np.zeros((data_size, 4), dtype=obs_dtype)
current_added = 0

prev_state = np.asarray(obs, dtype=np.int32)
store_data = True
print(env.observation_space.shape, env.action_space.n)
ep_reward = 0
for i in range(data_size*2):
    action = agent.get_action_and_value(torch.tensor(obs, dtype=torch.float32))[0].item()

    predicted_state = state_predictor(torch.tensor(obs, dtype=torch.float32))
    obs, reward, done, info = env.step(action)
    ep_reward += reward
    next_state = np.asarray(obs, dtype=obs_dtype)
    # env.render()
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
        print(f'Episode reward: {ep_reward}')
        ep_reward = 0

# store data
if store_data:
    np.save('obs_data_cart.npy', obs_data)
    np.save('next_obs_data_cart.npy', next_obs_data)

