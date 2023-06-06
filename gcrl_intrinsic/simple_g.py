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


# Make sure you've registered the environment before running this code
env = gym.make("SimpleGridWorld-v0", grid_size=5)
env = TupleToBoxWrapper(env)  # Apply the wrapper
vec_env = DummyVecEnv([lambda: env])


model = PPO("MlpPolicy", vec_env, verbose=2)
model.learn(total_timesteps=10000)

obs = env.reset()
env.render()
for _ in range(20):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()



# import gym

# env = gym.make('SimpleGridWorld-v0', grid_size=5)

# obs = env.reset()
# env.render()

# for _ in range(10):
#     action = env.action_space.sample()
#     obs, reward, done, _ = env.step(action)
#     env.render()
#     if done:
#         break
