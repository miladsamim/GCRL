import os
import sys
dir_paths = ['dependencies']
sys.path += [os.path.join(os.getcwd(), dir_path) for dir_path in dir_paths]
from gcsl import envs

env = envs.create_env('lunar')
env.reset()
for i in range(100):
    env.step(env.action_space.sample())
    env.render(mode='human')