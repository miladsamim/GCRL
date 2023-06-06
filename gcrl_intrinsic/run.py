import gymnasium as gym 

env = gym.make("FetchPush-v2", render_mode='human')

env.reset()
env.render()
for i in range(4000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

env.close()