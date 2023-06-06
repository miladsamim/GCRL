from gym.envs.registration import register

register(
        id='Gridworld-v0',
        entry_point='custom_gym.envs:GridworldEnv',
)

register(
    id='SimpleGridWorld-v0',
    entry_point='custom_gym.envs:SimpleGridWorld',
)
