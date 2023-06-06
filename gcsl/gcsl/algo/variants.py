from gcsl.algo import buffer, networks, buffer2
from gcsl.envs.env_utils import DiscretizedActionEnv
from gym.spaces import Box, Discrete
import numpy as np

"""
Main function defined up top. Helpers below.
"""

def get_params(env, env_params, idm_args=None, model_args=None):
    env = discretize_environment(env, env_params)
    policy = default_markov_policy(env, env_params, idm_args=idm_args, model_args=model_args)
    buffer_kwargs = dict(
        env=env,
        max_trajectory_length=get_horizon(env_params), 
        buffer_size=20000,
    )
    rep_buffer = buffer2.ReplayBuffer2 if idm_args.use_transformer else buffer.ReplayBuffer
    replay_buffer = rep_buffer(**buffer_kwargs)
    gcsl_kwargs = default_gcsl_params(env, env_params)
    gcsl_kwargs['validation_buffer'] = rep_buffer(**buffer_kwargs)
    return env, policy, replay_buffer, gcsl_kwargs

def get_horizon(env_params):
    return env_params.get('max_trajectory_length', 50)

def discretize_environment(env, env_params):
    if isinstance(env.action_space, Discrete):
        return env
    granularity = env_params.get('action_granularity', 3)
    env_discretized = DiscretizedActionEnv(env, granularity=granularity)
    return env_discretized

def default_markov_policy(env, env_params, idm_args=None, model_args=None):
    assert isinstance(env.action_space, Discrete)
    if idm_args.use_transformer:
        return idm_args.policy_class(env, model_args)
    elif env.action_space.n > 100: # Too large to maintain single action for each
        policy_class = networks.IndependentDiscretizedStochasticGoalPolicy
    else:
        policy_class = networks.DiscreteStochasticGoalPolicy
    return policy_class(
                env,
                state_embedding=None,
                goal_embedding=None,
                layers=[400, 300], #[400, 300], # TD3-size
                max_horizon=None, # Do not pass in horizon.
                # max_horizon=get_horizon(env_params), # Use this line if you want to include horizon into the policy
                freeze_embeddings=True,
                add_extra_conditioning=False,
            )

def default_gcsl_params(env, env_params):
    return dict(
        max_path_length=env_params.get('max_trajectory_length', 50),
        goal_threshold=env_params.get('goal_threshold', 0.05),
        explore_timesteps=10000,
        start_policy_timesteps=1000,
        eval_freq=env_params.get('eval_freq', 2000),
        eval_episodes=env_params.get('eval_episodes', 50),
        save_every_iteration=False,
        max_timesteps=env_params.get('max_timesteps', 1e6),
        expl_noise=0,
        batch_size=256,
        n_accumulations=1,
        policy_updates_per_step=1,
        train_policy_freq=None,
        lr=5e-4,
    )