import numpy as np
import einops
from rlutil.logging import logger
from gcsl.algo.masking import (generate_k_ahead_goal_mask, 
                               generate_random_goal_mask,
                               generate_traj_mask)
from gcsl.algo.dino_loss import DINOLoss
from gcsl.algo.vicreg_loss import variance_loss, covariance_loss

import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu
import torch.nn.functional as F

import time
import tqdm
import os.path as osp
import copy
import pickle
try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_enabled = True
except:
    print('Tensorboard not installed!')
    tensorboard_enabled = False

class GCSL:
    """Goal-conditioned Supervised Learning (GCSL).

    Parameters:
        env: A gcsl.envs.goal_env.GoalEnv
        policy: The policy to be trained (likely from gcsl.algo.networks)
        replay_buffer: The replay buffer where data will be stored
        validation_buffer: If provided, then 20% of sampled trajectories will
            be stored in this buffer, and used to compute a validation loss
        max_timesteps: int, The number of timesteps to run GCSL for.
        max_path_length: int, The length of each trajectory in timesteps

        # Exploration strategy
        
        explore_timesteps: int, The number of timesteps to explore randomly
        expl_noise: float, The noise to use for standard exploration (eps-greedy)

        # Evaluation / Logging Parameters

        goal_threshold: float, The distance at which a trajectory is considered
            a success. Only used for logging, and not the algorithm.
        eval_freq: int, The policy will be evaluated every k timesteps
        eval_episodes: int, The number of episodes to collect for evaluation.
        save_every_iteration: bool, If True, policy and buffer will be saved
            for every iteration. Use only if you have a lot of space.
        log_tensorboard: bool, If True, log Tensorboard results as well

        # Policy Optimization Parameters
        
        start_policy_timesteps: int, The number of timesteps after which
            GCSL will begin updating the policy
        batch_size: int, Batch size for GCSL updates
        n_accumulations: int, If desired batch size doesn't fit, use
            this many passes. Effective batch_size is n_acc * batch_size
        policy_updates_per_step: float, Perform this many gradient updates for
            every environment step. Can be fractional.
        train_policy_freq: int, How frequently to actually do the gradient updates.
            Number of gradient updates is dictated by `policy_updates_per_step`
            but when these updates are done is controlled by train_policy_freq
        lr: float, Learning rate for Adam.
        demonstration_kwargs: Arguments specifying pretraining with demos.
            See GCSL.pretrain_demos for exact details of parameters        
    """
    def __init__(self,
        env,
        policy,
        replay_buffer,
        validation_buffer=None,
        max_timesteps=1e6,
        max_path_length=50,
        # Exploration Strategy
        explore_timesteps=1e4,
        expl_noise=0.1,
        # Evaluation / Logging
        goal_threshold=0.05,
        eval_freq=5e3,
        eval_episodes=200,
        save_every_iteration=False,
        log_tensorboard=False,
        # Policy Optimization Parameters
        start_policy_timesteps=0,
        batch_size=100,
        n_accumulations=1,
        policy_updates_per_step=1,
        train_policy_freq=None,
        demonstrations_kwargs=dict(),
        lr=5e-4,
        idm_args=None,
    ):
        self.env = env
        self.device = 'cuda' if ptu.USE_GPU else 'cpu'
        self.policy = policy.to(self.device)
        self.replay_buffer = replay_buffer
        self.validation_buffer = validation_buffer

        self.is_discrete_action = hasattr(self.env.action_space, 'n')

        self.max_timesteps = max_timesteps
        self.max_path_length = max_path_length

        self.explore_timesteps = explore_timesteps
        self.expl_noise = expl_noise

        self.goal_threshold = goal_threshold
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_every_iteration = save_every_iteration

        self.idm_args = idm_args
        self.ssl = idm_args.ssl
        if self.ssl:
            args = idm_args.model_args
            out_dim = env.action_space.n
            self.teacher = self.policy.__class__(env, args).to(self.device)
            self.dino_loss = DINOLoss(args.out_dim, args.teacher_sharpening, args.student_sharpening, args.centering_momentum).to(self.device)
            # there is no backpropagation through the teacher, so no need for gradients
            # set teacher weights to student 
            self.teacher.load_state_dict(self.policy.state_dict())
            for p in self.teacher.parameters():
                p.requires_grad = False
            
        self.start_policy_timesteps = start_policy_timesteps

        if train_policy_freq is None:
            train_policy_freq = self.max_path_length

        self.train_policy_freq = train_policy_freq
        self.batch_size = batch_size
        self.n_accumulations = n_accumulations
        self.policy_updates_per_step = policy_updates_per_step
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.log_tensorboard = log_tensorboard and tensorboard_enabled
        self.summary_writer = None


    def loss_fn(self, observations, goals, actions, horizons=None, weights=None, **kwargs):
        obs_dtype = torch.float32
        action_dtype = torch.int64 if self.is_discrete_action else torch.float32

        observations_torch = torch.tensor(observations, dtype=obs_dtype, device=self.device)
        actions_torch = torch.tensor(actions, dtype=action_dtype, device=self.device)
        
        if self.idm_args.use_transformer:
            conditional_nll = self.policy.nll(observations_torch, actions_torch, **kwargs)    
            return torch.mean(conditional_nll)
        else:
            goals_torch = torch.tensor(goals, dtype=obs_dtype, device=self.device)
            horizons_torch = torch.tensor(horizons, dtype=obs_dtype, device=self.device)
            weights_torch = torch.tensor(weights, dtype=torch.float32, device=self.device)
            if self.idm_args.vicreg:
                s = self.policy(observations_torch, goals_torch)
                s = einops.rearrange(s, 'b h -> b h')
                targets = einops.rearrange(actions_torch, 'b -> b')
                loss = F.cross_entropy(s, targets, reduction='mean')
                std_loss = variance_loss(s)
                cov_loss = covariance_loss(s)
                lambda_ = self.idm_args.lambda_
                mu = self.idm_args.mu
                nu = self.idm_args.nu
                loss = lambda_ * loss + mu * std_loss + nu * cov_loss
                return loss 
            else:   
                conditional_nll = self.policy.nll(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)
                nll = conditional_nll
                return torch.mean(nll * weights_torch)

    def ssl_loss(self, observations, goals, actions, mask=None, update_center=True, 
                 dino_loss=True, t1=None, t2=None):
        obs_dtype = torch.float32
        action_dtype = torch.int64 if self.is_discrete_action else torch.float32
        observations_torch = torch.tensor(observations, dtype=obs_dtype, device=self.device)
        actions_torch = torch.tensor(actions, dtype=action_dtype, device=self.device)
        
        if self.idm_args.use_transformer:
            b_size, t, h = observations_torch.size()
            # mask = generate_traj_mask(b_size,t+1,t1+1,t2+1, device=self.device, mask_type=self.idm_args.model_args.mask_type)
            # mask = einops.repeat(mask, 'b t1 t2 -> (b repeat) t1 t2', repeat=self.idm_args.model_args.nhead)
            out, expansion = self.policy(observations_torch, mask=mask, t1=t1, t2=t2, expand=True)
            logits, targets = self.policy.extract_trajectory(out, actions_torch, t1, t2)
            s = einops.rearrange(expansion, 'b () h -> b h')
            if self.idm_args.use_claw:
                targets = self.policy.unflattened(targets)
                targets = einops.rearrange(targets, 'b n_dim -> (b n_dim)')    
                logits = einops.rearrange(logits, 'b (n_dim granularity) -> (b n_dim) granularity', granularity=self.policy.granularity)
            loss = F.cross_entropy(logits, targets, reduction='mean')
        else:
            goals_torch = torch.tensor(goals, dtype=obs_dtype, device=self.device)
            s = self.policy(observations_torch, goals_torch, mask=mask)
            if dino_loss:
                t = self.teacher(observations_torch, goals_torch, mask=mask)
                loss = self.dino_loss(s, t.detach(), update_center=update_center) # centering updated here
            else:
                targets = einops.rearrange(actions_torch, 'b -> b')
                loss = F.cross_entropy(s, targets, reduction='mean')
        if self.idm_args.vicreg:
            s = einops.rearrange(s, 'b h -> b h')
            if self.idm_args.expander:
                s = self.policy.expander(s)
            std_loss = variance_loss(s)
            cov_loss = covariance_loss(s)
            lambda_ = self.idm_args.lambda_
            mu = self.idm_args.mu 
            nu = self.idm_args.nu 
            loss = lambda_ * loss + mu * std_loss + nu * cov_loss

        return loss 

    def ssl_update(self, loss):
        # centering already updated when computing loss
        # student update
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        # teacher polyak
        self.dino_loss.polyak(self.teacher, self.policy, tau=self.idm_args.model_args.polyak_momentum, device=self.device)

    def sample_online_trajectory(self, greedy=False, noise=0, render=False, validation=False):
        goal_state = self.env.sample_goal()
        goal = self.env.extract_goal(goal_state)

        states = []
        actions = []

        state = self.env.reset()

        if self.idm_args.use_transformer:
            self.policy.reset_state_hist()
            goal = goal_state
        observation = self.env.observation(state)
        acts, predicted_reached_state = self.policy.act_vectorized(observation[None], goal[None], 
                                            greedy=greedy, noise=noise,
                                            restrictive=self.idm_args.restrictive_masking)
        actions.extend(acts)

        for t in range(self.max_path_length):
            if render:
                self.env.render()
            states.append(state)            
            state, _, _, _ = self.env.step(actions[t])
    
        if not validation:
            loss = self.policy.online_loss(predicted_reached_state, goal, states[-1])
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
        else:
            loss = 0

        return np.stack(states), np.array(actions), goal_state, loss
    
    def sample_trajectory(self, greedy=False, noise=0, render=False):

        goal_state = self.env.sample_goal()
        goal = self.env.extract_goal(goal_state)

        states = []
        actions = []

        state = self.env.reset()
        action = None # signal empty action at very first step

        if self.idm_args.use_transformer:
            self.policy.reset_state_hist()
            # goal = goal_state

        for t in range(self.max_path_length):
            if render:
                self.env.render()

            states.append(state)

            observation = self.env.observation(state)
            horizon = np.arange(self.max_path_length) >= (self.max_path_length - 1 - t) # Temperature encoding of horizon
            if self.idm_args.use_transformer:
                if self.idm_args.pact:
                    action = self.policy.act_vectorized(observation[None], action, goal[None],  # feed prev action too
                                                        greedy=greedy, noise=noise)[0]
                else:    
                    if self.ssl:
                        action = self.teacher.act_vectorized(observation[None], goal[None], 
                                                            greedy=greedy, noise=noise,
                                                            restrictive=self.idm_args.restrictive_masking)[0]
                    else:
                        action = self.policy.act_vectorized(observation[None], goal[None], 
                                                            greedy=greedy, noise=noise,
                                                            restrictive=self.idm_args.restrictive_masking)[0]
            else:
                if self.ssl:
                    action = self.teacher.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=noise)[0]
                else:
                    action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=noise)[0]
            
            if not self.is_discrete_action:
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            actions.append(action)
            state, _, _, _ = self.env.step(action)
        
        return np.stack(states), np.array(actions), goal_state

    def take_policy_step(self, buffer=None):
        if buffer is None:
            buffer = self.replay_buffer

        avg_loss = 0
        self.policy_optimizer.zero_grad()
        
        for _ in range(self.n_accumulations):
            if self.idm_args.use_transformer:
                observations, acts, goal_mask, time_state_idxs, time_goal_idxs = buffer.sample_batch(self.batch_size, 
                                                                                                     masking=self.idm_args.masking,
                                                                                                     mask_type=self.idm_args.model_args.mask_type)
                if self.ssl:
                    loss = self.ssl_loss(observations, None, acts, mask=None, update_center=True, dino_loss=self.idm_args.dino_loss,
                                         t1=time_state_idxs, t2=time_goal_idxs)
                    self.ssl_update(loss)
                else:
                    loss = self.loss_fn(observations, None, acts, mask=goal_mask,
                                        time_state_idxs=time_state_idxs, time_goal_idxs=time_goal_idxs)
            else:
                observations, actions, goals, _, horizons, weights = buffer.sample_batch(self.batch_size)
                if self.ssl:
                    loss = self.ssl_loss(observations, goals, actions, mask=None, update_center=True, dino_loss=self.idm_args.dino_loss)
                    self.ssl_update(loss)
                else:
                    loss = self.loss_fn(observations, goals, actions, horizons, weights)
            
            if not self.ssl:
                loss.backward()
                self.policy_optimizer.step()

            avg_loss += ptu.to_numpy(loss)
        
        return avg_loss / self.n_accumulations

    def validation_loss(self, buffer=None):
        if buffer is None:
            buffer = self.validation_buffer

        if buffer is None or buffer.current_buffer_size == 0:
            return 0

        avg_loss = 0
        for _ in range(self.n_accumulations):
            if self.idm_args.use_transformer:
                observations, acts, goal_mask, time_state_idxs, time_goal_idxs = buffer.sample_batch(self.batch_size, 
                                                                                                     masking=self.idm_args.masking,
                                                                                                     mask_type=self.idm_args.model_args.mask_type)
                if self.ssl:
                    loss = self.ssl_loss(observations, None, acts, mask=None, update_center=True, dino_loss=self.idm_args.dino_loss,
                                         t1=time_state_idxs, t2=time_goal_idxs)
                else:
                    loss = self.loss_fn(observations, None, acts, mask=goal_mask,
                                        time_state_idxs=time_state_idxs, time_goal_idxs=time_goal_idxs)
            else:
                observations, actions, goals, lengths, horizons, weights = buffer.sample_batch(self.batch_size)
                if self.ssl:
                    loss = self.ssl_loss(observations, goals, actions, mask=None, update_center=False, dino_loss=self.idm_args.dino_loss)
                else:
                    loss = self.loss_fn(observations, goals, actions, horizons, weights)
            avg_loss += ptu.to_numpy(loss)

        return avg_loss / self.n_accumulations

    def pretrain_demos(self, demo_replay_buffer=None, demo_validation_replay_buffer=None, demo_train_steps=0):
        if demo_replay_buffer is None:
            return

        self.policy.train()
        with tqdm.trange(demo_train_steps) as looper:
            for _ in looper:
                loss = self.take_policy_step(buffer=demo_replay_buffer)
                with torch.no_grad():
                    validation_loss = self.validation_loss(buffer=demo_validation_replay_buffer)

                if running_loss is None:
                    running_loss = loss
                else:
                    running_loss = 0.99 * running_loss + 0.01 * loss
                if running_validation_loss is None:
                    running_validation_loss = validation_loss
                else:
                    running_validation_loss = 0.99 * running_validation_loss + 0.01 * validation_loss

                looper.set_description('Loss: %.03f Validation Loss: %.03f'%(running_loss, running_validation_loss))
        
    def train(self):
        start_time = time.time()
        last_time = start_time

        # Evaluate untrained policy
        total_timesteps = 0
        timesteps_since_train = 0
        timesteps_since_eval = 0
        timesteps_since_reset = 0

        iteration = 0
        running_loss = None
        running_validation_loss = None
        if logger.get_snapshot_dir() and self.log_tensorboard:
            self.summary_writer = SummaryWriter(osp.join(logger.get_snapshot_dir(),'tensorboard'))

        # Evaluation Code
        self.policy.eval()
        self.evaluate_policy(self.eval_episodes, total_timesteps=0, greedy=True, prefix='Eval')
        logger.record_tabular('policy loss', 0)
        logger.record_tabular('timesteps', total_timesteps)
        logger.record_tabular('epoch time (s)', time.time() - last_time)
        logger.record_tabular('total time (s)', time.time() - start_time)
        last_time = time.time()
        logger.dump_tabular()
        # End Evaluation Code
        
        with tqdm.tqdm(total=self.eval_freq, smoothing=0) as ranger:
            while total_timesteps < self.max_timesteps:

                # Interact in environmenta according to exploration strategy.
                if self.idm_args.online_loss:
                    states, actions, goal_state, loss = self.sample_online_trajectory()
                elif total_timesteps < self.explore_timesteps:
                    states, actions, goal_state = self.sample_trajectory(noise=1)
                else:
                    # states, actions, goal_state = self.sample_trajectory(greedy=True, noise=self.expl_noise)
                    states, actions, goal_state = self.sample_trajectory(greedy=False, noise=self.expl_noise)

                # With some probability, put this new trajectory into the validation buffer
                if self.validation_buffer is not None and np.random.rand() < 0.2:
                    self.validation_buffer.add_trajectory(states, actions, goal_state)
                else:
                    self.replay_buffer.add_trajectory(states, actions, goal_state)

                total_timesteps += self.max_path_length
                timesteps_since_train += self.max_path_length
                timesteps_since_eval += self.max_path_length
                
                ranger.update(self.max_path_length)
                
                # Take training steps
                if timesteps_since_train >= self.train_policy_freq and total_timesteps > self.start_policy_timesteps:
                    timesteps_since_train %= self.train_policy_freq
                    self.policy.train()
                    for _ in range(int(self.policy_updates_per_step * self.train_policy_freq)):
                        if self.idm_args.online_loss:
                            loss = loss # done in trajectory sampling above
                            validation_loss = 0 # skip
                        else:
                            loss = self.take_policy_step()
                            validation_loss = self.validation_loss()
                        if running_loss is None:
                            running_loss = loss
                        else:
                            running_loss = 0.9 * running_loss + 0.1 * loss

                        if running_validation_loss is None:
                            running_validation_loss = validation_loss
                        else:
                            running_validation_loss = 0.9 * running_validation_loss + 0.1 * validation_loss

                    self.policy.eval()
                    ranger.set_description('Loss: %s Validation Loss: %s'%(running_loss, running_validation_loss))

                    if self.summary_writer:
                        self.summary_writer.add_scalar('Losses/Train', running_loss, total_timesteps)
                        self.summary_writer.add_scalar('Losses/Validation', running_validation_loss, total_timesteps)
                
                # Evaluate, log, and save to disk
                if timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    iteration += 1
                    # Evaluation Code
                    self.policy.eval()
                    self.evaluate_policy(self.eval_episodes, total_timesteps=total_timesteps, greedy=True, prefix='Eval')
                    logger.record_tabular('policy loss', running_loss or 0) # Handling None case
                    logger.record_tabular('timesteps', total_timesteps)
                    logger.record_tabular('epoch time (s)', time.time() - last_time)
                    logger.record_tabular('total time (s)', time.time() - start_time)
                    last_time = time.time()
                    logger.dump_tabular()
                    
                    # Logging Code
                    if logger.get_snapshot_dir():
                        modifier = str(iteration) if self.save_every_iteration else ''
                        torch.save(
                            self.policy.state_dict(),
                            osp.join(logger.get_snapshot_dir(), 'policy%s.pkl'%modifier)
                        )
                        if hasattr(self.replay_buffer, 'state_dict'):
                            d =self.replay_buffer.state_dict()
                            with open(osp.join(logger.get_snapshot_dir(), 'buffer%s.pkl'%modifier), 'wb') as f:
                                pickle.dump(d, f)
                            file_name = osp.join(logger.get_snapshot_dir(), 'stored_actions%s.npz'%modifier)
                            # np.savez(file_name, acts=d['store_actions'])

                        # if not self.idm_args.use_transformer:
                        #     full_dict = dict(env=self.env, policy=self.policy)
                        #     with open(osp.join(logger.get_snapshot_dir(), 'params%s.pkl'%modifier), 'wb') as f:
                        #         pickle.dump(full_dict, f)
                    
                    ranger.reset()
                    
    def evaluate_policy(self, eval_episodes=200, greedy=True, prefix='Eval', total_timesteps=0):
        env = self.env
        
        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        for index in tqdm.trange(eval_episodes, leave=True):
            if self.idm_args.online_loss:
                states, actions, goal_state, _ = self.sample_online_trajectory(validation=True)
            else:
                states, actions, goal_state = self.sample_trajectory(noise=0, greedy=greedy)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(states[-1], goal_state)
            
            final_dist_vec[index] = final_dist
            success_vec[index] = (final_dist < self.goal_threshold)

        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)

        logger.record_tabular('%s num episodes'%prefix, eval_episodes)
        logger.record_tabular('%s avg final dist'%prefix,  np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio'%prefix, np.mean(success_vec))
        if self.summary_writer:
            self.summary_writer.add_scalar('%s/avg final dist'%prefix, np.mean(final_dist_vec), total_timesteps)
            self.summary_writer.add_scalar('%s/success ratio'%prefix,  np.mean(success_vec), total_timesteps)
        diagnostics = env.get_diagnostics(all_states, all_goal_states)
        for key, value in diagnostics.items():
            logger.record_tabular('%s %s'%(prefix, key), value)
        
        return all_states, all_goal_states
