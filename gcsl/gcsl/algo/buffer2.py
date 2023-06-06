import numpy as np
import einops

from gcsl.algo.masking import generate_random_goal_mask, generate_traj_mask
import rlutil.torch.pytorch_util as ptu

class ReplayBuffer2:
    """
    The base class for a replay buffer: stores gcsl.envs.GoalEnv states,
    and on sampling time, chooses out the observation, goals, etc using the 
    env.observation, etc class
    Modified Version of ReplayBuffer, made to use with transformer like training
    """

    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                ):
        """
        Args:
            env: A gcsl.envs.GoalEnv
            max_trajectory_length (int): The length of each trajectory (must be fixed)
            buffer_size (int): The maximum number of trajectories in the buffer
        """
        self.env = env
        # self._actions = np.zeros(
        #     (buffer_size, max_trajectory_length, 9), # For continous CLAW
        #     dtype=np.float32
        # )
        self._actions = np.zeros(
            (buffer_size, max_trajectory_length, *env.action_space.shape),
            dtype=env.action_space.dtype
        )
        self._states = np.zeros(
            (buffer_size, max_trajectory_length, *env.state_space.shape),
            dtype=env.state_space.dtype
        )
        self._desired_states = np.zeros(
            (buffer_size, *env.state_space.shape),
            dtype=env.state_space.dtype
        )
        
        internal_goal_shape = env._extract_sgoal(env.sample_goal()).shape
        self._internal_goals = np.zeros(
            (buffer_size, max_trajectory_length, *internal_goal_shape),
            dtype=env.observation_space.dtype,
        )
        
        self._length_of_traj = np.zeros(
            (buffer_size,),
            dtype=int
        )
        self.pointer = 0
        self.current_buffer_size = 0
        self.max_buffer_size = buffer_size
        self.max_trajectory_length = max_trajectory_length
        
    def add_trajectory(self, states, actions, desired_state, length_of_traj=None):
        """
        Adds a trajectory to the buffer

        Args:
            states (np.array): Environment states witnessed - Needs shape (self.max_path_length, *state_space.shape)
            actions (np.array): Actions taken - Needs shape (max_path_length, *action_space.shape)
            desired_state (np.array): The state attempting to be reached - Needs shape state_space.shape
        
        Returns:
            None
        """

        assert actions.shape == self._actions[0].shape
        assert states.shape == self._states[0].shape

        self._actions[self.pointer] = actions
        self._states[self.pointer] = states
        self._internal_goals[self.pointer] = self.env._extract_sgoal(states)
        self._desired_states[self.pointer] = desired_state
        if length_of_traj is None:
            length_of_traj = self.max_trajectory_length
        self._length_of_traj[self.pointer] = length_of_traj
 
        self.pointer += 1
        self.current_buffer_size = max(self.pointer, self.current_buffer_size)
        self.pointer %= self.max_buffer_size
    
    def _sample_indices(self, batch_size):
        traj_idxs = np.random.choice(self.current_buffer_size, batch_size)

        prop_idxs_1 = np.random.rand(batch_size)
        prop_idxs_2 = np.random.rand(batch_size)
        time_idxs_1 = np.floor(prop_idxs_1 * (self._length_of_traj[traj_idxs]-1)).astype(int)
        time_idxs_2 = np.floor(prop_idxs_2 * (self._length_of_traj[traj_idxs])).astype(int)
        time_idxs_2[time_idxs_1 == time_idxs_2] += 1

        time_state_idxs = np.minimum(time_idxs_1, time_idxs_2)
        time_goal_idxs = np.maximum(time_idxs_1, time_idxs_2)
        return traj_idxs, time_state_idxs, time_goal_idxs

    def sample_batch(self, batch_size, masking='random', mask_type=None):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals # not needed in transformer case
            lengths - Distance between observations and goals # not needed in transformer case
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...) # not needed in transformer case
            weights - Will be all ones (uniform) # not needed in transformer case but left as is
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices(batch_size)
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs, masking=masking, mask_type=None)

    def _get_batch(self, traj_idxs, time_state_idxs, time_goal_idxs, masking='random', mask_type=None):
        batch_size = len(traj_idxs)
        # observations = self.env.observation(self._states[traj_idxs])
        observations = self._states[traj_idxs]
        actions = self._actions[traj_idxs]
        # acts = []
        # for i in range(batch_size):
        #     acts.extend(actions[i][time_state_idxs[i]:time_goal_idxs[i]])
        # acts = np.array(acts)
        # assert len(acts) == sum(time_goal_idxs-time_state_idxs)
        if masking == 'random':
            goal_mask, goal_idxs = generate_random_goal_mask(1, self.max_trajectory_length, 
                                                             device='cuda' if ptu.USE_GPU else 'cpu',
                                                             restrictive=mask_type)
            goal_mask = einops.rearrange(goal_mask, '() t1 t2 -> t1 t2') 
        elif masking == 'traj':
            goal_mask = generate_traj_mask(batch_size, self.max_trajectory_length, 
                                           time_state_idxs, time_goal_idxs,
                                           device='cuda' if ptu.USE_GPU else 'cpu',
                                           mask_type=mask_type)
        else:
            goal_mask = None 
        
        return observations, actions, goal_mask, time_state_idxs, time_goal_idxs
    
    def save(self, file_name):
        np.savez(file_name,
            states=self._states[:self.current_buffer_size],
            actions=self._actions[:self.current_buffer_size],
            desired_states=self._desired_states[:self.current_buffer_size],
        )

    def load(self, file_name, replace=False):
        data = np.load(file_name)
        states, actions, desired_states = data['states'], data['actions'], data['desired_states']
        n_trajectories = len(states)
        for i in range(n_trajectories):
            self.add_trajectory(states[i], actions[i], desired_states[i])

    def state_dict(self):
        d = dict(internal_goals=self._internal_goals[:self.current_buffer_size])
        if self._states.shape[2] < 100: # Not images
            d.update(dict(
                states=self._states[:self.current_buffer_size],
                actions=self._actions[:self.current_buffer_size],
                desired_states=self._desired_states[:self.current_buffer_size],
            ))
        return d