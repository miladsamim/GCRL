import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, max_buffer_size, K):
        self.buffer = deque(maxlen=max_buffer_size)
        self.max_buffer_size = max_buffer_size
        self.K = K

    def insert(self, data):
        for rollout in data:
            if rollout["actions"].size(0) > self.K:
                self.buffer.append(rollout)

    def sample(self, batch_size, device):
        sampled_indices = random.sample(range(len(self.buffer)), batch_size)
        sampled_data = [self.buffer[idx] for idx in sampled_indices]

        states_batch = []
        actions_batch = []
        returns_to_go_batch = []
        timesteps_batch = []

        for rollout in sampled_data:
            start_idx = random.randint(0, rollout["actions"].size(0) - self.K)
            end_idx = start_idx + self.K
            states_batch.append(rollout["states"][start_idx:end_idx].type(torch.float32))
            actions_batch.append(rollout["actions"][start_idx:end_idx].type(torch.float32))
            returns_to_go_batch.append(rollout["target_return"][:,start_idx:end_idx].type(torch.float32).transpose(0,1))
            timesteps_batch.append(rollout["timesteps"][0,start_idx:end_idx])

        # stack and float32
        states_batch = torch.stack(states_batch).to(device)
        actions_batch = torch.stack(actions_batch).to(device)
        returns_to_go_batch = torch.stack(returns_to_go_batch).to(device)
        timesteps_batch = torch.stack(timesteps_batch).to(device)

        return states_batch, actions_batch, returns_to_go_batch, timesteps_batch