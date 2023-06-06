import sys
import os 
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(DIR_PATH)

import pickle as pkl

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, use_tanh=False):
        super(Predictor, self).__init__()
        self.state_embed = nn.Linear(input_dim, 128)
        self.reward_embed = nn.Linear(1, 128)
        self.use_tanh = use_tanh

        self.predict_state = nn.Sequential(nn.Linear(256, 400),
                                            nn.ReLU(),
                                            nn.Linear(400, 300),
                                            nn.ReLU(),
                                            nn.Linear(300, output_dim))

        
    def forward(self, state, reward):
        state = self.state_embed(state)
        reward = self.reward_embed(reward.unsqueeze(1))
        state_reward = torch.cat((state, reward), dim=1)
        if self.use_tanh:
            return torch.tanh(self.predict_state(state_reward))
        return self.predict_state(state_reward)

class TransitionDataset(Dataset):
    def __init__(self, data):
        self.data = []
        for traj in data.values():
            obs = torch.from_numpy(traj['observations']).float()
            rewards = torch.from_numpy(traj['rewards']).float()
            actions = torch.from_numpy(traj['actions']).float()
            
            rewards_sum = torch.sum(rewards) / 1000

            for i in range(len(obs) - 1):
                self.data.append((obs[i], rewards_sum, obs[i+1], actions[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    
# Hyperparameters
state_dim = 64
output_dim = 64
num_epochs = 2
batch_size = 32
action_dim = 2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assume 'state_dim' 
state_predictor = Predictor(state_dim, state_dim).to(device)
optimizer = torch.optim.Adam(state_predictor.parameters())
# Action predictor 
action_predictor = Predictor(state_dim, action_dim, use_tanh=True).to(device)
action_optimizer = torch.optim.Adam(action_predictor.parameters())

loss_fn = nn.MSELoss()

# load data/carracing-medium-v2.pkl
with open('data/carracing-medium-v2.pkl', 'rb') as f:
    data = pkl.load(f)
    print(data[0].keys())

    # data = {i: data[i] for i in range(5)}

dataset = TransitionDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

state_losses = []
action_losses = []
epoch_steps = len(dataloader) 

# Assume 'num_epochs' is given
for epoch in range(num_epochs):
    step = 1
    for state, reward, next_state, action in dataloader:
        # Move data to GPU if available
        state = state.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        action = action.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        action_optimizer.zero_grad()

        # Forward pass
        state_pred = state_predictor(state, reward)
        action_pred = action_predictor(state, reward)

        # Compute loss
        loss = loss_fn(state_pred, next_state)
        action_loss = loss_fn(action_pred, action)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        action_loss.backward()
        action_optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step}/{epoch_steps}], State Loss: {loss.item():.4f}, Action Loss: {action_loss.item():.4f}', end='\r')
        step += 1

    # Print epoch loss    
    print(f'\nEpoch [{epoch+1}/{num_epochs}], State Loss: {loss.item():.4f}, Action Loss: {action_loss.item():.4f}')
    state_losses.append(loss.item())
    action_losses.append(action_loss.item())

# Store losses as csv 
import pandas as pd
losses = pd.DataFrame({'state_loss': state_losses, 'action_loss': action_losses})
losses.to_csv('losses.csv', index=False)

# Save models
torch.save(state_predictor.state_dict(), 'state_predictor_carracer.pt')
torch.save(action_predictor.state_dict(), 'action_predictor_carracer.pt')
