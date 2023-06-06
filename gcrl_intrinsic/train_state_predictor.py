import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Load data
obs_data = np.load('obs_data_cart.npy')
next_obs_data = np.load('next_obs_data_cart.npy')

# Normalize data
# obs_data = obs_data / 10.0
# next_obs_data = next_obs_data

# print(obs_data.mean(axis=0, keepdims=True))
# print(obs_data.std(axis=0, keepdims=True))
# print(obs_data.max(axis=0, keepdims=True))
# obs_data = (obs_data - obs_data.max(axis=0, keepdims=True)) #/ (obs_data.std(axis=0, keepdims=True) + )
# next_obs_data = (next_obs_data - next_obs_data.max(axis=0, keepdims=True)) #/ next_obs_data.std(axis=0, keepdims=True)


# Hyperparameters
grid_size = 10
learning_rate = 0.001
batch_size = 32
epochs = 10
report_every = 100

# Model definition
# state_predictor = nn.Sequential(
#     nn.Linear(4, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, grid_size+grid_size),
# )
state_predictor = nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(state_predictor.parameters(), lr=learning_rate)

# Dataset and DataLoader
# dataset = torch.utils.data.TensorDataset(torch.tensor(obs_data, dtype=torch.float32), torch.tensor(next_obs_data[:,:2], dtype=torch.int64))
dataset = torch.utils.data.TensorDataset(torch.tensor(obs_data, dtype=torch.float32), torch.tensor(next_obs_data[:, [0,2]], dtype=torch.float32))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training
for epoch in range(epochs):
    for step, (obs_batch, next_obs_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # logits = state_predictor(obs_batch)
        # x_logits, y_logits = logits[:,:grid_size], logits[:,grid_size:]
        
        # x_labels, y_labels = next_obs_batch[:,0].long(), next_obs_batch[:,1].long()
        # loss_x = criterion(x_logits, x_labels)
        # loss_y = criterion(y_logits, y_labels)
        
        # loss = loss_x + loss_y
        pred_state = state_predictor(obs_batch)

        loss = F.mse_loss(pred_state, next_obs_batch)
        loss.backward()
        optimizer.step()
        
        if step % report_every == 0:
            with torch.no_grad():
                # pred_x, pred_y = x_logits.argmax(dim=1), y_logits.argmax(dim=1)
                # acc_x, acc_y = (pred_x == x_labels).float().mean(), (pred_y == y_labels).float().mean()
                # total_accuracy = ((pred_x == x_labels) & (pred_y == y_labels)).float().mean()
                
                # print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Accuracy X: {acc_x.item()}, Accuracy Y: {acc_y.item()}, Total Accuracy: {total_accuracy.item()}")
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
                # save state_predictor 
                # torch.save(state_predictor.state_dict(), 'state_predictor_simple.pt')
                torch.save(state_predictor.state_dict(), 'state_predictor_cart.pt')
