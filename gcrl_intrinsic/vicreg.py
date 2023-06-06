import torch
import torch.nn.functional as F

# VICREG
# Variance
def variance_loss(x, gamma=1, eps=0.0001):
    std_x = torch.sqrt(x.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(gamma - std_x)) 
    return std_loss
# Covariance 
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
def covariance_loss(x):
    b_size, num_features = x.shape[0], x.shape[1]
    x = x - x.mean(dim=0)
    cov_x = (x.T @ x) / (b_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) 
    return cov_loss