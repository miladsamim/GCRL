import torch 
import einops
import numpy as np

def noise_to_targets(targets, noise_rate=0.5, l=0,h=4, device='cpu'):
    targets = targets.clone()
    r = torch.rand(*targets.shape, device=device)
    targets_noise = torch.randint(l,h,targets.shape, device=device)
    targets[r.gt(1-noise_rate)] = targets_noise[r.gt(1-noise_rate)]
    return targets

def generate_square_subsequent_mask(sz: int, device='cpu') -> torch.Tensor:
    return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

def generate_restrictive_square_subsequent_mask(sz: int, device='cpu') -> torch.Tensor:
    mask = torch.eye(sz, device=device)
    mask[torch.eye(sz) == 0] = float('-inf')
    mask[torch.eye(sz) == 1] = 0
    return mask

def random_goal_idxs(b_size, t, even_only=False):
    """Generates random goal idxs to use in masking s.t. that each goal state is sampled
       from the upper triangular matrix, so that the goal state is a future state for each
       timesteps t. 
       Except for last state as that would have no future state availble anyway -> so set to 0.
       even_only is for pact like interleaved setup to mark only even indicies that is states"""
    goal_idxs = []
    for i in range(t-1):
        goal_idxs.append(torch.randint(i+1,t, (b_size, 1, 1)))
        if even_only:
            g_idx = goal_idxs[-1]
            g_idx = g_idx if g_idx % 2 == 0 else g_idx-1
            goal_idxs[-1] = g_idx
    goal_idxs.append(torch.zeros((b_size,1,1), dtype=torch.int)) # last has no goal state anyway
    return einops.rearrange(goal_idxs, 't b () () -> b t ()')

def generate_random_goal_mask(b_size: int, sz: int, device='cpu', mask_type=False, even_only=False) -> torch.Tensor:
    if mask_type=='restrictive':
        mask = generate_restrictive_square_subsequent_mask(sz, device).repeat(b_size,1,1)
    else:
        mask = generate_square_subsequent_mask(sz, device=device).repeat(b_size,1,1)
    goal_idxs = random_goal_idxs(b_size, sz, even_only=even_only).to(device)
    goal_mask = mask.scatter(2, goal_idxs, torch.zeros_like(goal_idxs, dtype=torch.float32, device=device))
    return goal_mask, goal_idxs

def k_ahead_goal_idxs(t: int, k:int) -> torch.Tensor:
    """Generates idxs k_ahead for every timestep.
       if t_i+k > t, then we simple set those to last available timestep that is t-1"""
    k = np.clip(k,1,t-1) # k-ahead goal state
    ahead_goals = torch.arange(t)
    ahead_goals[:-k] = ahead_goals[:-k] + k 
    ahead_goals[-k:] = t-1
    return ahead_goals.unsqueeze(-1)

def generate_k_ahead_goal_mask(sz: int, k: int, device='cpu') -> torch.Tensor:
    mask = generate_square_subsequent_mask(sz, device=device)
    goal_idxs = k_ahead_goal_idxs(sz, k).to(device)
    goal_mask = mask.scatter(1, goal_idxs, torch.zeros_like(goal_idxs, dtype=torch.float32, device=device))
    return goal_mask, goal_idxs

def generate_traj_mask(b_size: int, sz: int, t1:np.array, t2:np.array,
                       device='cpu', mask_type='base') -> torch.Tensor:
    mask = torch.full((b_size,sz,sz),float('-inf'), device=device)
    for i in range(b_size):
        if mask_type == 'restrictive': # unmask s_k and g
            mask[i][:,t1[i]] = 0
            mask[i][:,t2[i]] = 0
        elif mask_type == 'pre': # unmask s_0,...,s_k,...,s_g
            mask[i][:,:t2[i]+1] = 0 # +1 to include s_g
        elif mask_type == 'all': # unmask everything
            mask = torch.zeros(b_size, sz, sz, device=device)
        elif mask_type == 'base': # unmask s_k,...,s_g
            mask[i][:,t1[i]:t2[i]+1] = 0 # +1 to include s_g
        else:
            raise Exception(f"mask_type = {mask_type} is an invalid mask type!")
    return mask

def zerofy(X, time_state_idxs, goal_state_idxs, device='cpu'):
    b_size = X.shape[0]
    Z = torch.zeros_like(X, device=device)
    arange = torch.arange(b_size, device=device)
    Z[[arange, time_state_idxs]] = X[[arange, time_state_idxs]]
    Z[[arange, goal_state_idxs]] = X[[arange, goal_state_idxs]]
    return Z