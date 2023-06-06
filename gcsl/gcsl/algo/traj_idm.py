import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F

import math 
import einops
import rlutil.torch.pytorch_util as ptu

from collections import deque
from gcsl.algo.masking import (generate_square_subsequent_mask,
                               noise_to_targets,
                               zerofy,
                               generate_traj_mask)
from gcsl.algo.positional_encoding import PositionalEncoding

class GPT_IDM(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(args.d_model, args.nhead, dropout=args.dropout, dim_feedforward=args.dim_f)
        self.gpt = nn.TransformerEncoder(self.encoder, args.layers)

    def forward(self, X, mask=None):
        X = self.gpt(X, mask=mask)
        return X

class DiscreteTrajIDMPolicy(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.args = args
        self.device = args.device
        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.net = GPT_IDM(env, args)
        self.state_hist = deque(maxlen=args.max_len)
        self.act_token = nn.Embedding(1, args.d_model)
        self.pos_embd = nn.Embedding(args.max_len, args.d_model)
        self.goal_emb = nn.Sequential(
                            nn.Linear(8+5, 400),
                            nn.ReLU(),
                            nn.Linear(400, 300),
                            nn.ReLU(),
                            nn.Linear(300, args.d_model))
        self.out = nn.Linear(args.d_model, self.dim_out)

    def forward(self, obs, mask=None, t1=None, t2=None):
        X = self.add_act_tokens(obs, t1, t2, device=self.device)
        b_size = X.shape[0]
        idxs = torch.arange(b_size, device=self.device)
        X_g = self.env.extract_goal(X[idxs, t2+1])
        X_g, ps = einops.pack((X[idxs, t1], X_g), 'b *')
        X_g = self.goal_emb(X_g)
        X[idxs, t2+1] = X_g
        X = einops.rearrange(X, 'b t h -> t b h')
        X = self.net(X, mask=mask)
        X = self.out(X)
        return X 

    def reset_state_hist(self):
        self.state_hist = deque(maxlen=self.args.max_len)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0,
            marginal_policy=None, restrictive=False):
        return self.act(obs[0], goal[0], greedy, noise, restrictive)
    
    def act(self, obs, goal, greedy=False, noise=0, restrictive=False):
        self.state_hist.append(obs)
        X = einops.rearrange(list(self.state_hist), 't h -> t h')
        X, ps = einops.pack((X, goal), '* h')
        X = einops.rearrange(X, 't h -> () t h')
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = len(self.state_hist)
        t1 = np.array([n-1])
        t2 = np.array([n]) 
        mask = generate_traj_mask(1,n+2,t1+1,t2+1, device=self.device, mask_type=self.args.mask_type)
        mask = einops.repeat(mask, 'b t1 t2 -> (b repeat) t1 t2', repeat=self.args.nhead)
        X = self.forward(X, mask=mask, t1=t1, t2=t2)
        logits = X[n]
        noisy_logits = logits * (1 - noise)
        probs = torch.softmax(noisy_logits, dim=1)
        if greedy:
            samples = torch.argmax(probs, dim=1)
        else: 
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        return ptu.to_numpy(samples)

    def nll(self, obs, actions, mask=None, time_state_idxs=None, time_goal_idxs=None): 
        b_size, t, h = obs.size()
        mask = generate_traj_mask(b_size,t+1,time_state_idxs+1,time_goal_idxs+1, device=self.device, mask_type=self.args.mask_type)
        mask = einops.repeat(mask, 'b t1 t2 -> (b repeat) t1 t2', repeat=self.args.nhead)
        out = self.forward(obs, mask=mask, t1=time_state_idxs, t2=time_goal_idxs)
        logits, targets = self.extract_trajectory(out, actions, time_state_idxs, time_goal_idxs)
        return F.cross_entropy(logits, targets, reduction='mean')

    def add_act_tokens(self, X:torch.tensor, t1:np.array, t2:np.array, device:str='cpu') -> torch.tensor:
        """Use original sized state and goal indicies, will expand trajectories with +1
           s0,s1,s2,...,s_T -> s_0,s_1,...,s_k,a_k,a_{k+1},...,s_g,s_{g+1},...,s_T"""
        b_size, t, h = X.size()
        new_X = torch.zeros(b_size, t+1, h, dtype=X.dtype, device=device)
        for i in range(b_size):
            new_X[i,:t1[i]+1] = X[i,:t1[i]+1]
            new_X[i, t1[i]+1:t2[i]+1] = self.pos_embd.weight[t1[i]:t2[i]] + self.act_token.weight[0]
            new_X[i, t2[i]+1:] = X[i,t2[i]:]
        return new_X

    def extract_trajectory(self, out, actions, time_state_idxs, time_goal_idxs):
        logits = []
        targets = []
        out = einops.rearrange(out, 't b c_out -> b t c_out')
        b_size = out.shape[0]
        for i in range(b_size):
            logits.append(out[i][time_state_idxs[i]+1:time_goal_idxs[i]+1+1])
            targets.append(actions[i][time_state_idxs[i]:time_goal_idxs[i]+1])
        logits, _ = einops.pack(logits, '* c_out')
        targets, _ = einops.pack(targets, '*')
        return logits, targets 
