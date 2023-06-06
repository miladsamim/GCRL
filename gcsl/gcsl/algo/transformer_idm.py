import torch 
import torch.nn as nn
import torch.nn.functional as F

import math 
import einops
import rlutil.torch.pytorch_util as ptu

from collections import deque
from gcsl.algo.masking import (generate_square_subsequent_mask,
                               noise_to_targets,
                               zerofy)
from gcsl.algo.positional_encoding import PositionalEncoding

class GPT_IDM(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.dim_out = env.action_space.n
        self.state_dim = env.state_space.shape[0]
        self.state_emb = nn.Linear(8, args.d_model)
        self.encoder = nn.TransformerEncoderLayer(args.d_model, args.nhead, dropout=args.dropout)
        self.gpt = nn.TransformerEncoder(self.encoder, args.layers)
        self.positional_encoder = PositionalEncoding(args.d_model, dropout=0.1, max_len=args.max_len)
        self.out = nn.Linear(args.d_model, self.dim_out)

    def forward(self, X, mask=None, special_pe=False):
        X = einops.rearrange(X, 'b t h -> t b h')
        X = self.state_emb(X)
        if special_pe:
            # add positional encoding, set goal state t-1
            # X = self.positional_encoder(X)
            X[:-1] = X[:-1] + self.positional_encoder.pe[:X.size(0)-1]
            X[-1] = X[-1] + self.positional_encoder.pe[self.args.max_len-1]
        else:
            X = self.positional_encoder(X)
        X = self.gpt(X, mask=mask)
        X = self.out(X)
        return X

class DiscreteIDMPolicy(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.net = GPT_IDM(env, args)
        self.state_hist = deque(maxlen=args.max_len)

    def forward(self, obs, goal=None, horizon=None, mask=None, special_pe=False):
        return self.net(obs, mask=mask, special_pe=special_pe)

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
        mask = generate_square_subsequent_mask(X.shape[1], device=self.device).T if restrictive else None
        X = self.forward(X, mask=mask, special_pe=True)
        logits = X[-2] # -2, as seq order is s_0,...,s_t,s_g and we don't want action
                       # after having reached the goal, but action at s_t towards the goal
        noisy_logits = logits * (1 - noise)
        probs = torch.softmax(noisy_logits, dim=1)
        if greedy:
            samples = torch.argmax(probs, dim=1)
        else: 
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        
        return ptu.to_numpy(samples)

    def nll(self, obs, actions, mask=None, time_state_idxs=None, time_goal_idxs=None): 
        # print(mask[0][0], time_state_idxs[0], time_goal_  idxs[0])
        if self.args.use_traj:
            if self.args.zerofy:
                s_idx = time_state_idxs[0]
                obs = zerofy(obs, time_state_idxs, time_goal_idxs, device=self.device)
            mask = einops.repeat(mask, 'b t1 t2 -> (b repeat) t1 t2', repeat=self.args.nhead)
            out = self.forward(obs, mask=mask)
            logits, targets = self.extract_trajectory(out, actions, time_state_idxs, time_goal_idxs)
        else:
            out = self.forward(obs, mask=mask)
            logits = einops.rearrange(out, 't b c_out -> (t b) c_out')
            targets = einops.rearrange(actions, 'b t -> (t b)')
        targets = noise_to_targets(targets, noise_rate=0.3, device=self.device)
        return F.cross_entropy(logits, targets, reduction='none')

    def extract_trajectory(self, out, actions, time_state_idxs, time_goal_idxs):
        logits = []
        targets = []
        out = einops.rearrange(out, 't b c_out -> b t c_out')
        b_size = out.shape[0]
        for i in range(b_size):
            logits.append(out[i][time_state_idxs[i]:time_goal_idxs[i]+1])
            targets.append(actions[i][time_state_idxs[i]:time_goal_idxs[i]+1])
        logits, _ = einops.pack(logits, '* c_out')
        targets, _ = einops.pack(targets, '*')
        return logits, targets 
