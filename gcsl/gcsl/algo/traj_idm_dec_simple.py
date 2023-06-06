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

class DiscreteTrajDEC_IDMSinglePolicy(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.args = args
        self.device = args.device
        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        # self.encoder = nn.TransformerDecoderLayer(args.d_model, args.nhead, dropout=args.dropout, dim_feedforward=args.dim_f)
        # self.net = nn.TransformerDecoder(self.encoder, args.layers)
        self.net = nn.Transformer(d_model=args.d_model, nhead=args.nhead, num_encoder_layers=args.layers,
                                  num_decoder_layers=args.layers, dim_feedforward=args.dim_f, dropout=args.dropout)
        self.state_hist = deque(maxlen=args.max_len)
        self.act_token = nn.Embedding(1, args.d_model)
        # self.pos_embd = nn.Embedding(args.max_len, args.d_model)
        self.state_embd = nn.Linear(8, args.d_model)
        self.goal_emb = nn.Sequential(
                            nn.Linear(8+5, 400),
                            nn.ReLU(),
                            nn.Linear(400, 300),
                            nn.ReLU(),
                            nn.Linear(300, args.d_model))
        self.out = nn.Linear(args.d_model, self.dim_out)

    def forward(self, obs, mask=None, t1=None, t2=None):
        b_size, t, h = obs.size()
        idxs = torch.arange(b_size, device=self.device)
        X_g = self.env.extract_goal(obs[idxs, t2])
        obs = obs[idxs, t1]
        X_g, ps = einops.pack((obs, X_g), 'b *')
        X_g = self.goal_emb(X_g)
        obs = self.state_embd(obs)
        memory, _ = einops.pack((obs, X_g), '* b h')
        tgt = einops.repeat(self.act_token.weight, '() h -> repeat h', repeat=b_size)
        tgt = einops.rearrange(tgt, 'b h -> () b h')

        tgt = self.net(src=memory, tgt=tgt)
        logits = self.out(tgt)
        logits = einops.rearrange(logits, 't b h -> b t h')
        return logits

    def reset_state_hist(self):
        self.state_hist = deque(maxlen=self.args.max_len)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0,
            marginal_policy=None, restrictive=False):
        return self.act(obs, goal, greedy, noise, restrictive)
    
    def act(self, obs, goal, greedy=False, noise=0, restrictive=False):
        self.state_hist.append(obs)
        X, ps = einops.pack((obs, goal), '* h')
        X = einops.rearrange(X, 't h -> () t h')
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = len(self.state_hist)
        t1 = np.array([0]) 
        t2 = np.array([1]) 
        # t1 = np.array([n-1]) 
        # t2 = np.array([self.args.max_len]) # always decode all the way toward goal pos 
        X = self.forward(X, t1=t1, t2=t2)
        logits = X[:,0] # first decoded action
        noisy_logits = logits * (1 - noise)
        probs = torch.softmax(noisy_logits, dim=1)
        if greedy:
            samples = torch.argmax(probs, dim=1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        return ptu.to_numpy(samples)

    def nll(self, obs, actions, mask=None, time_state_idxs=None, time_goal_idxs=None):
        out = self.forward(obs, mask=mask, t1=time_state_idxs, t2=time_goal_idxs)
        logits, targets = self.extract_trajectory(out, actions, time_state_idxs, time_goal_idxs)
        return F.cross_entropy(logits, targets, reduction='mean')

    def extract_trajectory(self, out, actions, t1, t2):
        b_size = actions.shape[0]
        # simplification single act only 
        idxs = torch.arange(b_size, device=self.device)
        targets = actions[idxs, t1]
        out = einops.rearrange(out, 'b t c_out -> (b t) c_out')
        return out, targets 
