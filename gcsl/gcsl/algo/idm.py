import torch 
import torch.nn as nn
import torch.nn.functional as F

import math 
import einops
import rlutil.torch.pytorch_util as ptu
import numpy as np

from collections import deque
from gcsl.algo.masking import generate_square_subsequent_mask, noise_to_targets
from gcsl.algo.positional_encoding import PositionalEncoding

class GPT_S_IDM(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.dim_out = env.action_space.n
        self.state_dim = env.state_space.shape[0]
        self.encoder = nn.TransformerEncoderLayer(args.d_model, args.nhead, dropout=args.dropout, dim_feedforward=args.dim_f)
        self.gpt = nn.TransformerEncoder(self.encoder, args.layers)
        self.positional_encoder = PositionalEncoding(args.d_model, dropout=0.1, max_len=args.max_len)

    def forward(self, X, mask=None, special_pe=False):
        # if special_pe:
        #     # add positional encoding, set goal state t-1
        #     # X = self.positional_encoder(X)
        #     X[:-1] = X[:-1] + self.positional_encoder.pe[:X.size(0)-1]
        #     X[-1] = X[-1] + self.positional_encoder.pe[self.args.max_len-1]
        # else:
        #     X = self.positional_encoder(X)
        X = self.gpt(X, mask=mask)
        return X

class S_DiscreteIDMPolicy(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.state_emb = nn.Linear(8,args.d_model)
        self.goal_emb = nn.Linear(5,args.d_model) 
        # self.state_emb = nn.Sequential(
        #                     nn.Linear(8, 400),
        #                     nn.ReLU(),
        #                     nn.Linear(400, 300),
        #                     nn.ReLU(),
        #                     nn.Linear(300, args.d_model))
        # self.goal_emb = nn.Sequential(
        #                     nn.Linear(5, 400),
        #                     nn.ReLU(),
        #                     nn.Linear(400, 300),
        #                     nn.ReLU(),
        #                     nn.Linear(300, args.d_model))
        self.goal_mixer = nn.Sequential(
                              nn.Linear(8+5, 400),
                              nn.ReLU(),
                              nn.Linear(400, 300),
                              nn.ReLU(),
                              nn.Linear(300, args.d_model))
        self.net = GPT_S_IDM(env, args)
        # self.net = nn.MultiheadAttention(args.d_model, args.nhead)
        self.out = nn.Linear(args.d_model, self.dim_out)
        # self.out = nn.Sequential(
        #             nn.Linear(args.d_model, 128), #
        #             nn.ReLU(),
        #             nn.Linear(128,128),
        #             nn.ReLU(),
        #             nn.Linear(128, self.dim_out))
        self.state_hist = deque(maxlen=args.max_len)

    def forward(self, obs, goal=None, horizon=None, mask=None, special_pe=False):
        return self.net(obs, mask=mask, special_pe=special_pe)

    def reset_state_hist(self):
        self.state_hist = deque(maxlen=self.args.max_len    )

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0,
            marginal_policy=None, restrictive=False):
        return self.act(obs[0], goal[0], greedy, noise, restrictive)
    
    def act(self, obs, goal, greedy=False, noise=0, restrictive=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        s0 = self.state_emb(obs)
        # goal = self.goal_emb(goal)
        g, ps = einops.pack((obs, goal), '*')
        g = self.goal_mixer(g)
        X, ps = einops.pack((s0,g), '* h')
        X = einops.rearrange(X, 't h -> t () h')
        # obs = einops.rearrange(obs, 'h -> () () h')
        # goal = einops.rearrange(goal, 'h -> () () h')
        X = self.forward(X, special_pe=True)
        # X, aw = self.net(obs,goal,goal)
        # X = einops.rearrange(X, 't b h -> b (t h)')
        logits = self.out(X[0])
        # logits = X[0]
        # X, ps = einops.pack((obs,goal), '*')
        # logits = self.out(X.unsqueeze(0))
        noisy_logits = logits #* (1 - noise)
        probs = torch.softmax(noisy_logits, dim=1)
        if greedy:
            if np.random.rand() < 0.9*(noise*10):
                samples = torch.distributions.categorical.Categorical(probs=
                                            torch.tensor([[0.25,0.25,0.25,0.25]])).sample()
            else:
                samples = torch.argmax(probs, dim=1)
        else: 
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        return ptu.to_numpy(samples)

    def nll(self, obs, goal, actions, horizon=None, mask=None): 
        s0 = self.state_emb(obs)
        g, ps = einops.pack((obs, goal), 'b *')
        g = self.goal_mixer(g)
        X, ps = einops.pack((s0, g), '* b h')
        X = self.forward(X)
        # obs = einops.rearrange(obs, 'b h -> () b h')
        # goal = einops.rearrange(goal, 'b h -> () b h')
        # X, aw = self.net(obs,goal,goal)
        # X = einops.rearrange(X, 't b h -> b (t h)')
        logits = self.out(X)
        logits = X[0]
        # X, ps = einops.pack((obs, goal), 'b *')
        # logits = self.out(X)
        # logits = einops.rearrange(X, 't b c_out -> b (t c_out)')
        # X = einops.rearrange(X, 'b t c_out -> b (t c_out)')
        # logits = self.out(X)
        targets = einops.rearrange(actions, 'b -> b')       
        # targets = noise_to_targets(targets, noise_rate=.5, device=self.device)
        return F.cross_entropy(logits, targets, reduction='none')       
