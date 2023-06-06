import torch 
import torch.nn as nn
import torch.nn.functional as F

import math 
import einops
import rlutil.torch.pytorch_util as ptu

from collections import deque
from gcsl.algo.masking import generate_square_subsequent_mask, generate_restrictive_square_subsequent_mask
from gcsl.algo.positional_encoding import PositionalEncoding

class DEC_GPT_IDM(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.dim_out = env.action_space.n
        self.state_dim = env.state_space.shape[0]
        self.state_emb = nn.Linear(8, args.d_model)
        self.decoder = nn.TransformerDecoderLayer(args.d_model, args.nhead, dropout=args.dropout)
        self.gpt = nn.TransformerDecoder(self.decoder, args.layers)
        self.positional_encoder = PositionalEncoding(args.d_model, dropout=0.1, max_len=args.max_len)
        self.act_emb = nn.Embedding(1, args.d_model)
        self.act_emb.weight.requires_grad = False 
        self.out = nn.Linear(args.d_model, self.dim_out)

    def forward(self, X, mask=None, special_pe=False, t_idx=None):
        b_size, t, _ = X.shape
        X = einops.rearrange(X, 'b t h -> t b h')
        X = self.state_emb(X)
        if special_pe:
            assert t_idx is not None
            # add positional encoding, set goal state t-1
            # X = self.positional_encoder(X)
            X[:-1] = X[:-1] + self.positional_encoder.pe[t_idx]
            X[-1] = X[-1] + self.positional_encoder.pe[self.args.max_len-1]
        else:
            X = self.positional_encoder(X)

        act_X = self.act_emb(torch.zeros(t, b_size, dtype=torch.int, device=self.args.device))
        act_mask = generate_restrictive_square_subsequent_mask(t, device=self.args.device)

        X = self.gpt(tgt=act_X, memory=X, tgt_mask=act_mask,
                     memory_mask=mask)
        X = self.out(X)
        return X

class DiscreteDEC_IDMPolicy(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.net = DEC_GPT_IDM(env, args)
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
        X, ps = einops.pack((obs, goal), '* h') # only use last + goal (restrictive)
        X = einops.rearrange(X, 't h -> () t h')
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        mask = generate_square_subsequent_mask(X.shape[1], device=self.device).T if restrictive else None
        X = self.net(X, mask=mask, special_pe=True, t_idx=len(self.state_hist)-1)
        logits = X[-2] # -2, as seq order is s_0,...,s_t,s_g and we don't want action
                       # after having reached the goal, but action at s_t towards the goal
        noisy_logits = logits * (1 - noise)
        probs = torch.softmax(noisy_logits, dim=1)
        if greedy:
            samples = torch.argmax(probs, dim=1)
        else: 
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        
        return ptu.to_numpy(samples)

    def nll(self, obs, goal, actions, horizon=None, mask=None): 
        logits = self.forward(obs, mask=mask)
        logits = einops.rearrange(logits, 't b c_out -> (t b) c_out')
        targets = einops.rearrange(actions, 'b t -> (t b)')       
        return F.cross_entropy(logits, targets, reduction='none')       
