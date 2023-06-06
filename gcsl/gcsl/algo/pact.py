import torch 
import torch.nn as nn
import torch.nn.functional as F

import math 
import numpy as np
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
        self.device = args.device
        self.dim_out = env.action_space.n
        self.state_dim = env.state_space.shape[0]
        self.encoder = nn.TransformerEncoderLayer(args.d_model, args.nhead, dropout=args.dropout, dim_feedforward=args.dim_f)
        self.gpt = nn.TransformerEncoder(self.encoder, args.layers)
        # self.positional_encoder = PositionalEncoding(args.d_model, dropout=0.1, max_len=args.max_len)

    def forward(self, X, mask=None, special_pe=False):
        X = self.gpt(X, mask=mask)
        return X

class DiscretePACTPolicy(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.net = GPT_IDM(env, args)
        self.state_hist = deque(maxlen=args.max_len)
        self.act_hist = deque(maxlen=args.max_len)
        self.state_emb = nn.Linear(8, args.d_model)
        self.goal_emb = nn.Linear(8, args.d_model)
        self.state_out = nn.Sequential(nn.Linear(2*args.d_model, 128),
                                       nn.ReLU(),
                                       nn.Linear(128,128),
                                       nn.ReLU(),
                                       nn.Linear(128, 8))
        self.act_emb = nn.Embedding(self.dim_out, args.d_model)
        self.act_out = nn.Sequential(nn.Linear(args.d_model*2, 128),
                                       nn.ReLU(),
                                       nn.Linear(128,args.d_model),
                                       nn.ReLU())

        self.time_emb = nn.Embedding(args.max_len, args.d_model)

    def forward(self, obs, actions, goals, mask=None, special_pe=False):
        # positional embeddings
        b_sz, t, h = obs.shape
        time_pos = self.time_emb(torch.arange(t, device=self.device))
        time_pos = einops.rearrange(time_pos, 't h -> () t h')
        # To emb
        X_s = self.state_emb(obs)
        X_a = self.act_emb(actions)
        X_g = self.goal_emb(goals)
        # pe 
        X_s = X_s + time_pos
        X_a = X_a + time_pos
        # merge 
        seq, ps = einops.pack((X_s, X_a), 'b t * h') 
        seq = einops.rearrange(seq, 'b t type h -> b (t type) h')
        seq, ps = einops.pack((X_g, seq), 'b * h')
        # gpt 
        seq = einops.rearrange(seq, 'b t h -> t b h')
        seq = self.net(seq, mask=mask)
        seq = einops.rearrange(seq, 't b h -> b t h')
        # unpack
        g, X_h = einops.unpack(seq, [[],[-1]], 'b * h')
        X_h_sa = einops.rearrange(X_h, 'b (t type) h -> b t (type h)', type=2) 
        X_h_s = X_h[:,::2]
        
        state_preds = self.state_out(X_h_sa)
        X_g = einops.repeat(X_g, 'b h -> b repeat h', repeat=t)
        X_h_s, ps = einops.pack((X_h_s,X_g), 'b t *')
        X_h_s = self.act_out(X_h_s)
        act_preds = X_h_s @ self.act_emb.weight.T 

        return state_preds, act_preds

    def reset_state_hist(self):
        self.state_hist = deque(maxlen=self.args.max_len) # as goal uses 3 last places, embs are set at exact match
        self.act_hist = deque(maxlen=self.args.max_len) # above comment

    def act_vectorized(self, obs, prev_act, goal, greedy=False, noise=0):
        return self.act(obs[0], prev_act, goal[0], greedy, noise)
    
    def act(self, obs, prev_act, goal, greedy=False, noise=0):
        self.state_hist.append(obs)
        
        # we don't have act at s_0, just place dummy will not be used for predicting the act anyway        
        self.act_hist.append(prev_act) # s0,_,s1,a0
        l = list(self.act_hist) 
        l = l[1:] # skipe None act
        l.append(0) # append dummy act, to make packing simpler, not used for future act predictions anyway
        X_a = einops.rearrange(np.array(l), 't -> () t')
        X_a = torch.tensor(X_a, dtype=torch.int64, device=self.device)

        X_s = einops.rearrange(list(self.state_hist), 't h -> () t h')
        X_g = einops.rearrange(goal, 'h -> () h')
        X_s = torch.tensor(X_s, dtype=torch.float32, device=self.device)
        X_g = torch.tensor(X_g, dtype=torch.float32, device=self.device)

        state_preds, act_preds = self.forward(X_s, X_a, X_g)
        logits = act_preds[:,-1]
        noisy_logits = logits * (1 - noise)
        probs = torch.softmax(noisy_logits, dim=1)
        if greedy:
            samples = torch.argmax(probs, dim=1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        return ptu.to_numpy(samples)

    def nll(self, obs, actions, **kwargs): 
        k = np.random.randint(3,self.args.max_len)
        mask = generate_square_subsequent_mask(2*k+1, device=self.device)
        goals = obs[:, k]
        obs = obs[:,:k]
        actions = actions[:,:k]

        # forward
        state_preds, act_preds = self.forward(obs, actions, goals, mask=mask)
        # state loss 
        state_loss = F.mse_loss(state_preds[:-1], obs[1:]) # predict one ahead 
        # act loss
        act_preds = einops.rearrange(act_preds, 'b t a_dim -> (b t) a_dim')
        targets = einops.rearrange(actions, 'b t -> (b t)')
        act_loss = F.cross_entropy(act_preds, targets)
        
        loss = state_loss + act_loss
        return loss