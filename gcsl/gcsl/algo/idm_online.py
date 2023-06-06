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
        self.encoder = nn.TransformerEncoderLayer(args.d_model, args.nhead, dropout=args.dropout)
        self.gpt = nn.TransformerEncoder(self.encoder, args.layers)
        self.positional_encoder = PositionalEncoding(args.d_model, dropout=0.1, max_len=args.max_len+1)

    def forward(self, X, mask=None, special_pe=False):
        X = einops.rearrange(X, 't b h -> t b h')
        if special_pe:
            # add positional encoding, set goal state t-1
            # X = self.positional_encoder(X)
            X[:-1] = X[:-1] + self.positional_encoder.pe[:X.size(0)-1]
            X[-1] = X[-1] + self.positional_encoder.pe[self.args.max_len-1]
        else:
            X = self.positional_encoder(X)
        X = self.gpt(X, mask=mask)
        return X

class DiscreteOnlineIDMPolicy(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.net = GPT_IDM(env, args)
        self.max_len = args.max_len
        self.state_hist = deque(maxlen=args.max_len)
        self.register_buffer('s_predicted_reach', torch.rand(self.args.d_model))
        self.s_predicted_reach.requires_grad = False

        self.state_emb = nn.Linear(8, args.d_model)
        self.act_out = nn.Linear(args.d_model, self.dim_out)
        self.state_out = nn.Linear(args.d_model, 8)

    def forward(self, obs, goal=None, horizon=None, mask=None, special_pe=False):
        return self.net(obs, mask=mask, special_pe=special_pe)

    def reset_state_hist(self):
        self.state_hist = deque(maxlen=self.args.max_len)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0,
            marginal_policy=None, restrictive=False):
        return self.act(obs, goal, greedy, noise, restrictive)
    
    def act(self, obs, goal, greedy=False, noise=0, restrictive=False):
        """Get s_0 and g, generate acts between + s_r, add pe to s_0,acts,g, 
           for s_r add special token"""
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        X = torch.zeros((self.max_len+1,1,self.args.d_model), dtype=torch.float32, device=self.device)
        X[0] = self.state_emb(obs)
        X[self.max_len-1] = self.state_emb(goal)
        X[self.max_len,0] = self.s_predicted_reach 

        X = self.forward(X, mask=None, special_pe=False)
        X_act = self.act_out(X[:self.max_len])
        X_act = einops.rearrange(X_act, 't () c_out -> t c_out')
        X_act = X_act.argmax(dim=1)
        X_s_predicted_reach = self.state_out(X[self.max_len])
        X_s_predicted_reach = einops.rearrange(X_s_predicted_reach, '() h -> h')
        
        return ptu.to_numpy(X_act), X_s_predicted_reach
    
    def online_loss(self, X_s_predicted_reach, goal, actual_reached):
        goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        actual_reached = torch.tensor(actual_reached, dtype=torch.float32, device=self.device)
        goal_loss = F.mse_loss(X_s_predicted_reach, goal)
        reach_loss = F.mse_loss(X_s_predicted_reach, actual_reached)
        loss = goal_loss + reach_loss
        return loss 

    def nll(self, obs, actions, mask=None, time_state_idxs=None, time_goal_idxs=None): 
        # print(mask[0][0], time_state_idxs[0], time_goal_idxs[0])
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
