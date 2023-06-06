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
from gcsl.algo.mha import MultiHeadAttention

class GatedFusion(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(n_head=1, d_model=d_model, d_k=d_model, d_v=d_model)
        self.w_s = nn.Linear(d_model, d_model)
        self.w_g_attn = nn.Linear(d_model, d_model)

    def forward(self, H_states, H_goal):
        H_states = einops.rearrange(H_states, 't b h -> b t h')
        H_goal = einops.rearrange(H_goal, 'b h -> b () h')
        H_goal_attn, _ = self.mha(H_states, H_goal, H_goal)
        lambda_ = torch.sigmoid(self.w_s(H_states) + self.w_g_attn(H_goal_attn))
        H_fuse = (1 - lambda_) * H_states + lambda_ * H_goal_attn
        H_fuse = einops.rearrange(H_fuse, 'b t h -> t b h')
        return H_fuse

class FiLM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(nn.Linear(d_model, 128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Linear(128, 2*d_model))
    
    def forward(self, F, X):
        F = einops.rearrange(F, 't b h -> t b h')
        X = einops.rearrange(X, 'b h -> () b h')
        P = self.net(X)
        gamma, beta = P[:,:,:self.d_model], P[:,:,self.d_model:]
        F = gamma * F + beta + F    
        return F  


class DiscreteTrajDEC_IDMPolicyClawContinous(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.args = args
        self.device = args.device
        self.action_space = env.action_space
        self.dim_out = 9#env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.goal_dim = env.goal_space.shape[0]
        self.net = nn.Transformer(d_model=args.d_model, nhead=args.nhead, num_encoder_layers=args.layers,
                                  num_decoder_layers=args.layers, dim_feedforward=args.dim_f, norm_first=args.norm_first,
                                  dropout=args.dropout)
        self.state_hist = deque(maxlen=args.max_len)
        self.act_token = nn.Embedding(1, args.d_model)
        # self.pos_embd = nn.Embedding(args.max_len, args.d_model)
        # self.pos_embd = PositionalEncoding(args.d_model, dropout=0.1, max_len=args.max_len+1)
        self.state_embd = nn.Linear(self.state_dim, args.d_model)
        # self.goal_emb = nn.Linear(self.goal_dim, args.d_model)
        self.goal_emb = nn.Sequential(nn.Linear(self.state_dim+self.goal_dim, 400),
                                      nn.ReLU(),
                                      nn.Linear(400, 300),
                                      nn.ReLU(),
                                      nn.Linear(300, args.d_model))
        # self.goal_film = FiLM(args.d_model)
        self.gated_fusion = GatedFusion(args.d_model)

        self.out = nn.Sequential(nn.Linear(args.d_model, self.dim_out), nn.Tanh())

        self.expander = nn.Sequential(nn.Linear(args.d_model, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 64))

    def forward(self, obs, mask=None, t1=None, t2=None, special=False, X_g=None, expand=False):
        b_size, t, h = obs.size()
        idxs = torch.arange(b_size, device=self.device)
        if special:
            X_g = einops.rearrange(X_g, 'h -> () h')
        else:
            X_g = self.env.extract_goal(obs[idxs, t2])
            obs = self.env.observation(obs)

        X_g, ps = einops.pack((obs[idxs, t1], X_g), 'b *')
        X_g = self.goal_emb(X_g)
        obs = self.state_embd(obs)  
        # add pos encoding only to states, not goal, after state emb are fully computed
        obs = einops.rearrange(obs, 'b t h -> t b h')
        # obs = self.pos_embd(obs)
        # obs = self.goal_film(obs, X_g)
        obs = self.gated_fusion(obs, X_g)
        obs[t2, idxs] = X_g # s0,...,sk,...,sg,...,sT
        ## concat approach
        # X_g = einops.repeat(X_g, 'b h -> repeat b h', repeat = t)
        # obs, ps = einops.pack((obs, X_g), 't b *') 
        
        state_pad_masks = self.make_state_pad_masks(b_size, t, t1, t2, device=self.device)

        tgt = einops.repeat(self.act_token.weight, '() h -> repeat h', repeat=b_size)
        tgt = einops.rearrange(tgt, 'b h -> () b h')
        # obs = einops.rearrange(obs, 'b t h -> t b h')
        tgt = self.net(src=obs, tgt=tgt, src_key_padding_mask=state_pad_masks, memory_key_padding_mask=state_pad_masks)
        acts = self.out(tgt)
        acts = einops.rearrange(acts, 't b h -> b t h')
        if expand:
            expansion = self.expander(tgt)
            expansion = einops.rearrange(expansion, 't b h -> b t h')
            return acts, expansion
        else:
            return acts

    def reset_state_hist(self):
        self.state_hist = deque(maxlen=self.args.max_len)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0,
            marginal_policy=None, restrictive=False):
        return self.act(obs[0], goal[0], greedy, noise, restrictive)
    
    def act(self, obs, goal, greedy=False, noise=0, restrictive=False):
        self.state_hist.append(obs)
        X = einops.rearrange(list(self.state_hist), 't h -> t h')
        X_g = torch.tensor(goal, dtype=torch.float32, device=self.device)
        X, ps  = einops.pack((X,np.zeros(self.state_dim)), '* h') 
        X = einops.rearrange(X, 't h -> () t h')
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = len(self.state_hist)
        # t1 = np.array([0])
        # t2 = np.array([1]) 
        t1 = np.array([n-1], dtype=np.int64)
        t2 = np.array([n], dtype=np.int64) 
        X = self.forward(X, t1=t1, t2=t2, special=True, X_g=X_g)
        act = X[:,0] # first decoded action
        return ptu.to_numpy(act)

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
    
    def make_state_pad_masks(self, b_size:int, sz:int, t1:np.array, t2:np.array, device='cpu') -> torch.Tensor: 
        src_key_padding_masks = []
        for i in range(b_size):
            src_key_padding_mask = [False] * (t1[i]+1) # +1 to also unmask s_k
            src_key_padding_mask += [True] * (sz - (t1[i]+1))
            # simplification to test for unmask s_k and s_g
            # src_key_padding_mask = [True] * sz 
            # src_key_padding_mask[t1[i]] = False
            src_key_padding_mask[t2[i]] = False
            src_key_padding_masks.append(src_key_padding_mask)
        src_key_padding_masks = np.asarray(src_key_padding_masks)
        return torch.tensor(src_key_padding_masks, device=device)
