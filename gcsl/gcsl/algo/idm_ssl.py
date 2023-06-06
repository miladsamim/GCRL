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
from gcsl.algo.dino_loss import DINOLoss
from gcsl.algo.gated_fusion import GatedFusion

class GPT_S_IDM(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.dim_out = env.action_space.n
        self.state_dim = env.state_space.shape[0]
        self.encoder = nn.TransformerEncoderLayer(args.d_model, args.nhead, dropout=args.dropout)
        self.gpt = nn.TransformerEncoder(self.encoder, args.layers)
        self.positional_encoder = PositionalEncoding(args.d_model, dropout=0.1, max_len=args.max_len)
        # self.out = nn.Linear(args.d_model, self.dim_out)

    def forward(self, X, mask=None, special_pe=False):
        X = einops.rearrange(X, 'b t h -> t b h')
        # if special_pe:
        #     # add positional encoding, set goal state t-1
        #     # X = self.positional_encoder(X)
        #     X[:-1] = X[:-1] + self.positional_encoder.pe[:X.size(0)-1]
        #     X[-1] = X[-1] + self.positional_encoder.pe[self.args.max_len-1]
        # else:
        #     X = self.positional_encoder(X)
        X = self.gpt(X, mask=mask)
        # X = self.out(X)
        return X

class S_DiscreteIDMPolicy_SSL(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.state_emb = nn.Linear(8,args.d_model)
        self.token_emb = nn.Embedding(1, args.d_model)
        # self.state_emb = nn.Sequential(
        #                     nn.Linear(8, 400),
        #                     nn.ReLU(),
        #                     nn.Linear(400, 300),
        #                     nn.ReLU(),
        #                     nn.Linear(300, args.d_model))
        self.goal_emb = nn.Linear(5, args.d_model)
        # self.goal_emb = nn.Sequential(
        #                     nn.Linear(8+5, 400),
        #                     nn.ReLU(),
        #                     nn.Linear(400, 300),
        #                     nn.ReLU(),
        #                     nn.Linear(300, args.d_model))
        # self.net = GPT_S_IDM(env, args)
        # self.dec_layer = nn.TransformerDecoderLayer(args.d_model, args.nhead, dropout=args.dropout, dim_feedforward=args.dim_f)
        # self.decoder = nn.TransformerDecoder(self.dec_layer, args.layers)
        self.transformer = nn.Transformer(d_model=args.d_model, nhead=args.nhead, num_encoder_layers=args.layers,
                                          num_decoder_layers=args.layers, dim_feedforward=args.dim_f, dropout=args.dropout)
        # self.expander = nn.Sequential(
        #                     nn.Linear(self.dim_out, 128),
        #                     nn.ReLU(),
        #                     nn.Linear(128,128),
        #                     nn.ReLU(),
        #                     nn.Linear(128,128))
        # self.net = nn.MultiheadAttention(args.d_model, args.nhead)
        self.out = nn.Linear(args.d_model, self.dim_out)
        # self.out = nn.Sequential(
        #             nn.Linear(args.d_model, 400), #
        #             nn.ReLU(),
        #             nn.Linear(400,300),
        #             nn.ReLU(),
        #             nn.Linear(300, self.dim_out))
        self.state_hist = deque(maxlen=args.max_len)

        self.gated_fusion = GatedFusion(args.d_model)

    def forward(self, obs, goal, mask=None, special_pe=False, use_expander=False):
        # goal, ps = einops.pack((obs, goal), 'b *')
        # obs = self.state_emb(obs)
        # goal = self.goal_emb(goal)
        # cross attn dec 
        s0 = self.state_emb(obs)
        g = self.goal_emb(goal)
        g, ps = einops.pack((obs, goal), 'b *')
        # g = self.goal_emb(g)
        s0 = einops.rearrange(s0, 'b h -> b () h')
        g = einops.rearrange(g, 'b h -> b () h')
        s = self.gated_fusion(s0, g)
        act_token = einops.repeat(self.token_emb.weight, '() h -> repeat h', repeat=obs.shape[0])
        # act_token = self.token_emb.weight + g
        act_token = einops.rearrange(act_token, 'b h -> () b h')
        # memory, ps = einops.pack((s0, g), ' * b h')
        memory = einops.rearrange(s, 'b () h -> () b h')
        # X = self.decoder(act_token, memory)[0]
        X = self.transformer(src=memory, tgt=act_token)[0]

        # idm best
        ####
        # s0 = self.state_emb(obs)
        # g, ps = einops.pack((obs, goal), 'b *')
        # g = self.goal_emb(g)
        # act_token = einops.repeat(self.token_emb.weight, '() h -> repeat h', repeat=obs.shape[0])
        # X, ps = einops.pack((s0,act_token,g), 'b * h')
        ####
        # act_token = einops.rearrange(act_token, 'b h -> () b h')
        # X, ps = einops.pack((s0,g), 'b * h')
        # # ma 
        # key_val, ps = einops.pack((obs, goal), '* b h')
        # b_sz, h = obs.shape
        # obs = einops.rearrange(obs, 'b h -> () b h')
        # act_token = einops.repeat(self.token_emb.weight, '() h -> repeat h', repeat=b_sz)
        # act_token = einops.rearrange(act_token, 'b h -> () b h')
        # X, aw = self.net(act_token,key_val,key_val)
        # X, aw = self.net(obs,key_val,key_val)
        # X = X[0]
        # # X = einops.rearrange(X, 't b h -> b (t h)')
        # X = einops.rearrange(X, 'b h -> b h')
        # X = self.out(X)
        # simple side net + merger net 
        # X, ps = einops.pack((obs, goal), 'b *')
        # X = self.out(X)
        # gpt
        # X, ps = einops.pack((obs, goal), 'b * h')
        # X = self.net(X, mask=mask, special_pe=special_pe)[1]
        X = self.out(X)
        if use_expander:
            X = self.expander(X)
        return X

    def reset_state_hist(self):
        self.state_hist = deque(maxlen=self.args.max_len)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0,
            marginal_policy=None, restrictive=False):
        return self.act(obs, goal, greedy, noise, restrictive)
    
    def act(self, obs, goal, greedy=False, noise=0, restrictive=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        logits = self.forward(obs, goal, special_pe=True, use_expander=False)
        # X = einops.rearrange(X, 't h -> t () h')
        # obs = einops.rearrange(obs, 'h -> () () h')
        # goal = einops.rearrange(goal, 'h -> () () h')
        # X, aw = self.net(obs,goal,goal)
        # X = einops.rearrange(X, 't b h -> b (t h)')
        # logits = self.out(X)
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


    def nll(self, obs, goal, actions, horizon=None):
        preds = self.forward(obs, goal)
        loss = F.cross_entropy(preds, actions, reduction='none') 
        return loss