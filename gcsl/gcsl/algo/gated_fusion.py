import torch 
import torch.nn as nn
from einops import rearrange

from gcsl.algo.mha import MultiHeadAttention



class GatedFusion(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(n_head=1, d_model=d_model, d_k=d_model, d_v=d_model)
        self.w_s = nn.Linear(d_model, d_model)
        self.w_g_attn = nn.Linear(d_model, d_model)

    def forward(self, H_states, H_goal):
        # H_states = rearrange(H_states, 't b h -> b t h')
        # H_goal = rearrange(H_goal, 'b h -> b () h')
        H_goal_attn, _ = self.mha(H_states, H_goal, H_goal)
        lambda_ = torch.sigmoid(self.w_s(H_states) + self.w_g_attn(H_goal_attn))
        H_fuse = (1 - lambda_) * H_states + lambda_ * H_goal_attn
        # H_fuse = rearrange(H_fuse, 'b t h -> t b h')
        return H_fuse