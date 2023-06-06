from typing import Tuple
import torch 
import torch.nn as nn

from einops import rearrange, repeat
import random 
import numpy as np

def to_2d_mask_idm_pos(mask, repeats, device='cpu', use_idm=True):
    K = mask.shape[1]
    first_nonzero_pre = mask.argmax(dim=1)
    mask = repeat(mask, 'n t -> (n repeat) t', repeat=repeats)
    mask = rearrange(mask, '(n repeat) t -> n (t repeat)', repeat=repeats)
    mask_2d = generate_square_subsequent_mask(K*repeats, device=device)
    # mask_2d = torch.zeros(K*repeats, K*repeats, device=device)
    mask_2d = repeat(mask_2d, 't1 t2 -> b t1 t2', b=mask.shape[0]).clone()
    first_nonzero = mask.argmax(dim=1)
    idxs = torch.arange(K*repeats, device=device)
    positions = torch.zeros(mask.shape[0], device=device, dtype=torch.long)
    for i in range(mask.shape[0]):
        mask_2d[i, first_nonzero[i]:, :first_nonzero[i]] = float('-inf') # only really need to ensure those after pad do not attend to pad 
        positions[i] = random.randint(first_nonzero_pre[i], K-1)
        pos = int(positions[i])*repeats
        # if use_idm: # only in training
        #     mask_2d[i, :, pos+1:] = float('-inf')
        # mask_2d[i, idxs, idxs] = 0 # set diagonal to 0 to avoid nan collapse
    
    return mask_2d, positions

def to_2d_mask_idm(mask, repeats, device='cpu', use_idm=True):
    K = mask.shape[1]
    mask = repeat(mask, 'n t -> (n repeat) t', repeat=repeats)
    mask = rearrange(mask, '(n repeat) t -> n (t repeat)', repeat=repeats)
    mask_2d = generate_square_subsequent_mask(K*repeats, device=device)
    # mask_2d = torch.zeros(K*repeats, K*repeats, device=device)
    mask_2d = repeat(mask_2d, 't1 t2 -> b t1 t2', b=mask.shape[0]).clone()
    first_nonzero = mask.argmax(dim=1)
    idxs = torch.arange(K*repeats, device=device)

    for i in range(mask.shape[0]):
        mask_2d[i, first_nonzero[i]:, :first_nonzero[i]] = float('-inf') # only really need to ensure those after pad do not attend to pad 
        pos = random.randint(first_nonzero[i], K*repeats-1)
        if use_idm: # only in training
            mask_2d[i, :, pos+1:] = float('-inf')
        # mask_2d[i, idxs, idxs] = 0 # set diagonal to 0 to avoid nan collapse
    
    return mask_2d

def to_2d_mask(mask, repeats, device='cpu'):
    K = mask.shape[1]
    mask = repeat(mask, 'n t -> (n repeat) t', repeat=3)
    mask = rearrange(mask, '(n repeat) t -> n (t repeat)', repeat=3)
    mask_2d = generate_square_subsequent_mask(K*repeats, device=device)
    # mask_2d = torch.zeros(K*repeats, K*repeats, device=device)
    mask_2d = repeat(mask_2d, 't1 t2 -> b t1 t2', b=mask.shape[0]).clone()
    first_nonzero = mask.argmax(dim=1)
    idxs = torch.arange(K*repeats, device=device)

    for i in range(mask.shape[0]):
        mask_2d[i, :, :first_nonzero[i]] = float('-inf')
        mask_2d[i, idxs, idxs] = 0 # set diagonal to 0 to avoid nan collapse
    
    return mask_2d

def generate_square_subsequent_mask(sz: int, device='cpu') -> torch.Tensor:
    return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

def create_modified_padding_mask(mask: torch.Tensor, K: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a modified memory_key_padding_mask tensor based on the original mask tensor.
    
    Args:
        mask (torch.Tensor): A tensor containing the original mask.
        K (int): The sequence length.
        device (torch.device): The device (CPU or GPU) to which tensors should be moved.
    
    Returns:
        memory_key_padding_mask (torch.Tensor): The modified padding mask tensor.
        pos (torch.Tensor): The tensor containing randomly generated positions.
    """
    # find first nonzero entry in mask
    memory_key_padding_mask = mask.clone()
    first_nonzero = mask.argmax(dim=1)

    # find random position pos between first_nonzero and K and mask[pos:]
    pos = []
    for i in range(mask.size(0)):
        pos.append(random.randint(first_nonzero[i], K-1))
        memory_key_padding_mask[i, pos[i]+1:] = 0
    pos = torch.tensor(pos).to(device=device)

    return memory_key_padding_mask, pos, first_nonzero

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        residual = q
        q = rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        v = rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', [q, k]) / np.sqrt(q.shape[-1])
        if mask is not None:
            attn = attn.masked_fill(mask[None], -np.inf)
        attn = torch.softmax(attn, dim=3)
        output = torch.einsum('hblt,hbtv->hblv', [attn, v])
        output = rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn

class GatedFusion(nn.Module):
    def __init__(self, n_head, d_model) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_model, d_v=d_model)
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