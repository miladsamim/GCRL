import torch 
import torch.nn as nn

import einops

from decision_transformer.models.gated_fusion import GatedFusion
from decision_transformer.models.model import TrajectoryModel

class TransformerIDMFull(TrajectoryModel):
    def __init__(self, state_dim, act_dim, max_ep_len, args) -> None:
        super().__init__(state_dim, act_dim, max_length=args.K)
        self.args = args
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = args.K
        self.device = args.device
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.embed_dim, nhead=args.n_head, dim_feedforward=args.embed_dim*4, dropout=args.dropout, activation=args.activation_function)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.n_layer)

        self.predict_action = nn.Sequential(nn.Linear(args.embed_dim, self.act_dim),
                                    nn.Tanh())
        
        self.embed_state = nn.Linear(self.state_dim, args.embed_dim)
        self.embed_action = nn.Linear(self.act_dim, args.embed_dim)
        self.pred_embed_action = nn.Embedding(1, args.embed_dim)
        self.embed_return = nn.Linear(1, args.embed_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, args.embed_dim)
    
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, tgt_mask=None):
        # assert attention_mask is not None 
        rewards = None # not used
        n, t, d = states.shape

        timesteps = einops.rearrange(timesteps, 'n t -> t n')
        states = einops.rearrange(states, 'n t d -> t n d')
        actions = einops.rearrange(actions, 'n t d -> t n d')
        returns_to_go = einops.rearrange(returns_to_go, 'n t d -> t n d')

        # embed
        time_embds = self.embed_timestep(timesteps)
        state_embds = self.embed_state(states)
        action_embds = self.embed_action(actions)
        returns_embds = self.embed_return(returns_to_go)

        state_embds = state_embds + time_embds # add time embedding 
        action_embds = action_embds + time_embds # add time embedding
        return_embds = returns_embds + time_embds # add time embedding

        # interleave state, action, return
        stacked_token_embeds, ps =  einops.pack((state_embds, action_embds, returns_embds), 't n *')
        stacked_token_embeds = einops.rearrange(stacked_token_embeds, 't n (d1 d2) -> (t d1) n d2', d1=3)

        hidden_states = self.encoder(src=stacked_token_embeds, mask=tgt_mask)

        states = hidden_states[::3] # get states

        action_preds = self.predict_action(states)

        return action_preds
    
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
            # we don't care about the past rewards in this model
            states = states.reshape(1, -1, self.state_dim)
            actions = actions.reshape(1, -1, self.act_dim)
            returns_to_go = returns_to_go.reshape(1, -1, 1)
            timesteps = timesteps.reshape(1, -1)

            if self.max_length is not None:
                states = states[:,-self.max_length:]
                actions = actions[:,-self.max_length:]
                returns_to_go = returns_to_go[:,-self.max_length:]
                timesteps = timesteps[:,-self.max_length:]

                # pad all tokens to sequence length
                attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
                attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
                states = torch.cat(
                    [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                    dim=1).to(dtype=torch.float32)
                actions = torch.cat(
                    [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                                device=actions.device), actions],
                    dim=1).to(dtype=torch.float32)
                returns_to_go = torch.cat(
                    [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                    dim=1).to(dtype=torch.float32)
                timesteps = torch.cat(
                    [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                    dim=1
                ).to(dtype=torch.long)
            else:
                attention_mask = None

            action_preds = self.forward(
                states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
            return action_preds[0, -1]