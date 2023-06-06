import torch 
import torch.nn as nn

import einops

from decision_transformer.models.gated_fusion import (GatedFusion, 
                                                      generate_square_subsequent_mask, 
                                                      to_2d_mask, to_2d_mask_idm)
from decision_transformer.models.model import TrajectoryModel

class TransformerIDM(TrajectoryModel):
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
        self.embed_ln = nn.LayerNorm(self.args.embed_dim)
        self.embed_state = nn.Linear(self.state_dim, args.embed_dim)
        self.embed_action = nn.Linear(self.act_dim, args.embed_dim)
        self.pred_embed_action = nn.Embedding(1, args.embed_dim)
        self.embed_return = nn.Linear(1, args.embed_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, args.embed_dim)

        self.gated_fusion = GatedFusion(4, args.embed_dim)
    
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, tgt_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # mix state_embeddings and returns_embeddings with a gated fusion
        state_embeddings = self.gated_fusion(state_embeddings, returns_embeddings)
    
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.args.embed_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        # stacked_attention_mask = torch.stack(
        #     (attention_mask, attention_mask, attention_mask), dim=1
        # ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        stacked_inputs = einops.rearrange(stacked_inputs, 'b t d -> t b d')
        transformer_outputs = self.encoder(
            src=stacked_inputs,
            mask=tgt_mask,
        )
        transformer_outputs = einops.rearrange(transformer_outputs, 't b d -> b t d')

        x = transformer_outputs

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.args.embed_dim).permute(0, 2, 1, 3)

        # get predictions
        # return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        # state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,0])  # predict next action given state

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
            
            # tgt_mask = to_2d_mask(attention_mask, 3, device=self.device)
            tgt_mask = to_2d_mask_idm(attention_mask, 2, device=self.device, use_idm=False)

            action_preds = self.forward(
                states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, tgt_mask=tgt_mask, **kwargs)
            return action_preds[0, -1]