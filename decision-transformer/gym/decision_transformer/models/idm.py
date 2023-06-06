import torch 
import torch.nn as nn 

from einops import rearrange, pack

from decision_transformer.models.gated_fusion import to_2d_mask_idm, GatedFusion
from decision_transformer.models.model import TrajectoryModel

class TransformerIDMSimple(TrajectoryModel):
    def __init__(self, state_dim, act_dim, max_ep_len, args) -> None:
        super().__init__(state_dim, act_dim, max_length=args.K)
        self.args = args
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = args.K
        self.device = args.device
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.embed_dim, nhead=args.n_head, dim_feedforward=args.embed_dim*4, dropout=args.dropout, activation=args.activation_function)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.n_layer)

        self.embed_ln = nn.LayerNorm(args.embed_dim)
        self.embed_state = nn.Linear(self.state_dim, args.embed_dim)
        self.embed_action = nn.Linear(self.act_dim, args.embed_dim)
        self.embed_return = nn.Linear(1, args.embed_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, args.embed_dim)

        self.representation_predictor = nn.Sequential(nn.Linear(2*args.embed_dim, 128),
                                                      nn.ReLU(),  
                                                      nn.Linear(128, 2*args.embed_dim))
        
        self.predict_action = nn.Sequential(nn.Linear(args.embed_dim, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, self.act_dim),
                                            nn.Tanh())
        
        

        self.gated_fusion = GatedFusion(4, args.embed_dim)

        self.expander = nn.Sequential(nn.Linear(args.embed_dim, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, args.embed_dim*2))

    def forward(self, states, actions, rewards, returns_to_go, 
                timesteps, attention_mask=None, use_actions=False, inference=False):
        # embed
        S = self.embed_state(states)
        R = self.embed_return(returns_to_go)
        T = self.embed_timestep(timesteps)
        # add timeemb
        S = S + T
        R = R + T

        if use_actions:
            mask = to_2d_mask_idm(attention_mask, 2, device=self.device, use_idm=False)
            A = self.embed_action(actions) # encode
            A = self.predict_action(A) # decode 
            A = self.embed_action(A) # encode
            A = A + T
            SA, ps = pack((S, A), 'b t *')
            SA = rearrange(SA, 'b t (stack d) -> b (t stack) d', stack=2)
        else:
            mask = to_2d_mask_idm(attention_mask, 1, device=self.device, use_idm=False)
            SA = S
        
        
        SA = self.embed_ln(SA)

        # gated fusion
        SA = self.gated_fusion(SA, R)
        # encode
        SA = rearrange(SA, 'b t d -> t b d')
        out = self.encoder(SA, mask=mask)
        # predict
        S_hidden = out[::2]
        representation = self.expander(S_hidden)
        if use_actions:
            act_pred = self.predict_action(S_hidden[:,:,-self.args.embed_dim:]) # decode
            act_pred = rearrange(act_pred, 't b d -> b t d')
            return act_pred, representation
        else:
            return None, representation
    
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
                    
            action_preds, _ = self.forward(
                states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, use_actions=True, inference=True)
            return action_preds[0, -1]