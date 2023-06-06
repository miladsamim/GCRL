import torch 
import einops
import torch.nn as nn

from decision_transformer.models.gated_fusion import (GatedFusion, 
                                                      generate_square_subsequent_mask, 
                                                      to_2d_mask, to_2d_mask_idm)
from decision_transformer.models.model import TrajectoryModel

class GCSL_IDM(TrajectoryModel):
    def __init__(self, state_dim, act_dim, max_ep_len, args) -> None:
        super().__init__(state_dim, act_dim, max_length=args.K)
        self.args = args
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = args.K
        self.device = args.device


        self.embed_return = nn.Linear(1, args.embed_dim)
        self.embed_state = nn.Linear(self.state_dim, args.embed_dim)
        self.gcsl_net = nn.Sequential(nn.Linear(args.embed_dim*3, 400),
                                        nn.ReLU(),
                                        nn.Linear(400, 300),
                                        nn.ReLU(),
                                        nn.Linear(300, args.embed_dim))
        
        self.expander = nn.Sequential(nn.Linear(args.embed_dim, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, args.embed_dim))
        
        self.representation_predictor = nn.Sequential(nn.Linear(args.embed_dim, 128),
                                                      nn.ReLU(),  
                                                      nn.Linear(128, args.embed_dim))

        self.predict_action = nn.Sequential(nn.Linear(args.embed_dim, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, self.act_dim),
                                            nn.Tanh())
    
    def forward(self, states, goals, rtc, mode='representation'):
        rtc = self.embed_return(rtc[:, 0])
        states = self.embed_state(states)        
        goals = self.embed_state(goals)
        
        if mode=='representation':
            noise1 = torch.randn_like(states).to(self.device)
            noise2 = torch.randn_like(states).to(self.device)
            noise3 = torch.randn_like(states).to(self.device)

            states_rtc_noise, _ = einops.pack((states, rtc, noise1), 'b *')
            noise_noise_goals, _ = einops.pack((noise2, noise3, goals), 'b *')
            
            states_rtc_noise = self.gcsl_net(states_rtc_noise)
            noise_noise_goals = self.gcsl_net(noise_noise_goals)
            
            states_rtc_noise_rep = self.representation_predictor(self.expander(states_rtc_noise))
            noise_noise_goals_rep = self.expander(noise_noise_goals)
            
            return states_rtc_noise_rep, noise_noise_goals_rep
        elif mode == 'inference': 
            noise = torch.randn_like(states).to(self.device)  
            states_rtc_noise, _ = einops.pack((states, rtc, noise), 'b *')
            states_rtc_noise = self.gcsl_net(states_rtc_noise)
            action = self.predict_action(states_rtc_noise)
            return action
        elif mode == 'online':
            noise = torch.randn_like(states).to(self.device)
            state_rtc_goals, _ = einops.pack((states, noise, goals), 'b *')
            state_rtc_goals = self.gcsl_net(state_rtc_goals)
            action = self.predict_action(state_rtc_goals)
            return action
    
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
            in_states = states
            states = states.reshape(1, -1, self.state_dim)
            returns_to_go = returns_to_go.reshape(1, -1, 1)

            if self.max_length is not None:
                states = states[:,-self.max_length:]
                returns_to_go = returns_to_go[:,-self.max_length:]

                # pad all tokens to sequence length
                attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
                attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
                states = torch.cat(
                    [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                    dim=1).to(dtype=torch.float32)
                returns_to_go = torch.cat(
                    [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                    dim=1).to(dtype=torch.float32)
            else:
                attention_mask = None
            
            state = states[:, -1]
            goal = state # we don't have a goal setter
            action_preds = self.forward(state, goal, returns_to_go, mode='inference')
            return action_preds[0]