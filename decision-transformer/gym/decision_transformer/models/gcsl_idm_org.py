import torch 
import einops
import torch.nn as nn

from decision_transformer.models.gated_fusion import (GatedFusion, 
                                                      generate_square_subsequent_mask, 
                                                      to_2d_mask, to_2d_mask_idm)
from decision_transformer.models.model import TrajectoryModel

class GCSL_IDM_OLD(TrajectoryModel):
    def __init__(self, state_dim, act_dim, max_ep_len, args) -> None:
        super().__init__(state_dim, act_dim, max_length=args.K)
        self.args = args
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = args.K
        self.device = args.device

        self.gcsl_net = nn.Sequential(nn.Linear(self.state_dim*2 , 400),
                                        nn.ReLU(),
                                        nn.Linear(400, 300),
                                        nn.ReLU(),
                                        nn.Linear(300, self.act_dim),
                                        nn.Tanh())

    def forward(self, states, goals):
        states_n_goals,_ = einops.pack((states, goals), 'b *')
        action_preds = self.gcsl_net(states_n_goals)
        return action_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
            # we don't care about the past rewards in this model
            states = states.reshape(1, -1, self.state_dim)

            if self.max_length is not None:
                states = states[:,-self.max_length:]

                # pad all tokens to sequence length
                attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
                attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
                states = torch.cat(
                    [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                    dim=1).to(dtype=torch.float32)
            else:
                attention_mask = None

            states = states[:, -1]
            goals = states # we don't have a goal setter

            action_preds = self.forward(states, goals)
            return action_preds[0]

class GCSL_IDM(TrajectoryModel):
    def __init__(self, state_dim, act_dim, max_ep_len, args) -> None:
        super().__init__(state_dim, act_dim, max_length=args.K)
        self.args = args
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = args.K
        self.device = args.device

        self.state_embedding = nn.Sequential(nn.Linear(state_dim, 128))
        self.reward_embedding = nn.Sequential(nn.Linear(1, 128))                                         

        self.gcsl_net = nn.Sequential(nn.Linear(128*2 , 400),
                                        nn.ReLU(),
                                        nn.Linear(400, 300),
                                        nn.ReLU(),
                                        nn.Linear(300, self.act_dim),
                                        nn.Tanh())
        
        self.state_embedding_p = nn.Sequential(nn.Linear(state_dim, 128))
        self.reward_embedding_p = nn.Sequential(nn.Linear(1, 128))                                         

        self.gcsl_net_p = nn.Sequential(nn.Linear(128*2 , 400),
                                        nn.ReLU(),
                                        nn.Linear(400, 300),
                                        nn.ReLU(),
                                        nn.Linear(300, state_dim))
    
    def forward(self, states, rewards):
        states = self.state_embedding(states)
        rewards = self.reward_embedding(rewards)
        states_n_rewards,_ = einops.pack((states, rewards), 'b *')
        action_preds = self.gcsl_net(states_n_rewards)
        return action_preds
    
    def state_prediction(self, states, reward):
        states = self.state_embedding_p(states)
        rewards = self.reward_embedding_p(reward)
        states_n_rewards,_ = einops.pack((states, rewards), 'b *')
        states_preds = self.gcsl_net_p(states_n_rewards)
        return states_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
            # we don't care about the past rewards in this model
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
            
            states = states[:, -1]
            rewards = returns_to_go[:, -1]

            action_preds = self.forward(states, rewards)
            return action_preds[0]