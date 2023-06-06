import torch
import torch.nn.functional as F
import einops
import numpy as np
import random

from decision_transformer.training.trainer import Trainer
from decision_transformer.models.gated_fusion import (create_modified_padding_mask, 
                                                      generate_square_subsequent_mask,
                                                      to_2d_mask, to_2d_mask_idm)
from decision_transformer.models.vicreg_loss import variance_loss, covariance_loss

def get_goal_idxs(first_nonzero, b_size, max_timesteps, device):
    goal_idxs = []
    for i in range(b_size):
        goal_idxs.append(random.randint(first_nonzero[i], max_timesteps-1))
    return torch.tensor(goal_idxs, device=device)

class GCSL_IDM_TRAINER(Trainer):
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        
        first_nonzero = attention_mask.argmax(dim=1)
        idxs = torch.arange(states.shape[0], device=self.model.device)

        goal_idxs = get_goal_idxs(first_nonzero, states.shape[0], states.shape[1], self.model.device)

        s_k_rep, s_g_rep = self.model.forward(
            states[idxs,  first_nonzero], states[idxs, goal_idxs], rtg, mode='representation'
        )

        # BC training
        # action_pred = self.model.forward(
        #     states[idxs,  first_nonzero], states[idxs, goal_idxs], rtg, mode='online'
        # )

        # loss = F.mse_loss(action_pred, actions[idxs, first_nonzero])
        
        # vicreg 
        if self.model.args.vicreg:
            loss = F.mse_loss(s_k_rep, s_g_rep)
            std_loss = .5*variance_loss(s_k_rep) + .5*variance_loss(s_g_rep)
            cov_loss = .5*covariance_loss(s_k_rep) + .5*covariance_loss(s_g_rep)

            lambda_ = self.model.args.vicreg_lambda
            mu = self.model.args.vicreg_mu
            nu = self.model.args.vicreg_nu
            loss = lambda_ * loss + mu * std_loss + nu * cov_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss.detach().cpu().item()

        return loss.detach().cpu().item()
    
    def online_training_step(self, buffer):
        states, actions, rtg, timesteps = buffer.sample(self.batch_size, self.model.args.device)

        first_nonzero = torch.randint(0, states.shape[1]-1, (states.shape[0],), device=self.model.device)
        idxs = torch.arange(states.shape[0], device=self.model.device)

        goal_idxs = get_goal_idxs(first_nonzero+1, states.shape[0], states.shape[1], self.model.device)
        action_preds = self.model.forward(
            states[idxs,  first_nonzero], states[idxs, goal_idxs], rtg, mode='online'
        )

        action_targets = actions[idxs, first_nonzero]
        loss = F.mse_loss(action_preds, action_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item()        

