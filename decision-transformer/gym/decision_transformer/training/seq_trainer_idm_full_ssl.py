import torch
import torch.nn.functional as F
import einops
import numpy as np
import random

from decision_transformer.training.trainer import Trainer
from decision_transformer.models.vicreg_loss import variance_loss, covariance_loss

class SequenceTrainerIDMFullSSL(Trainer):
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        action_preds, representations = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, use_actions=False
        )

        # vicreg 
        if self.model.args.vicreg:
            # temporal consistency
            reps = einops.rearrange(representations, 'b (t t_window) d -> b t t_window d', t_window=2)
            reps_ = einops.rearrange(reps, 'b t t_window d -> t_window (b t) d')

            # use predictor: predictor(t_i) = t_i+1
            loss = F.mse_loss(self.model.representation_predictor(reps_[:-1]), reps_[1:])
            # single timestep difference
            # reps_ = reps[:,-1,-1,:]
            reps_ = einops.rearrange(reps_, 't_window (bt) d -> (t_window bt) d')
            
            std_loss = variance_loss(reps_)
            cov_loss = covariance_loss(reps_)
            lambda_ = self.model.args.vicreg_lambda
            mu = self.model.args.vicreg_mu
            nu = self.model.args.vicreg_nu
            loss = lambda_ * loss + mu * std_loss + nu * cov_loss
        else:
            print("ERROR: MUST USE VICREG FOR IDM FULL")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss.detach().cpu().item()

        return loss.detach().cpu().item()

    def online_training_step(self, buffer):
        states, actions, rtg, timesteps = buffer.sample(self.batch_size, self.model.args.device)
        attention_mask = torch.ones_like(timesteps, dtype=torch.float32).to(self.model.args.device)

        action_preds, representations = self.model.forward(
            states, actions, None, rtg, timesteps, attention_mask=attention_mask, use_actions=True, inference=False
        )

        # temporal consistency
        reps = einops.rearrange(representations, 'b (t t_window) d -> b t t_window d', t_window=2)
        reps_ = einops.rearrange(reps, 'b t t_window d -> t_window (b t) d')
        loss = F.mse_loss(reps_[:-1], reps_[1:])
        # single timestep difference
        # reps_ = reps[:,-1,-1,:]
        reps_ = einops.rearrange(reps_, 't_window bt d -> (t_window bt) d')
        
        std_loss = variance_loss(reps_)
        cov_loss = covariance_loss(reps_)
        lambda_ = self.model.args.vicreg_lambda
        mu = self.model.args.vicreg_mu
        nu = self.model.args.vicreg_nu

        # action ar loss 
        action_preds = action_preds[:,:-1,:] # remove last timestep
        action_target = actions[:,1:,:] # remove first timestep
        action_loss = F.mse_loss(action_preds, action_target)
        loss = .1 * action_loss + loss 
        loss = lambda_ * loss + mu * std_loss + nu * cov_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item()        
