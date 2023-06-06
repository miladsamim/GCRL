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

class SequenceTrainerIDMFull(Trainer):
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # key_padding_mask, pos, first_nonzero = create_modified_padding_mask(attention_mask, self.model.max_length, self.model.device)
        # idxs = torch.arange(states.shape[0], device=self.model.device)
        action_target = torch.clone(actions)
        # tgt_mask = to_2d_mask(attention_mask, 3, device=self.model.device)
        tgt_mask = to_2d_mask_idm(attention_mask, 2, device=self.model.device)

        action_preds, representations = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=None, tgt_mask=tgt_mask
        )

        # single dec training
        first_nonzero = attention_mask.argmax(dim=1)
        idxs = torch.arange(states.shape[0], device=self.model.device)
        action_preds = action_preds[idxs, first_nonzero]
        action_target = action_target[idxs, first_nonzero]  
        

        # act_dim = action_preds.shape[2] 
        # action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        # single act 
        # action_preds = action_preds.reshape(-1, self.model.max_length, self.model.act_dim)
        # idxs = torch.arange(action_preds.shape[0], device=self.model.device)
        # pos = random.randint(0, self.model.max_length-1)

        # loss = self.loss_fn(
        #     None, action_preds[idxs, pos], None,
        #     None, action_target[idxs, pos], None,
        # )

        # vicreg 
        if self.model.args.vicreg:
            # representations = einops.rearrange(representations, 'b t d -> (b t) d')
            # temporal consistency
            reps = einops.rearrange(representations, 'b (t t_window) d -> b t t_window d', t_window=2)
            reps_ = einops.rearrange(reps, 'b t t_window d -> t_window (b t) d')
            reps_loss = F.mse_loss(reps_[:-1], reps_[1:])
            loss = loss + reps_loss
            # single timestep difference
            reps_ = reps[:,-1,-1,:]
            
            std_loss = variance_loss(reps_)
            cov_loss = covariance_loss(reps_)
            lambda_ = self.model.args.vicreg_lambda
            mu = self.model.args.vicreg_mu
            nu = self.model.args.vicreg_nu
            loss = lambda_ * loss + mu * std_loss + nu * cov_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
