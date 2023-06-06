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

class GCSL_IDM_TRAINER(Trainer):
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        
        first_nonzero = attention_mask.argmax(dim=1)
        idxs = torch.arange(states.shape[0], device=self.model.device)

        action_preds = self.model.forward(
            states[idxs,  first_nonzero], rtg[idxs, -1]
        )

        action_target = actions[idxs, first_nonzero]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        # vicreg 
        if self.model.args.vicreg:
            representations = einops.rearrange(representations, 'b t d -> (b t) d')
            std_loss = variance_loss(representations)
            cov_loss = covariance_loss(representations)
            lambda_ = self.model.args.vicreg_lambda
            mu = self.model.args.vicreg_mu
            nu = self.model.args.vicreg_nu
            loss = lambda_ * loss + mu * std_loss + nu * cov_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        # state prediction training
        states_preds = self.model.state_prediction(states[idxs,  first_nonzero], rtg[idxs, -1])
        states_target = states[idxs, first_nonzero+1]
        self.optimizer.zero_grad()
        state_loss = F.mse_loss(states_preds, states_target)
        state_loss.backward()
        self.optimizer.step()



        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            self.diagnostics['training/state_error'] = torch.mean((states_preds-states_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()