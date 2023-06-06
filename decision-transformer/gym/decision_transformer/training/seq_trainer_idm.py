import torch
import einops
import numpy as np

from decision_transformer.training.trainer import Trainer
from decision_transformer.models.gated_fusion import create_modified_padding_mask, to_2d_mask_idm_pos

class SequenceTrainerIDM(Trainer):
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # src_n_memory_key_padding_mask, pos, first_nonzero = create_modified_padding_mask(attention_mask, self.model.max_length, self.model.device)
        mask, pos = to_2d_mask_idm_pos(attention_mask, 2, self.model.device)
        idxs = torch.arange(states.shape[0], device=self.model.device)
        action_target = actions[idxs, pos]#torch.clone(actions)

        action_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=None, tgt_mask=mask
        )

        action_preds = action_preds[idxs, pos]
        
        
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
