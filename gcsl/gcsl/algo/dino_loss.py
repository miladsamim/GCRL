import torch 
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output, update_center=True):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)

        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        loss = loss.mean()
        if update_center:
            self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def polyak(self, teacher, student, tau=0.99, device='cpu'):
        one = torch.ones(1, requires_grad=False).to(device)
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(tau)
            t_param.data.addcmul_(s_param.data, one, value=(1-tau))

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)