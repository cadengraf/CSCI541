"""Dimension-independent geometry transfer plus a training-only projection."""

import torch
from torch import nn
import torch.nn.functional as F


class IdentityDistillation(nn.Module):
    def __init__(self, student_dimensions, teacher_dimensions):
        super().__init__()
        self.projection = nn.Linear(student_dimensions, teacher_dimensions, bias=False)

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, student, teacher, labels):
        student = F.normalize(student.float(), dim=1)
        teacher = F.normalize(teacher.detach().float(), dim=1)
        projected = F.normalize(self.projection(student), dim=1)
        alignment = (1 - (projected * teacher).sum(1)).mean()
        # Equal positive/negative weighting prevents abundant negatives from
        # dominating. Diagonal self-similarities carry no retrieval information.
        error = (student @ student.T - teacher @ teacher.T).square()
        positive = labels[:, None] == labels[None, :]
        positive.fill_diagonal_(False)
        negative = labels[:, None] != labels[None, :]
        if not positive.any() or not negative.any():
            raise ValueError('Distillation needs repeated IDs and different IDs in each batch')
        relation = .5 * error[positive].mean() + .5 * error[negative].mean()
        return alignment, relation


def check_geometry():
    """Known dimension-changing isometry has zero loss; perturbations train only the student."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        student = torch.randn(4, 3, requires_grad=True)
        teacher = F.pad(F.normalize(student.detach(), dim=1), (0, 2)).requires_grad_(True)
        labels = torch.tensor([0, 0, 1, 1])
        loss = IdentityDistillation(3, 5)
        with torch.no_grad():
            loss.projection.weight.zero_()
            loss.projection.weight[:3].copy_(torch.eye(3))
        alignment, relation = loss(student, teacher, labels)
        assert abs(alignment.item()) < 1e-6 and relation.item() < 1e-6
        alignment, relation = loss(student, teacher[[2, 1, 0, 3]], labels)
        (alignment + relation).backward()
        assert alignment.item() > 0 and relation.item() > 0
        assert teacher.grad is None
        assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0
        assert torch.isfinite(loss.projection.weight.grad).all()
