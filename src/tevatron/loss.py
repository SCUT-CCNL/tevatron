import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributed as dist


class SimpleContrastiveLoss:

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits, target, reduction=reduction)


class SimpleContrastiveLoss4Splade:
    def __init__(self, *args):
        self.q_flops_loss_factor = 0.01
        self.p_flops_loss_factor = 0.01
        self.negatives_x_device = False
        self.world_size = dist.get_world_size() if self.negatives_x_device else 1

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))
        loss = F.cross_entropy(logits, target, reduction=reduction)
        q_flops_loss = self.q_flops_loss_factor * self._flops(x)
        p_flops_loss = self.p_flops_loss_factor * self._flops(y)
        if self.negatives_x_device:
            q_flops_loss *= self.world_size
            p_flops_loss *= self.world_size
        return loss + q_flops_loss + p_flops_loss

    @staticmethod
    def _flops(inputs):
        return torch.sum(torch.mean(torch.abs(inputs), dim=0) ** 2)


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)
