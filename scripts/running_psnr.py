import torch
from typing import Optional

class RunningPSNR:
    """Compute the running PSNR during validation step in training."""

    def __init__(self):
        """Constructor."""
        self.N = None
        self.mse_sum = None
        self.max = self.min = None
        self.reset()

    def reset(self):
        """Reset the running PSNR computation."""
        self.mse_sum = 0
        self.N = 0
        self.max = self.min = None

    def update(self, rec: torch.Tensor, tar: torch.Tensor) -> None:
        """Update the running PSNR statistics given a new batch."""
        ins_max = torch.max(tar).item()
        ins_min = torch.min(tar).item()
        if self.max is None:
            assert self.min is None
            self.max = ins_max
            self.min = ins_min
        else:
            self.max = max(self.max, ins_max)
            self.min = min(self.min, ins_min)

        mse = (rec - tar) ** 2
        elementwise_mse = torch.mean(mse.view(len(mse), -1), dim=1)
        self.mse_sum += torch.nansum(elementwise_mse)
        self.N += len(elementwise_mse) - torch.sum(torch.isnan(elementwise_mse))

    def get(self) -> Optional[torch.Tensor]:
        """Get the actual PSNR value given the running statistics."""
        if self.N == 0 or self.N is None:
            return None
        rmse = torch.sqrt(self.mse_sum / self.N)
        return 20 * torch.log10((self.max - self.min) / rmse)