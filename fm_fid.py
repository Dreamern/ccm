import torch
from torch import Tensor
from typing import Any
import torch.distributed
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class FmFID(FrechetInceptionDistance):
    def __init__(self, feature: int = 2048, reset_real_features: bool = True, normalize: bool = False, **kwargs: Any) -> None:
        mean: Tensor
        std: Tensor
        num_samples: Tensor

        super().__init__(feature, reset_real_features, normalize, **kwargs)

    def compute_with_ref(self, mu=None, sigma=None) -> Tensor:
        
        torch.distributed.all_reduce(self.fake_features_sum)
        torch.distributed.all_reduce(self.fake_features_num_samples)
        torch.distributed.all_reduce(self.fake_features_cov_sum)
        
        local_rank = self.fake_features_num_samples.device
        mean_real = torch.tensor(mu).double().to(local_rank)
        cov_real = torch.tensor(sigma).double().to(local_rank)

        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        fid = _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)
        return fid.item()
