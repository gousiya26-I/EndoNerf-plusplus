import torch
import torch.nn as nn


class FeatureBlender(nn.Module):
    """Blend per-level hash features.

    If time_conditioned=True, the per-level weights are modulated by raw time t.
    Otherwise, a global learnable softmax over levels is used.
    """

    def __init__(self, num_levels, time_conditioned=False, time_hidden_dim=32):
        super().__init__()
        self.num_levels = int(num_levels)
        self.time_conditioned = bool(time_conditioned)

        # Uniform after softmax at initialization.
        self.level_logits = nn.Parameter(torch.zeros(self.num_levels))

        if self.time_conditioned:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, time_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(time_hidden_dim, self.num_levels),
            )

    def forward(self, features, t=None):
        """Blend features along the level dimension.

        Args:
            features: Tensor [..., num_levels, feat_dim] or list/tuple of per-level tensors.
            t: Optional raw time tensor [..., 1]. Only used when time_conditioned=True.
        """
        if isinstance(features, (list, tuple)):
            features = torch.stack(features, dim=-2)

        if features.dim() < 3:
            raise ValueError(
                f"Expected features with shape [..., num_levels, feat_dim], got {features.shape}"
            )
        if features.shape[-2] != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} levels, got {features.shape[-2]}")

        if self.time_conditioned and t is not None:
            if isinstance(t, (list, tuple)):
                t = t[0]
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=features.dtype, device=features.device)
            else:
                t = t.to(device=features.device, dtype=features.dtype)

            if t.dim() == 0:
                t = t.view(1, 1)
            elif t.dim() == 1:
                t = t.unsqueeze(-1)

            t = t[..., :1]
            time_logits = self.time_mlp(t)
            base = self.level_logits.view(*([1] * (time_logits.dim() - 1)), self.num_levels)
            logits = base + time_logits
            weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        else:
            weights = torch.softmax(self.level_logits, dim=0)
            weights = weights.view(*([1] * (features.dim() - 2)), self.num_levels, 1)

        return torch.sum(features * weights, dim=-2)
