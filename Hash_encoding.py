import math

import torch
import torch.nn as nn


class MultiResHashEncoder(nn.Module):
    """Multi-resolution hash-grid encoder for 3D canonical coordinates.

    Input shape: [..., 3]
    Output shape: [..., num_levels, features_per_level]
    """

    def __init__(
        self,
        num_levels=16,
        features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
    ):
        super().__init__()
        self.num_levels = int(num_levels)
        self.features_per_level = int(features_per_level)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.hashmap_size = 1 << self.log2_hashmap_size
        self.base_resolution = int(base_resolution)
        self.finest_resolution = int(finest_resolution)

        if self.num_levels < 1:
            raise ValueError("num_levels must be >= 1")

        if self.num_levels == 1:
            self.b = 1.0
        else:
            self.b = math.exp(
                (math.log(self.finest_resolution) - math.log(self.base_resolution))
                / (self.num_levels - 1)
            )

        self.tables = nn.ModuleList(
            [nn.Embedding(self.hashmap_size, self.features_per_level) for _ in range(self.num_levels)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for table in self.tables:
            nn.init.uniform_(table.weight, a=-1e-4, b=1e-4)

    @staticmethod
    def _hash(coords: torch.Tensor, hashmap_size: int) -> torch.Tensor:
        if coords.shape[-1] != 3:
            raise ValueError(f"Expected coords[..., 3], got {coords.shape}")

        coords = coords.to(torch.long)
        x, y, z = coords.unbind(dim=-1)
        hashed = x * 1_540_863_946
        hashed ^= y * 1_257_487_969
        hashed ^= z * 1_034_312_349
        hashed = torch.remainder(hashed, hashmap_size)
        return hashed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 3:
            raise ValueError(f"MultiResHashEncoder expects [..., 3] inputs, got {x.shape}")

        orig_shape = x.shape[:-1]
        x = x.reshape(-1, 3)

        device = x.device
        dtype = x.dtype
        level_feats = []

        for lvl, table in enumerate(self.tables):
            resolution = int(math.floor(self.base_resolution * (self.b ** lvl)))
            resolution = max(resolution, 1)

            scaled = x * resolution - 0.5
            base = torch.floor(scaled)
            frac = scaled - base
            base = base.to(torch.long)

            feats = torch.zeros(x.shape[0], self.features_per_level, device=device, dtype=dtype)
            for ox in (0, 1):
                wx = frac[:, 0] if ox == 1 else (1.0 - frac[:, 0])
                for oy in (0, 1):
                    wy = frac[:, 1] if oy == 1 else (1.0 - frac[:, 1])
                    for oz in (0, 1):
                        wz = frac[:, 2] if oz == 1 else (1.0 - frac[:, 2])
                        weight = (wx * wy * wz).unsqueeze(-1)
                        corner = base + torch.tensor([ox, oy, oz], device=device, dtype=torch.long)
                        idx = self._hash(corner, self.hashmap_size)
                        feats = feats + weight * table(idx)

            level_feats.append(feats)

        out = torch.stack(level_feats, dim=1)
        return out.reshape(*orig_shape, self.num_levels, self.features_per_level)
