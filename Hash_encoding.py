import math
import torch
import torch.nn as nn


class MultiResHashEncoder(nn.Module):
    """Multi-resolution hash-grid encoder with trilinear interpolation."""

    def __init__(
        self,
        num_levels=12,
        features_per_level=2,
        log2_hashmap_size=17,
        base_resolution=16,
        finest_resolution=256,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        if num_levels <= 1:
            self.b = 1.0
        else:
            self.b = math.exp(
                (math.log(finest_resolution) - math.log(base_resolution))
                / (num_levels - 1)
            )

        self.tables = nn.ModuleList(
            [nn.Embedding(self.hashmap_size, features_per_level) for _ in range(num_levels)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.tables:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)

    def hash(self, coords):
        """Fast spatial hash for integer 3D coordinates."""
        coords = coords.long()
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        hashed = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
        return torch.remainder(hashed, self.hashmap_size)

    def forward(self, x):
        """
        Args:
            x: [..., 3] canonical coordinates.

        Returns:
            Tensor of shape [..., num_levels, features_per_level]
        """
        if x.shape[-1] != 3:
            raise ValueError(f"Expected input with last dim = 3, got {x.shape[-1]}")

        feats_all = []
        out_dtype = self.tables[0].weight.dtype

        for lvl in range(self.num_levels):
            resolution = max(1, int(round(self.base_resolution * (self.b ** lvl))))

            x_scaled = x * resolution
            x0 = torch.floor(x_scaled).long()
            w = x_scaled - x0.float()

            feat = torch.zeros(
                *x.shape[:-1],
                self.features_per_level,
                device=x.device,
                dtype=out_dtype,
            )

            for ix in (0, 1):
                for iy in (0, 1):
                    for iz in (0, 1):
                        corner = torch.stack(
                            [
                                x0[..., 0] + ix,
                                x0[..., 1] + iy,
                                x0[..., 2] + iz,
                            ],
                            dim=-1,
                        )

                        idx = self.hash(corner)
                        f = self.tables[lvl](idx)

                        wx = w[..., 0] if ix else (1.0 - w[..., 0])
                        wy = w[..., 1] if iy else (1.0 - w[..., 1])
                        wz = w[..., 2] if iz else (1.0 - w[..., 2])
                        weight = (wx * wy * wz).unsqueeze(-1)

                        feat = feat + weight * f

            feats_all.append(feat)

        return torch.stack(feats_all, dim=-2)
