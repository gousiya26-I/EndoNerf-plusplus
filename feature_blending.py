
import torch
import torch.nn as nn

class FeatureBlender(nn.Module):
    def __init__(self, num_levels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_levels))

    def forward(self, features):
        w = torch.softmax(self.weights, dim=0)

        out = 0
        for i in range(len(features)):
            out = out + w[i] * features[i]

        return out
