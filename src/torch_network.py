import torch
import torch.nn as nn


class Network(nn.Module):
    """Thid is a network class"""

    def __init__(self, architecture: nn.Sequential):
        super().__init__()
        self.flatten = nn.Flatten()
        self.architecture = architecture

    def forward(self, data: torch.Tensor):
        flattened_data = self.flatten(data)
        return self.architecture(flattened_data)


# ------------------------------ Testing what the code does --------------------------- #
# if __name__ == "__main__":
