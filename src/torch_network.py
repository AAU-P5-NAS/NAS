import torch
import torch.nn as nn


class Network(nn.Module):
    """Thid is a network class"""

    def __init__(self, sizes):
        super(Network, self).__init__()
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Create biases as nn.Parameter tensors
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(y, 1)) for y in sizes[1:]])

        # Create weights as nn.Parameter tensors
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        )


# ------------------------------ Testing what the code does --------------------------- #
if __name__ == "__main__":
    size = [3, 5, 2]
    net = Network(size)
    print("Baises = ", net.biases)
    print("Wheights = ", net.weights)
