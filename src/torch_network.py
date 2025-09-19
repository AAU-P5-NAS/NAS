import torch
import torch.nn as nn


class Network(nn.Module):
    """Thid is a network class"""

    def __init__(self, sizes: list[int]):
        super().__init__()
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Create biases as nn.Parameter tensors
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(y, 1)) for y in sizes[1:]])

        # Create weights as nn.Parameter tensors
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        )

    def forward(self, x):
        # x should be of shape (input_dim, batch_size)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = torch.relu(torch.matmul(w, x) + b)
        x = torch.matmul(self.weights[-1], x) + self.biases[-1]
        return x


# ------------------------------ Testing what the code does --------------------------- #
if __name__ == "__main__":
    size = [3, 5, 2]
    net = Network(size)
    # Print weights and biases more clearly
    for i, (w, b) in enumerate(zip(net.weights, net.biases), 1):
        print(f"Layer {i} weights:\n{w.data}")
        print(f"Layer {i} biases:\n{b.data}\n")
