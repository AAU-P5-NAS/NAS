"""This code is from the book: Neural Networks and Deep Learning By Michael Nielsen and its corrosponding repository:
(https://github.com/unexploredtest/neural-networks-and-deep-learning)"""

import numpy as np

# Terminology
# Sigmoid Neuron:  basic element of a neural network, it takes a set of inputs and gives an output
# the input is a value x that multiplued by a wheight w added to a bais b    z = xw + b
# The output of a neuron is also called an activation
# The activation of a neuron is calculated using an activation function such as sigmoid function.
# The activation function is aplyied  to z hence: output = sigmoid(z)


class Network(object):
    """This class is to construct a neural network"""

    def __init__(self, sizes):
        """The init method takes a list sizes e.g sizes = [20, 10, 4],
        each element represent the number of neurons in a layer starting with the input layer."""

        self.sizes = sizes
        self.num_layers = len(sizes)

        # A list of arrays, where each array contains the biases for one layer of the network (excluding the input layer).
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # A list of weight matrices.
        # For each pair of layers, the matrix has shape (next_layer_size, current_layer_size).
        # Entry W[n, m] is the weight from neuron m in the current layer to neuron n in the next layer.
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Returns the output if the network if 'a' is the input"""

        # Note:
        # w: matrix of size (next_layer_size, current_layer_size)
        # a: column vector of shape (current_layer_size, 1)
        # b: column vector of shape (next_layer_size, 1)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self):
        """Train the neural network using mini-batch stochastic gradient descent."""

    def updage_mini_batch(self):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch."""

    def backprop(self):
        """The implementation of the backpropogation algo"""

    def evaluate(self):
        """Return the number of test inputs for which the neural
        network outputs the correct result."""

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives for the output activation"""
        return output_activations - y


# Helper Functions
def sigmoid(z):
    """The sigmoid function"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """The derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))


# ------------------------------ Testing what the code does --------------------------- #
if __name__ == "__main__":
    size = [3, 5, 2]
    net = Network(size)
    print("Baises = ", net.biases)
    print("Wheights = ", net.weights)
