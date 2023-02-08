from layer import Layer
import numpy as np


class FullyConnected(Layer):
    def __init__(self, output_dim):
        # self.output_dim = output dimension
        self.output_dim = output_dim

        # initialize weights and biases
        # None, as we don't know the input shape yet
        self.weights = None
        self.bias = None

    def forward(self, input):
        # input: (batch_size, input_dim)
        batch_size, input_dim = input.shape

        # weights: (input_dim, output_dim)
        # initialize weights xaiver initialization
        if self.weights is None:
            self.weights = np.random.randn(input_dim, self.output_dim) * (
                np.sqrt(2 / (
                    input_dim + self.output_dim
                ))
            )

        # bias: (1, output_dim)
        # initialize biases to 0
        if self.bias is None:
            self.bias = np.zeros((1, self.output_dim))

        # output: (batch_size, output_dim)
        output = np.matmul(input, self.weights) + self.bias

        self.cache = input
        return output

    def backward(self, output_error, learning_rate):
        # output_error: (batch_size, output_dim)
        # learning_rate:
        # input: (batch_size, input_dim)
        input = self.cache
        batch_size, output_dim = output_error.shape

        # weights: (input_dim, output_dim)
        weights_error = np.matmul(input.T, output_error) * 1/batch_size
        self.weights -= learning_rate * weights_error

        # bias: (1, output_dim)
        bias_error = np.sum(output_error, axis=0) * 1/batch_size
        self.bias -= learning_rate * bias_error

        # input_error: (batch_size, input_dim)
        input_error = np.matmul(output_error, self.weights.T)

        # # clear cache
        # input = None
        # self.cache = None
        return input_error
