# fully connected layer, inherits from layer class
from layer import Layer
import numpy as np


class FullyConnected(Layer):
    def __init__(self, input_size, output_size):
        # self.input_size = input_size
        # self.output_size = output_size
        # self.weights = np.random.randn(input_size, output_size) * 0.1
        # self.bias = np.zeros(output_size)
        pass

    def forward(self, input):
        # self.input = input
        # self.output = np.dot(input, self.weights) + self.bias
        # return self.output
        pass

    def backward(self, output_error, learning_rate):
        # input_error = np.dot(output_error, self.weights.T)
        # weights_error = np.dot(self.input.T, output_error)
        # bias_error = np.sum(output_error, axis=0)
        # self.weights -= learning_rate * weights_error
        # self.bias -= learning_rate * bias_error
        # return input_error
        pass
