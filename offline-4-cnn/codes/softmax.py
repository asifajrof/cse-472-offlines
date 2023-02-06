# softmax layer, inherits from layer class
from layer import Layer
import numpy as np


class Softmax(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        # self.input = input
        # self.output = np.exp(input) / np.sum(np.exp(input),
        #                                      axis=1, keepdims=True)
        # return self.output
        pass

    def backward(self, output_error, learning_rate):
        # input_error = output_error.copy()
        # return input_error
        pass
