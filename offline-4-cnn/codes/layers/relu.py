# relu layer, inherits from layer class
from layer import Layer
import numpy as np


class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        # self.input = input
        # self.output = np.maximum(0, input)
        # return self.output
        pass

    def backward(self, output_error, learning_rate):
        # input_error = output_error.copy()
        # input_error[self.input <= 0] = 0
        # return input_error
        pass
