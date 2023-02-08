# relu layer, inherits from layer class
from layer import Layer
import numpy as np


class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        output = np.maximum(0, input)
        self.cache = input
        return output

    def backward(self, output_error, learning_rate):
        input = self.cache
        input_error = output_error.copy()
        input_error[input <= 0] = 0

        # # clear cache
        # input = None
        # self.cache = None
        return input_error
