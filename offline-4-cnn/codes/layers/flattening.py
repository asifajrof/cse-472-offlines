# flattening layer, inherits from layer class
from layer import Layer
import numpy as np


class Flattening(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        # self.input = input
        # self.output = input.reshape(input.shape[0], -1)
        # return self.output
        pass

    def backward(self, output_error, learning_rate):
        # input_error = output_error.reshape(self.input.shape)
        # return input_error
        pass
