from layer import Layer
# import numpy as np


class Flattening(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        # input: convolutional filter map
        # output: flattened input. as row vectors.
        self.input_shape = input.shape
        output = input.reshape(input.shape[0], -1)
        return output

    def backward(self, output_error, learning_rate):
        # output_error: error of the next layer
        # learning_rate: learning rate
        input_error = output_error.reshape(self.input_shape)
        return input_error
