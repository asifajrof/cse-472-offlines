# softmax layer, inherits from layer class
from layer import Layer
import numpy as np


class Softmax(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        # np.exp overflow check
        input_limited = input - np.max(input, axis=1, keepdims=True)
        output = np.exp(input_limited)
        output = output / np.sum(output, axis=1, keepdims=True)
        return output

    def backward(self, output_error, learning_rate):
        input_error = output_error.copy()
        return input_error
