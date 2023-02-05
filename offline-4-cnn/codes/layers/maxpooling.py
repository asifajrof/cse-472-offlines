# max pooling layer, inherits from layer class
from layer import Layer
import numpy as np


class MaxPooling(Layer):
    def __init__(self, filter_size, stride):
        # self.filter_size = filter_size
        # self.stride = stride
        pass

    def forward(self, input):
        # self.input = input
        # num_channels, input_height, input_width = input.shape
        # output_height = int(
        #     (input_height - self.filter_size) / self.stride + 1)
        # output_width = int((input_width - self.filter_size) / self.stride + 1)
        # output = np.zeros((num_channels, output_height, output_width))
        # for channel in range(num_channels):
        #     for h in range(output_height):
        #         for w in range(output_width):
        #             output[channel, h, w] = np.max(
        #                 input[channel, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size])
        # self.output = output
        # return output
        pass

    def backward(self, output_error, learning_rate):
        # input_error = np.zeros(self.input.shape)
        # num_channels, input_height, input_width = self.input.shape
        # output_height, output_width = output_error.shape[1:]
        # for channel in range(num_channels):
        #     for h in range(output_height):
        #         for w in range(output_width):
        #             input_error[channel, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size] += np.where(self.input[channel, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size] == np.max(
        #                 self.input[channel, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size]), output_error[channel, h, w], 0)
        # return input_error
        pass
