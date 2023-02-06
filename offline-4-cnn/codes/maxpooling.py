# max pooling layer, inherits from layer class
from layer import Layer
import numpy as np


class MaxPooling(Layer):
    def __init__(self, filter_dim, stride):
        # filter_dim: filter dimension. (filter_height, filter_width)

        self.filter_height, self.filter_width = filter_dim
        self.stride = stride
        pass

    def forward(self, input):
        self.input = input
        batch_size, num_channels, height, width = input.shape

        output_height = int(
            (height - self.filter_height) / self.stride + 1)
        output_width = int(
            (width - self.filter_width) / self.stride + 1)
        # output = np.zeros(
        #     (batch_size, num_channels, output_height, output_width))
        # for h in range(output_height):
        #     for w in range(output_width):
        #         strided_window = input[
        #             :,
        #             :,
        #             h * self.stride:h * self.stride + self.filter_height,
        #             w * self.stride:w * self.stride + self.filter_width
        #         ]
        #         output[:, :, h, w] = np.max(strided_window, axis=(2, 3))

        # as strided
        input_strided = np.lib.stride_tricks.as_strided(
            input,
            shape=(
                batch_size,
                num_channels,
                output_height,
                output_width,
                self.filter_height,
                self.filter_width
            ),
            strides=(
                input.strides[0],
                input.strides[1],
                input.strides[2] * self.stride,
                input.strides[3] * self.stride,
                input.strides[2],
                input.strides[3]
            )
        )
        output = np.max(input_strided, axis=(4, 5))
        return output

    def backward(self, output_error, learning_rate):
        # output_error. (batch_size, num_channels, out_height, out_width)
        input_error = np.zeros(self.input.shape)
        batch_size, num_channels, height, width = self.input.shape
        # output_height, output_width = output_error.shape[1:]
        # for channel in range(num_channels):
        #     for h in range(output_height):
        #         for w in range(output_width):
        #             input_error[channel, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size] += np.where(self.input[channel, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size] == np.max(
        #                 self.input[channel, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size]), output_error[channel, h, w], 0)
        # return input_error
        pass
