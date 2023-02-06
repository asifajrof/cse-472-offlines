# convolution layer, inherits from layer class
from layer import Layer
from utils import *
import numpy as np


class Convolution(Layer):
    def __init__(self, num_filters, filter_dim, stride=1, padding=0):
        # num_filters: number of output channels
        # filter_dim: filter dimension (height, width)
        # stride: stride of the convolution. default: 1
        # padding: padding of the input (assuming zero padding and square padding). default: 0

        self.num_filters = num_filters
        self.filter_height, self.filter_width = filter_dim
        self.stride = stride
        self.padding = padding

        # initialize weights and biases
        # None, as we don't know the input shape yet
        self.weights = None
        self.biases = None

    def forward(self, input):
        # input: (batch_size, num_channels, input_height, input_width)
        batch_size, num_channels, input_height, input_width = input.shape

        # weights: (num_filters, num_channels, filter_height, filter_width)
        # initialize weights xaiver initialization
        if self.weights is None:
            self.weights = np.random.randn(self.num_filters, num_channels, self.filter_height, self.filter_width) * (
                np.sqrt(2 / (
                    self.filter_height * self.filter_width
                ))
            )

        # biases: (num_filters, 1)
        # initialize biases to 0
        if self.biases is None:
            self.biases = np.zeros(self.num_filters)

        # output: (batch_size, num_filters, output_height, output_width)
        output_height = int((input_height - self.filter_height +
                            2 * self.padding) / self.stride + 1)
        output_width = int((input_width - self.filter_width + 2 *
                           self.padding) / self.stride + 1)

        # output = np.zeros((batch_size, self.num_filters,
        #                   output_height, output_width))

        # np.pad: Number of values padded to the edges of each axis.
        # ((before_1, after_1), ... (before_N, after_N)) unique pad widths for each axis.
        # pad with 'constant' values. default: 0
        input_padded = np.pad(input, ((0, 0),
                                      (0, 0),
                                      (self.padding, self.padding),
                                      (self.padding, self.padding)
                                      ), 'constant')

        # for sample_index in range(batch_size):
        #     for filter_index in range(self.num_filters):
        #         for h in range(output_height):
        #             for w in range(output_width):
        #                 output[sample_index, filter_index, h, w] = np.sum(
        #                     input_padded[sample_index,
        #                                  :,
        #                                  h * self.stride:h * self.stride + self.filter_height,
        #                                  w * self.stride:w * self.stride + self.filter_width
        #                                  ] * self.weights[filter_index]
        #                 ) + self.biases[filter_index]

        # vectorized implementation
        # input_col = im2col(input, self.filter_height,
        #                    self.filter_width, self.stride, self.padding)
        # weights_col = self.weights.reshape(self.num_filters, -1)
        # bias_col = self.biases.reshape(-1, 1)

        # output_col = np.matmul(weights_col, input_col) + bias_col

        # output = np.array(np.hsplit(output_col, batch_size)).reshape(
        #     (batch_size, self.num_filters, output_height, output_width))

        # as strided
        input_strided = np.lib.stride_tricks.as_strided(
            input_padded,
            shape=(
                batch_size,
                num_channels,
                output_height,
                output_width,
                self.filter_height,
                self.filter_width
            ),
            strides=(
                input_padded.strides[0],
                input_padded.strides[1],
                input_padded.strides[2] * self.stride,
                input_padded.strides[3] * self.stride,
                input_padded.strides[2],
                input_padded.strides[3]
            )
        )
        # einsum
        output = np.einsum(
            'bcijkl,fckl->bfij',
            input_strided, self.weights
        )
        return output

    def backward(self, output_error, learning_rate):
        # num_filters, num_channels, filter_size, filter_size = self.filters.shape
        # input_height, input_width = self.input.shape[2:]
        # output_height, output_width = output_error.shape[1:]
        # input_error = np.zeros(self.input.shape)
        # for filter_num in range(num_filters):
        #     for h in range(output_height):
        #         for w in range(output_width):
        #             input_error[:, :, h * self.stride:h * self.stride + filter_size, w * self.stride:w *
        #                         self.stride + filter_size] += self.filters[filter_num] * output_error[filter_num, h, w]
        #             self.filters[filter_num] += learning_rate * self.input[:, :, h * self.stride:h * self.stride +
        #                                                                    filter_size, w * self.stride:w * self.stride + filter_size] * output_error[filter_num, h, w]
        pass
