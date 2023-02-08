from layer import Layer
# from utils import *
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
                    self.filter_height
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

        # np.pad: Number of values padded to the edges of each axis.
        # ((before_1, after_1), ... (before_N, after_N)) unique pad widths for each axis.
        # pad with 'constant' values. default: 0
        input_padded = np.pad(input, ((0, 0),
                                      (0, 0),
                                      (self.padding, self.padding),
                                      (self.padding, self.padding)
                                      ), 'constant')

        # # for loop
        #
        # output = np.zeros((batch_size, self.num_filters,
        #                   output_height, output_width))
        #
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

        # as strided
        # https://stackoverflow.com/a/53099870
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
        # https://ajcr.net/Basic-guide-to-einsum/
        output = np.einsum(
            'bcijkl,fckl->bfij',
            input_strided, self.weights
        )
        output += self.biases.reshape(1, -1, 1, 1)

        # self.cache = (input, input_padded, input_strided)
        self.cache = input_padded
        # self.cache = input
        return output

    def backward(self, output_error, learning_rate):
        # output_error: (batch_size, num_filters, output_height, output_width)
        # learning_rate: learning rate
        # input, input_padded, input_strided = self.cache
        input_padded = self.cache
        # input = self.cache

        # dilate the output error
        dilate = self.stride - 1
        # insert dilate number of 0 rows/cols between each row/col
        output_error_modified = np.insert(
            output_error,
            obj=np.arange(1, output_error.shape[2]).repeat(dilate),
            values=0,
            axis=2
        )
        output_error_modified = np.insert(
            output_error_modified,
            obj=np.arange(1, output_error_modified.shape[3]).repeat(dilate),
            values=0,
            axis=3
        )

        weights_error_height = input_padded.shape[2] - \
            output_error_modified.shape[2] + 1
        weights_error_width = input_padded.shape[3] - \
            output_error_modified.shape[3] + 1

        # check with original weights
        # dilate the output error at the end
        output_error_modified_weights = output_error_modified.copy()
        if weights_error_height > self.weights.shape[2]:
            output_error_modified_weights = np.insert(
                output_error_modified,
                obj=np.array([output_error_modified.shape[2]]).repeat(
                    weights_error_height - self.weights.shape[2]
                ),
                values=0,
                axis=2
            )
        if weights_error_width > self.weights.shape[3]:
            output_error_modified_weights = np.insert(
                output_error_modified_weights,
                obj=np.array([output_error_modified_weights.shape[3]]).repeat(
                    weights_error_width - self.weights.shape[3]
                ),
                values=0,
                axis=3
            )

        # update weights
        # as strided
        input_strided = np.lib.stride_tricks.as_strided(
            input_padded,
            shape=(
                input_padded.shape[0],
                input_padded.shape[1],
                self.weights.shape[2],
                self.weights.shape[3],
                output_error_modified_weights.shape[2],
                output_error_modified_weights.shape[3]
            ),
            strides=(
                input_padded.strides[0],
                input_padded.strides[1],
                input_padded.strides[2],
                input_padded.strides[3],
                input_padded.strides[2],
                input_padded.strides[3]
            )
        )
        weights_error = np.einsum(
            'bcklij,bfij->fckl',
            input_strided, output_error_modified_weights
        )
        weights_error = weights_error * 1/input_padded.shape[0]
        self.weights -= learning_rate * weights_error

        # update biases
        biases_error = np.sum(output_error_modified,
                              axis=(0, 2, 3)) * 1/input_padded.shape[0]
        self.biases -= learning_rate * biases_error

        # input error calculation
        padded_height = output_error_modified.shape[2] + \
            2 * (self.filter_height - 1)
        padded_width = output_error_modified.shape[3] + \
            2 * (self.filter_width - 1)

        input_error_height = padded_height - self.filter_height + 1
        input_error_width = padded_width - self.filter_width + 1

        # check with the original input
        # dilate the output error at the end
        if input_error_height < input_padded.shape[2]:
            output_error_modified = np.insert(
                output_error_modified,
                obj=np.array([output_error_modified.shape[2]]).repeat(
                    input_padded.shape[2] - input_error_height),
                values=0,
                axis=2
            )
        if input_error_width < input_padded.shape[3]:
            output_error_modified = np.insert(
                output_error_modified,
                obj=np.array([output_error_modified.shape[3]]).repeat(
                    input_padded.shape[3] - input_error_width),
                values=0,
                axis=3
            )

        # pad the output error with filter_size - 1
        output_error_modified = np.pad(
            output_error_modified,
            (
                (0, 0),
                (0, 0),
                (self.filter_height - 1, self.filter_height - 1),
                (self.filter_width - 1, self.filter_width - 1)
            ),
            'constant'
        )
        # rotate the weights by 180
        weights_modified = np.rot90(self.weights, 2, (2, 3))

        # assert input_padded.shape[2] == output_error_modified.shape[2] - \
        #     self.filter_height + 1
        # assert input_padded.shape[3] == output_error_modified.shape[3] - \
        #     self.filter_width + 1
        # as strided
        output_error_modified_strided = np.lib.stride_tricks.as_strided(
            output_error_modified,
            shape=(
                output_error_modified.shape[0],
                output_error_modified.shape[1],
                input_padded.shape[2],
                input_padded.shape[3],
                self.filter_height,
                self.filter_width
            ),
            strides=(
                output_error_modified.strides[0],
                output_error_modified.strides[1],
                output_error_modified.strides[2],
                output_error_modified.strides[3],
                output_error_modified.strides[2],
                output_error_modified.strides[3]
            )
        )
        # einsum
        input_error = np.einsum(
            'bfijkl,fckl->bcij',
            output_error_modified_strided, weights_modified
        )

        # drop the padded rows/cols
        input_error = input_error[:, :, self.padding:-
                                  self.padding, self.padding:-self.padding]

        # clear cache
        input_padded = None
        self.cache = None

        return input_error
