from layer import Layer
import numpy as np


class MaxPooling(Layer):
    def __init__(self, filter_dim, stride):
        # filter_dim: filter dimension. (filter_height, filter_width)

        self.filter_height, self.filter_width = filter_dim
        self.stride = stride
        pass

    def forward(self, input):
        batch_size, num_channels, height, width = input.shape

        output_height = int(
            (height - self.filter_height) / self.stride + 1)
        output_width = int(
            (width - self.filter_width) / self.stride + 1)

        # # for loop
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

        # special case when stride == filter_size
        max_value_mask = None
        # max_value_mask -> (batch_size, num_channels, input_height, input_width)
        if self.stride == self.filter_height:
            # initialize from output. repeating
            max_value_mask = output.repeat(
                self.stride,
                axis=2
            ).repeat(
                self.stride,
                axis=3
            )
            # pad for non-divisible input size
            max_pad_height = height - max_value_mask.shape[2]
            max_pad_width = width - max_value_mask.shape[3]
            if max_pad_height > 0 or max_pad_width > 0:
                max_value_mask = np.pad(
                    max_value_mask,
                    (
                        (0, 0),
                        (0, 0),
                        (0, max_pad_height),
                        (0, max_pad_width)
                    ),
                    'constant',
                )
            # compare with input
            # problem for multiple maxima :(
            max_value_mask = np.equal(max_value_mask, input)

        self.cache = input, max_value_mask

        return output

    def backward(self, output_error, learning_rate):
        # output_error. (batch_size, num_channels, out_height, out_width)
        input, max_value_mask = self.cache
        batch_size, num_channels, height, width = input.shape
        # special case when stride == filter_size
        if self.stride == self.filter_height:
            # check if max_value_mask is not None
            if max_value_mask is None:
                raise Exception(
                    'max_value_mask is None. Check if stride == filter_height')
            else:
                # output error needs to be tiled
                repeated_output_error = output_error.repeat(
                    self.stride,
                    axis=2
                ).repeat(
                    self.stride,
                    axis=3
                )
                # pad for non-divisible input size
                pad_height = height - repeated_output_error.shape[2]
                pad_width = width - repeated_output_error.shape[3]
                if pad_height > 0 or pad_width > 0:
                    repeated_output_error = np.pad(
                        repeated_output_error,
                        (
                            (0, 0),
                            (0, 0),
                            (0, pad_height),
                            (0, pad_width)
                        ),
                        'constant',
                    )
                # element-wise multiplication
                input_error = np.einsum(
                    'ijkl,ijkl->ijkl',
                    max_value_mask,
                    repeated_output_error
                )

        else:
            # for loop
            _, _, output_height, output_width = output_error.shape
            input_error = np.zeros(input.shape)

            for i in range(batch_size):
                for j in range(num_channels):
                    for h in range(output_height):
                        for w in range(output_width):
                            input_window = input[
                                i,
                                j,
                                h*self.stride:h*self.stride+self.filter_height,
                                w*self.stride:w*self.stride+self.filter_width
                            ]
                            # https://stackoverflow.com/a/9483964
                            max_index = np.unravel_index(
                                input_window.argmax(), input_window.shape)
                            max_index = (i, j, h*self.stride +
                                         max_index[0], w*self.stride+max_index[1])

                            # add overlapped gradients
                            # https://ai.stackexchange.com/a/17109
                            input_error[max_index] += output_error[i, j, h, w]

        # # clear cache
        # input = None
        # max_value_mask = None
        # self.cache = None
        return input_error
