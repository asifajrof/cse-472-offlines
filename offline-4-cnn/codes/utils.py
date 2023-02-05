import numpy as np


def get_indices(X_shape, filter_h, filter_w, stride=1, padding=0):
    # X_shape: (batch_size, num_channels, height, width)
    # filter_h: int
    # filter_w: int
    # stride: int
    # padding: int
    # return:

    batch_size, num_channels, height, width = X_shape

    # output size
    out_h = int((height - filter_h + 2 * padding) / stride + 1)
    out_w = int((width - filter_w + 2 * padding) / stride + 1)

    # indices

    # h
    # first level
    h_indices = np.repeat(np.arange(filter_h), filter_w)
    # duplicate for input channels
    h_indices = np.tile(h_indices, num_channels)
    # stride
    init_points = stride * np.repeat(np.arange(out_h), out_w)

    h_indices = h_indices.reshape(-1, 1) + init_points.reshape(1, -1)

    # w
    # first level
    w_indices = np.tile(np.arange(filter_w), filter_h)
    # duplicate for input channels
    w_indices = np.tile(w_indices, num_channels)
    # stride
    init_points = stride * np.tile(np.arange(out_w), out_h)

    w_indices = w_indices.reshape(-1, 1) + init_points.reshape(1, -1)

    # d
    d_indices = np.repeat(np.arange(num_channels),
                          filter_h * filter_w).reshape(-1, 1)

    return h_indices, w_indices, d_indices


def im2col(X, filter_h, filter_w, stride=1, padding=0):
    # X: (batch_size, num_channels, height, width)
    # filter_h: int
    # filter_w: int
    # stride: int
    # padding: int
    # return:

    # padding
    X_padded = np.pad(X, ((0, 0),
                          (0, 0),
                          (padding, padding),
                          (padding, padding)
                          ), 'constant')
    h, w, d = get_indices(X.shape, filter_h, filter_w, stride, padding)

    X_cols = X_padded[:, d, h, w]
    X_cols = np.concatenate(X_cols, axis=-1)
    return X_cols


# def convolve(X, Y, stride=1, padding=0):
#     # X: (batch_size, num_channels, height, width)
#     # Y: (num_filters, num_channels, filter_height, filter_width)
#     # stride: int
#     # padding: int
#     # return: (batch_size, num_filters, height, width)
