# ===================================================================================================================
# all imports
# ===================================================================================================================
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

# ===================================================================================================================
# initialze parameters
# ===================================================================================================================
random_seed = 0
np.random.seed(random_seed)
test_ratio = 0.2
# n_samples = 45000
n_samples = 100000
# n_epochs = 10
n_epochs = 5
# batch_size = 64
batch_size = 8
learning_rate = 0.001

RUN_TYPE = "test"    # "new" or "load" or "test"
model_filename = "model_45000-samples_0.2-test_ratio_42-random_seed_10-epochs_64-batch_size.pkl"

data_root = "../Assignment4-Materials/NumtaDB_with_aug/"
csv_filenames = [
    "training-a.csv",
    "training-b.csv",
    "training-c.csv",
    # "training-d.csv",
    # "training-e.csv"
]

csv_filenames_test = [
    # "training-a.csv",
    # "training-b.csv",
    # "training-c.csv",
    "training-d.csv",
    # "training-e.csv"
]

save_root = "../saved/"

image_output_dim = (28, 28)
lenet_model_params = [
    {
        "type": "convolution",
        "num_filters": 6,
        "filter_dim": (5, 5),
        "stride": 1,
        "padding": 0
    }, {
        "type": "relu"
    }, {
        "type": "maxpooling",
        "filter_dim": (2, 2),
        "stride": 2
    }, {
        "type": "convolution",
        "num_filters": 16,
        "filter_dim": (5, 5),
        "stride": 1,
        "padding": 0
    }, {
        "type": "relu"
    }, {
        "type": "maxpooling",
        "filter_dim": (2, 2),
        "stride": 2
    }, {
        "type": "flattening"
    }, {
        "type": "fullyconnected",
        "output_dim": 120
    }, {
        "type": "relu"
    }, {
        "type": "fullyconnected",
        "output_dim": 84
    }, {
        "type": "relu"
    }, {
        "type": "fullyconnected",
        "output_dim": 10
    }, {
        "type": "softmax"
    }
]

labels = np.arange(10)

# ===================================================================================================================
# utils.py
# ===================================================================================================================
# import pandas as pd
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import cv2
# from tqdm import tqdm
# import matplotlib.pyplot as plt


def read_all_csv(csv_root, csv_filenames):
    # merge all csv files into one
    merged_df = pd.DataFrame()
    for filename in csv_filenames:
        csv_path = Path(csv_root) / Path(filename)
        df = pd.read_csv(csv_path)
        # print(df.shape)
        # merge without append
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    return merged_df


def get_dataset(df, n_samples=5000, random_state=42):
    if n_samples > len(df):
        n_samples = len(df)
    # get labels -> digit
    # get filepaths -> "database name" + "filename"
    dataset_df = df[["digit", "database name", "filename"]]
    # get random n samples
    dataset_df = dataset_df.sample(n=n_samples, random_state=random_state)
    return dataset_df


def get_test_train_split(csv_root, csv_filenames, n_samples=5000, test_size=0.2, random_state=42):
    merged_df = read_all_csv(csv_root, csv_filenames)
    dataset_df = get_dataset(
        merged_df, n_samples=n_samples, random_state=random_state)

    X = dataset_df["database name"] + "/" + dataset_df["filename"]
    y = dataset_df["digit"]

    # stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test


def get_test_set(csv_root, csv_filenames, n_samples=5000, random_state=42):
    merged_df = read_all_csv(csv_root, csv_filenames)
    dataset_df = get_dataset(
        merged_df, n_samples=n_samples, random_state=random_state)
    X = dataset_df["database name"] + "/" + dataset_df["filename"]
    y = dataset_df["digit"]
    return X, y


def load_image(image_path_root, image_paths, output_dim=(28, 28)):
    images = []
    for image_path in tqdm(image_paths):
        try:
            # read image
            str_image_path = str(Path(image_path_root)/Path(image_path))
            # print(f'image path: {str_image_path}')
            image = cv2.imread(str_image_path)
            # resize
            # print(f'image shape: {image.shape}, output_dim: {output_dim}')
            image = cv2.resize(image, output_dim)
            # convert to rgb
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # black ink on white page. invert
            image = 255 - image
            # normalize
            image = image / 255
            # channel, height, width
            image = image.transpose(2, 0, 1)
            images.append(image)
        except Exception as e:
            print(e)
            print("Error loading image: ", image_path)
    return np.array(images)


def view_image_info(X, y, images, index):
    # print(f'file: {X.iloc[index]}')
    # print(f'label: {y.iloc[index]}')
    # channel, height, width
    image = images[index].transpose(1, 2, 0)
    plt.title(f'file: {X.iloc[index]}, label: {y.iloc[index]}')
    plt.imshow(image)
    plt.show()
    plt.close()


def one_hot_encoding(y):
    # y: (n_samples, )
    # y_one_hot: (n_samples, n_classes)
    n_samples = y.shape[0]
    n_classes = np.unique(y).shape[0]
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[range(n_samples), y] = 1
    return y_one_hot


def cross_entropy_loss(y_pred, y_true):
    # y_pred: (n_samples, n_classes)
    # y_true: (n_samples, n_classes)
    # loss: scalar
    EPS = 1e-8
    loss = -np.sum(y_true * np.log(y_pred + EPS)) / y_pred.shape[0]
    return loss


def accuracy(y_true, y_pred):
    # y_true: (n_samples,n_classes)
    # y_pred: (n_samples,n_classes)
    # acc: scalar
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    acc = np.sum(y_pred == y_true) / y_pred.shape[0]
    sklearn_acc = accuracy_score(y_true, y_pred)
    try:
        assert acc == sklearn_acc
    except AssertionError:
        print(f'AssertionError: acc: {acc}, sklearn_acc: {sklearn_acc}')
    return sklearn_acc


def macro_f1_score(y_true, y_pred):
    # y_true: (n_samples,n_classes)
    # y_pred: (n_samples,n_classes)
    # f1: scalar
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    return f1


def get_confusion_matrix(y_true, y_pred, labels):
    # y_true: (n_samples,n_classes)
    # y_pred: (n_samples,n_classes)
    # confusion_matrix: (n_classes,n_classes)
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    return cm

# ===================================================================================================================
# ===================================================================================================================

# ===================================================================================================================
# layer.py
# ===================================================================================================================


class Layer:
    def forward(self, input):
        # input.
        # returns output
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        # output_error -> part_L / part_output
        # learning_rate -> learning rate
        # calculates weights_error and bias_error
        # returns input_error
        raise NotImplementedError

# ===================================================================================================================
# ===================================================================================================================

# ===================================================================================================================
# convolution.py
# ===================================================================================================================
# from layer import Layer
# import numpy as np


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

        self.cache = input_padded
        return output

    def backward(self, output_error, learning_rate):
        # output_error: (batch_size, num_filters, output_height, output_width)
        # learning_rate: learning rate
        input_padded = self.cache

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

# ===================================================================================================================
# ===================================================================================================================

# ===================================================================================================================
# maxpooling.py
# ===================================================================================================================
# from layer import Layer
# import numpy as np


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

        # clear cache
        input = None
        max_value_mask = None
        self.cache = None
        return input_error

# ===================================================================================================================
# ===================================================================================================================

# ===================================================================================================================
# relu.py
# ===================================================================================================================
# from layer import Layer
# import numpy as np


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

        # clear cache
        input = None
        self.cache = None
        return input_error

# ===================================================================================================================
# ===================================================================================================================

# ===================================================================================================================
# flattening.py
# ===================================================================================================================
# from layer import Layer
# import numpy as np


class Flattening(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        # input: convolutional filter map
        # output: flattened input. as row vectors.
        input_shape = input.shape
        output = input.reshape(input.shape[0], -1)
        self.cache = input_shape
        return output

    def backward(self, output_error, learning_rate):
        # output_error: error of the next layer
        # learning_rate: learning rate
        input_shape = self.cache
        input_error = output_error.reshape(input_shape)

        # clear cache
        input_shape = None
        self.cache = None
        return input_error

# ===================================================================================================================
# ===================================================================================================================

# ===================================================================================================================
# fullyconnected.py
# ===================================================================================================================
# from layer import Layer
# import numpy as np


class FullyConnected(Layer):
    def __init__(self, output_dim):
        # self.output_dim = output dimension
        self.output_dim = output_dim

        # initialize weights and biases
        # None, as we don't know the input shape yet
        self.weights = None
        self.bias = None

    def forward(self, input):
        # input: (batch_size, input_dim)
        batch_size, input_dim = input.shape

        # weights: (input_dim, output_dim)
        # initialize weights xaiver initialization
        if self.weights is None:
            self.weights = np.random.randn(input_dim, self.output_dim) * (
                np.sqrt(2 / (
                    input_dim + self.output_dim
                ))
            )

        # bias: (1, output_dim)
        # initialize biases to 0
        if self.bias is None:
            self.bias = np.zeros((1, self.output_dim))

        # output: (batch_size, output_dim)
        output = np.matmul(input, self.weights) + self.bias

        self.cache = input
        return output

    def backward(self, output_error, learning_rate):
        # output_error: (batch_size, output_dim)
        # learning_rate:
        # input: (batch_size, input_dim)
        input = self.cache
        batch_size, output_dim = output_error.shape

        # weights: (input_dim, output_dim)
        weights_error = np.matmul(input.T, output_error) * 1/batch_size
        self.weights -= learning_rate * weights_error

        # bias: (1, output_dim)
        bias_error = np.sum(output_error, axis=0) * 1/batch_size
        self.bias -= learning_rate * bias_error

        # input_error: (batch_size, input_dim)
        input_error = np.matmul(output_error, self.weights.T)

        # clear cache
        input = None
        self.cache = None
        return input_error

# ===================================================================================================================
# ===================================================================================================================

# ===================================================================================================================
# softmax.py
# ===================================================================================================================
# from layer import Layer
# import numpy as np


class Softmax(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        EPS = 1e-8
        # np.exp overflow check
        input_limited = input - np.max(input, axis=1, keepdims=True)
        output = np.exp(input_limited)
        output = (output + EPS) / (np.sum(output, axis=1, keepdims=True) + EPS)
        return output

    def backward(self, output_error, learning_rate):
        input_error = output_error.copy()
        return input_error

# ===================================================================================================================
# ===================================================================================================================

# ===================================================================================================================
# basic_components.py
# ===================================================================================================================
# from layer import Layer
# from convolution import Convolution
# from relu import ReLU
# from maxpooling import MaxPooling
# from flattening import Flattening
# from fullyconnected import FullyConnected
# from softmax import Softmax

# ===================================================================================================================
# ===================================================================================================================

# ===================================================================================================================
# model.py
# ===================================================================================================================
# from basic_components import *
# from utils import *
# from tqdm import tqdm
# import numpy as np
# import pickle


# random_seed = 0
# np.random.seed(random_seed)
# test_ratio = 0.2
# # n_samples = 45000
# n_samples = 100000
# # n_epochs = 10
# n_epochs = 5
# # batch_size = 64
# batch_size = 8
# learning_rate = 0.001

# RUN_TYPE = "test"    # "new" or "load" or "test"
# model_filename = "model_45000-samples_0.2-test_ratio_42-random_seed_10-epochs_64-batch_size.pkl"

# data_root = "../Assignment4-Materials/NumtaDB_with_aug/"
# csv_filenames = [
#     "training-a.csv",
#     "training-b.csv",
#     "training-c.csv",
#     # "training-d.csv",
#     # "training-e.csv"
# ]

# csv_filenames_test = [
#     # "training-a.csv",
#     # "training-b.csv",
#     # "training-c.csv",
#     "training-d.csv",
#     # "training-e.csv"
# ]

# save_root = "../saved/"

# image_output_dim = (28, 28)
# lenet_model_params = [
#     {
#         "type": "convolution",
#         "num_filters": 6,
#         "filter_dim": (5, 5),
#         "stride": 1,
#         "padding": 0
#     }, {
#         "type": "relu"
#     }, {
#         "type": "maxpooling",
#         "filter_dim": (2, 2),
#         "stride": 2
#     }, {
#         "type": "convolution",
#         "num_filters": 16,
#         "filter_dim": (5, 5),
#         "stride": 1,
#         "padding": 0
#     }, {
#         "type": "relu"
#     }, {
#         "type": "maxpooling",
#         "filter_dim": (2, 2),
#         "stride": 2
#     }, {
#         "type": "flattening"
#     }, {
#         "type": "fullyconnected",
#         "output_dim": 120
#     }, {
#         "type": "relu"
#     }, {
#         "type": "fullyconnected",
#         "output_dim": 84
#     }, {
#         "type": "relu"
#     }, {
#         "type": "fullyconnected",
#         "output_dim": 10
#     }, {
#         "type": "softmax"
#     }
# ]

# labels = np.arange(10)


def create_model(model_params):
    model = []
    for layer in model_params:
        if layer["type"] == "convolution":
            model.append(Convolution(
                num_filters=layer["num_filters"],
                filter_dim=layer["filter_dim"],
                stride=layer["stride"],
                padding=layer["padding"]
            )
            )
        elif layer["type"] == "relu":
            model.append(ReLU())
        elif layer["type"] == "maxpooling":
            model.append(MaxPooling(
                filter_dim=layer["filter_dim"],
                stride=layer["stride"]
            )
            )
        elif layer["type"] == "flattening":
            model.append(Flattening())
        elif layer["type"] == "fullyconnected":
            model.append(FullyConnected(
                output_dim=layer["output_dim"]
            )
            )
        elif layer["type"] == "softmax":
            model.append(Softmax())
        else:
            raise Exception("Unknown layer type")

    return model


def save_model(model, save_root, save_filename):
    # pathlib Path make dir if not exist
    Path(save_root).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_root) / Path(save_filename)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {save_path}")


def load_model(load_root, load_filename):
    load_path = Path(load_root) / Path(load_filename)
    with open(load_path, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded from {load_path}")
    return model


def train_model(model, X, y, batch_size=32, learning_rate=0.001):
    # X: (n_samples, n_channels, height, width)
    # y: (n_samples, n_classes)

    # batching and training
    num_batches = X.shape[0] // batch_size
    batch_loss = []
    batch_acc = []
    for batch in tqdm(range(num_batches)):
        # get the batch
        batch_start = batch * batch_size
        batch_end = (batch + 1) * batch_size
        if batch_end > X.shape[0]:
            batch_end = X.shape[0]
        X_batch = X[batch_start:batch_end]
        y_batch = y[batch_start:batch_end]

        # forward
        forward_output = X_batch
        for layer in model:
            forward_output = layer.forward(input=forward_output)

        # calculate loss
        loss = cross_entropy_loss(
            y_pred=forward_output, y_true=y[batch_start:batch_end])
        batch_loss.append(loss)
        # calculate accuracy
        acc = accuracy(y_true=y_batch, y_pred=forward_output)
        batch_acc.append(acc)

        # backward
        grad = forward_output - y_batch
        # reverse the model
        model.reverse()
        for layer in model:
            grad = layer.backward(
                output_error=grad, learning_rate=learning_rate)
        # reverse the model back
        model.reverse()

    avg_loss = np.mean(batch_loss)
    avg_acc = np.mean(batch_acc)
    return avg_loss, avg_acc


def predict_model(model, X, y):
    # X: (n_samples, n_channels, height, width)
    # y: (n_samples, n_classes)
    print(f'Predicting {X.shape[0]} samples...')
    # forward
    forward_output = X
    for layer in tqdm(model):
        forward_output = layer.forward(input=forward_output)

    # calculate loss
    loss = cross_entropy_loss(y_pred=forward_output, y_true=y)
    # calculate accuracy
    acc = accuracy(y_true=y, y_pred=forward_output)
    # calculate f1 score
    f1 = macro_f1_score(y_true=y, y_pred=forward_output)

    return forward_output, loss, acc, f1


if __name__ == "__main__":
    if RUN_TYPE == "new":
        the_cnn = create_model(model_params=lenet_model_params)
    elif RUN_TYPE == "test":
        the_cnn = load_model(
            load_root=save_root, load_filename=model_filename)
    elif RUN_TYPE == "load":
        the_cnn = load_model(
            load_root=save_root, load_filename=model_filename)

    if RUN_TYPE == "new" or RUN_TYPE == "load":
        # dataset read
        X_train, X_test, y_train, y_test = get_test_train_split(
            csv_root=data_root,
            csv_filenames=csv_filenames,
            n_samples=n_samples, test_size=test_ratio, random_state=random_seed
        )
    elif RUN_TYPE == "test":
        # dataset read
        X_test, y_test = get_test_set(
            csv_root=data_root,
            csv_filenames=csv_filenames_test,
            n_samples=n_samples, random_state=random_seed
        )

    if RUN_TYPE == "new":
        # image read
        X_img_train = load_image(image_path_root=data_root,
                                 image_paths=X_train, output_dim=image_output_dim)
        # one hot encoding
        y_train_one_hot = one_hot_encoding(y_train)
        # image read
        X_img_test = load_image(image_path_root=data_root,
                                image_paths=X_test, output_dim=image_output_dim)
        # one hot encoding
        y_test_one_hot = one_hot_encoding(y_test)

        # epoch
        train_loss = []
        train_accuracy = []
        val_loss = []
        val_accuracy = []
        val_f1 = []
        final_pred = None
        for i in range(n_epochs):
            print(f'epoch: {i}')
            loss, acc = train_model(
                model=the_cnn, X=X_img_train, y=y_train_one_hot,
                batch_size=batch_size, learning_rate=learning_rate)
            train_loss.append(loss)
            train_accuracy.append(acc)
            print(f'training loss: {loss}')
            print(f'training accuracy: {acc}')

            # predict
            y_pred, loss, acc, f1 = predict_model(
                model=the_cnn, X=X_img_test, y=y_test_one_hot)
            val_loss.append(loss)
            val_accuracy.append(acc)
            val_f1.append(f1)
            print(f'validation loss: {loss}')
            print(f'validation accuracy: {acc}')
            print(f'validation f1 score: {f1}')

            if i == n_epochs - 1:
                final_pred = y_pred

        # plot
        print(f'per epoch train loss: {train_loss}')
        print(f'per epoch train accuracy: {train_accuracy}')
        print(f'per epoch validation loss: {val_loss}')
        print(f'per epoch validation accuracy: {val_accuracy}')
        print(f'per epoch validation f1 score: {val_f1}')
        plt.title(f'{n_samples}-samples\nMetrics')
        plt.plot(train_loss, label='train_loss')
        plt.plot(train_accuracy, label='train_accuracy')
        plt.plot(val_loss, label='val_loss')
        plt.plot(val_accuracy, label='val_accuracy')
        plt.plot(val_f1, label='val_f1')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
        plt.close()

        cm = get_confusion_matrix(
            y_true=y_test_one_hot, y_pred=final_pred, labels=labels)
        print(f'confusion matrix:\n{cm}')

        # save model as pickle
        save_model_filename = f"model_{n_samples}-samples_{test_ratio}-test_ratio_{random_seed}-random_seed_{n_epochs}-epochs_{batch_size}-batch_size.pkl"
        save_model(model=the_cnn, save_root=save_root,
                   save_filename=save_model_filename)

    elif RUN_TYPE == "load" or RUN_TYPE == "test":
        # predict
        # image read
        X_img_test = load_image(image_path_root=data_root,
                                image_paths=X_test, output_dim=image_output_dim)

        # one hot encoding
        y_test_one_hot = one_hot_encoding(y_test)

        y_pred, loss, acc, f1 = predict_model(
            model=the_cnn, X=X_img_test, y=y_test_one_hot)

        print(f'testing loss: {loss}')
        print(f'testing accuracy: {acc}')
        print(f'testing f1 score: {f1}')

        cm = get_confusion_matrix(
            y_true=y_test_one_hot, y_pred=y_pred, labels=labels)
        print(f'confusion matrix:\n{cm}')
