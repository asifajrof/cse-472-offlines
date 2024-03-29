import numpy as np
import pandas as pd


def load_dataset(csv_path):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :param csv_path: path to csv file
    :return:
    """
    # todo: implement
    X, y = None, None
    data_frame = pd.read_csv(csv_path)
    X = np.array(data_frame.iloc[:, 2:-1])  # 0,1 excluded. index and date
    y = np.array(data_frame.iloc[:, -1])
    # normalize
    for j in range(X.shape[1]):
        max = np.max(X[:, j])
        min = np.min(X[:, j])
        X[:, j] = (X[:, j] - min) / (max - min)

    X = np.insert(X, 0, 1, axis=1)
    # print(X)

    return X, y


def split_dataset(X, y, test_size=0.2, shuffle=True):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    X_train, y_train, X_test, y_test = None, None, None, None
    # index_list = np.arange(X.shape[0])
    # # shuffle
    # np.random.shuffle(index_list) if shuffle else None
    # # split
    # test_size_int = int(X.shape[0] * test_size)
    # X_test = X[index_list[:test_size_int]]
    # y_test = y[index_list[:test_size_int]]
    # X_train = X[index_list[test_size_int:]]
    # y_train = y[index_list[test_size_int:]]
    X_train = X
    y_train = y
    X_test1, y_test1 = load_dataset("datatest.txt")
    X_test2, y_test2 = load_dataset("datatest2.txt")
    return X_train, y_train, X_test1, y_test1, X_test2, y_test2


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    X_sample, y_sample = None, None
    index_list = np.random.choice(X.shape[0], X.shape[0], replace=True)
    X_sample = X[index_list]
    y_sample = y[index_list]
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
