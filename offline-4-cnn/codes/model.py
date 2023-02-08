from basic_components import *
from utils import *
from tqdm import tqdm
import numpy as np
import pickle

random_seed = 42
np.random.seed(random_seed)
test_ratio = 0.2
n_samples = 1000
n_epochs = 10
batch_size = 16
learning_rate = 0.001

NEW_RUN = False

data_root = "../Assignment4-Materials/NumtaDB_with_aug/"
csv_filenames = [
    "training-a.csv",
    "training-b.csv",
    "training-c.csv",
    # "training-d.csv",
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


def get_model(model_params):
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


def accuracy(y_pred, y_true):
    # y_pred: (n_samples,n_classes)
    # y_true: (n_samples,n_classes)
    # acc: scalar
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    acc = np.sum(y_pred == y_true) / y_pred.shape[0]
    return acc


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
        acc = accuracy(y_pred=forward_output, y_true=y_batch)
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


def save_model(model, save_root, save_filename):
    # pathlib Path make dir if not exist
    Path(save_root).mkdir(parents=True, exist_ok=True)
    if save_filename is None:
        save_filename = f"model_{n_samples}-samples_{test_ratio}-test_ratio_{random_seed}-random_seed_{n_epochs}-epochs_{batch_size}-batch_size.pkl"
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


def predict_model(model, X, y):
    # X: (n_samples, n_channels, height, width)
    # y: (n_samples, n_classes)
    # forward
    forward_output = X
    for layer in model:
        forward_output = layer.forward(input=forward_output)

    # calculate loss
    loss = cross_entropy_loss(y_pred=forward_output, y_true=y)
    # calculate accuracy
    acc = accuracy(y_pred=forward_output, y_true=y)

    # y_pred = np.argmax(forward_output, axis=1)
    # y_true = np.argmax(y, axis=1)

    # result_df = pd.DataFrame()
    # result_df['y_pred'] = y_pred
    # result_df['y_true'] = y_true
    return forward_output, loss, acc


if __name__ == "__main__":

    if NEW_RUN:
        the_cnn = get_model(model_params=lenet_model_params)
    else:
        the_cnn = load_model(
            load_root=save_root, load_filename="model_no_cache_1000-samples_0.2-test_ratio_42-random_seed_10-epochs_16-batch_size.pkl")

    # dataset read
    X_train, X_test, y_train, y_test = get_test_train_split(
        csv_root=data_root,
        csv_filenames=csv_filenames,
        n_samples=n_samples, test_size=test_ratio, random_state=random_seed
    )

    if NEW_RUN:
        # image read
        X_img_train = load_image(image_path_root=data_root,
                                 image_paths=X_train, output_dim=image_output_dim)

        # view an image
        # view_image_info(X=X_train, y=y_train, images=img_train, index=5)

        # one hot encoding
        y_train_one_hot = one_hot_encoding(y_train)

        # epoch
        epoch_loss = []
        epoch_accuracy = []
        for i in range(n_epochs):
            print(f'epoch: {i}')
            loss, acc = train_model(
                model=the_cnn, X=X_img_train, y=y_train_one_hot,
                batch_size=batch_size, learning_rate=learning_rate)
            epoch_loss.append(loss)
            epoch_accuracy.append(acc)

        # plot
        print(f'per epoch loss: {epoch_loss}')
        print(f'per epoch accuracy: {epoch_accuracy}')
        plt.plot(epoch_loss, label='loss')
        plt.plot(epoch_accuracy, label='accuracy')
        plt.legend()
        plt.show()
        plt.close()

        # save model as pickle
        save_model(model=the_cnn, save_root=save_root, save_filename=None)

    # predict
    # image read
    X_img_test = load_image(image_path_root=data_root,
                            image_paths=X_test, output_dim=image_output_dim)

    # one hot encoding
    y_test_one_hot = one_hot_encoding(y_test)

    y_pred, loss, acc = predict_model(
        model=the_cnn, X=X_img_test, y=y_test_one_hot)

    print(f'loss: {loss}')
    print(f'accuracy: {acc}')
    # # plot the probability distribution
    # plt.hist(y_pred, bins=y_pred.shape[1])
    # plt.show()
    # plt.close()
    # # view
    # for i in range(len(y_test)):
    #     view_image_info(X=X_test, y=y_test, images=img_test, index=i)
