from basic_components import *
from utils import *
from tqdm import tqdm
import numpy as np
import pickle

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
        save_model(model=the_cnn, save_root=save_root, save_filename=None)

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
