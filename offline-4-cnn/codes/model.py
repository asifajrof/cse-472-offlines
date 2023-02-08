from basic_components import *
from utils import *


def get_model():
    # Define the model
    # lenet
    # conv1: 6@28x28
    # pool1: 6@14x14
    # conv2: 16@10x10
    # pool2: 16@5x5
    # fc1: 120
    # fc2: 84
    # fc3: 10

    model = []

    model.append(Convolution(
        num_filters=6, filter_dim=(5, 5), stride=1, padding=0))
    model.append(ReLU())
    model.append(MaxPooling(filter_dim=(2, 2), stride=2))
    model.append(Convolution(num_filters=16,
                 filter_dim=(5, 5), stride=1, padding=0))
    model.append(ReLU())
    model.append(MaxPooling(filter_dim=(2, 2), stride=2))
    model.append(Flattening())
    model.append(FullyConnected(output_dim=120))
    model.append(ReLU())
    model.append(FullyConnected(output_dim=84))
    model.append(ReLU())
    model.append(FullyConnected(output_dim=10))
    model.append(Softmax())

    return model


def train_model(model, X):
    layer_input = X
    for layer in model:
        layer_input = layer.forward(input=layer_input)

    grad = None
    learning_rate = 0.01
    for layer in model:
        grad = layer.backward(output_error=grad, learning_rate=learning_rate)


the_cnn = get_model()

# dataset read
X_train, X_test, y_train, y_test = get_test_train_split(
    n_samples=100, test_size=0.2, random_state=42
)

# image read
