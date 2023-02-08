from basic_components import *
from utils import *
from tqdm import tqdm


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


def train_model(model, X, y):
    image_path_root = "../Assignment4-Materials/NumtaDB_with_aug"
    # X: (n_samples, n_channels, height, width)
    # y: (n_samples, n_classes)
    # batching and training
    batch_size = 32
    num_batches = X.shape[0] // batch_size
    # loss
    batch_loss = []
    batch_acc = []
    for batch in tqdm(range(num_batches)):
        # get the batch
        batch_start = batch * batch_size
        batch_end = (batch + 1) * batch_size
        if batch_end > X.shape[0]:
            batch_end = X.shape[0]
        # X_batch = X[batch_start:batch_end]
        # load image
        X_batch = load_image(image_path_root=image_path_root,
                             image_paths=X[batch_start:batch_end])

        # forward
        # print(f'forward\r')
        forward_output = X_batch
        for layer in model:
            # print(f'layer: {layer.__class__.__name__}')
            forward_output = layer.forward(input=forward_output)

        # calculate loss and accuracy
        loss = cross_entropy_loss(
            y_pred=forward_output, y_true=y[batch_start:batch_end])
        batch_loss.append(loss)
        # print(f'loss: {loss}\r')
        # calc accuracy
        y_pred_num = np.argmax(forward_output, axis=1)
        y_true_num = np.argmax(y[batch_start:batch_end], axis=1)
        accuracy = np.sum(y_pred_num == y_true_num) / y_pred_num.shape[0]
        batch_acc.append(accuracy)
        # print(f'accuracy: {accuracy}\r')
        # backward
        # print(f'backward\r')
        grad = forward_output - y[batch_start:batch_end]
        learning_rate = 0.001
        # reverse the model
        model.reverse()
        for layer in model:
            # print(f'layer: {layer.__class__.__name__}')
            grad = layer.backward(
                output_error=grad, learning_rate=learning_rate)
        # reverse the model back
        model.reverse()
    # loss
    # avg
    avg_loss = np.mean(batch_loss)
    avg_acc = np.mean(batch_acc)
    return avg_loss, avg_acc


# main
the_cnn = get_model()


# dataset read
X_train, X_test, y_train, y_test = get_test_train_split(
    n_samples=10000, test_size=0.2, random_state=42
)

# image read
# image_path_root = "../Assignment4-Materials/NumtaDB_with_aug"
# img_train = load_image(image_path_root=image_path_root,
#                        image_paths=X_train, output_dim=(28, 28))

# view an image
# view_image_info(X=X_train, y=y_train, images=img_train, index=5)

# one hot encoding
y_train_one_hot = one_hot_encoding(y_train)

# epoch
epoch = 5
losses = []
accuracies = []
for i in range(epoch):
    print(f'epoch: {i}')
    # loss, accuracy = train_model(model=the_cnn, X=img_train, y=y_train_one_hot)
    loss, accuracy = train_model(model=the_cnn, X=X_train, y=y_train_one_hot)
    losses.append(loss)
    accuracies.append(accuracy)

# plot
print(f'losses: {losses}')
print(f'accuracies: {accuracies}')
plt.plot(losses, label='loss')
plt.plot(accuracies, label='accuracy')
plt.legend()
plt.show()
plt.close()

# predict
# image read
image_path_root = "../Assignment4-Materials/NumtaDB_with_aug"
img_test = load_image(image_path_root=image_path_root,
                      image_paths=X_test, output_dim=(28, 28))

# forward
print(f'forward\r')
forward_output = img_test
for layer in the_cnn:
    # print(f'layer: {layer.__class__.__name__}')
    forward_output = layer.forward(input=forward_output)

# predict
y_pred = np.argmax(forward_output, axis=1)
# print(f'y_pred: {y_pred}')
# print(f'y_test: {y_test}')
result_df = pd.DataFrame()
result_df['y_pred'] = y_pred
result_df['y_test'] = np.array(y_test)
result_df['correct'] = result_df['y_pred'] == result_df['y_test']
print(result_df)
print(f'accuracy: {result_df["correct"].sum() / result_df.shape[0]}')

# view
# for i in range(len(y_test)):
#     view_image_info(X=X_test, y=y_test, images=img_test, index=i)
