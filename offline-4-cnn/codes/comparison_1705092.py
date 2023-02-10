import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from train_1705092 import save_model


def get_metrics(ground_truth_path, prediction_path):
    # read ground truth and prediction
    ground_truth = pd.read_csv(ground_truth_path)
    prediction = pd.read_csv(prediction_path)

    # extract digit and filenames of ground_truth
    comparison = ground_truth[['filename', 'digit']]
    # merge with prediction
    # rename Digit to digit
    prediction = prediction.rename(columns={'Digit': 'digit'})
    prediction = prediction.rename(columns={'FileName': 'filename'})
    comparison = comparison.merge(
        prediction, on='filename', suffixes=('_truth', '_pred'))

    # calculate accuracy, f1 score and confusion matrix
    acc = accuracy_score(
        y_true=comparison['digit_truth'], y_pred=comparison['digit_pred'])
    f1 = f1_score(y_true=comparison['digit_truth'],
                  y_pred=comparison['digit_pred'], average='macro')
    cm = confusion_matrix(
        y_true=comparison['digit_truth'], y_pred=comparison['digit_pred'])

    # save metrics
    metrics = {
        'accuracy': acc,
        'f1': f1,
        'confusion_matrix': cm
    }

    return metrics


if __name__ == '__main__':
    # read from command line ground truth and prediction path
    if len(sys.argv) != 3:
        print("Usage: python comparison_1705092.py <ground_truth_path> <prediction_path>")
        exit(1)

    ground_truth_path = sys.argv[1]
    prediction_path = sys.argv[2]

    metrics = get_metrics(ground_truth_path, prediction_path)

    save_metrics_filename = f"{prediction_path}.pkl"
    save_model(model=metrics, save_root='.',
               save_filename=save_metrics_filename)

    # print metrics
    # print(f"Accuracy:\n{metrics['accuracy']}")
    # print(f"F1 score:\n{metrics['f1']}")
    # print(f"Confusion matrix:\n{metrics['confusion_matrix']}")
    with open(f'{prediction_path}.txt', 'w') as f:
        f.write(f"Accuracy:\n{metrics['accuracy']}\n")
        f.write(f"F1 score:\n{metrics['f1']}\n")
        f.write(f"Confusion matrix:\n{metrics['confusion_matrix']}\n")

    # plot
    sns_plot = sns.heatmap(metrics['confusion_matrix'],
                           annot=True, fmt='d', cmap='bone')
    # xlabel
    plt.xlabel('Predicted')
    # ylabel
    plt.ylabel('Truth')
    # plt.show()
    sns_plot.get_figure().savefig(f'{prediction_path}.pdf')
    plt.close()
