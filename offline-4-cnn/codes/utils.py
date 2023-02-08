import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt

csv_root = "../Assignment4-Materials/NumtaDB_with_aug/"
csv_filenames = [
    # "training-a.csv",
    "training-b.csv",
    # "training-c.csv",
    # "training-d.csv",
    # "training-e.csv"
]


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


def get_test_train_split(n_samples=5000, test_size=0.2, random_state=42):
    merged_df = read_all_csv(csv_root, csv_filenames)
    dataset_df = get_dataset(
        merged_df, n_samples=n_samples, random_state=random_state)

    X = dataset_df["database name"] + "/" + dataset_df["filename"]
    y = dataset_df["digit"]

    # stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test


def load_image(image_path_root, image_paths, output_dim=(28, 28)):
    images = []
    for image_path in image_paths:
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
    print(f'file: {X.iloc[index]}')
    print(f'label: {y.iloc[index]}')
    # channel, height, width
    image = images[index].transpose(1, 2, 0)
    plt.imshow(image)
    plt.show()
    plt.close()
