from train_1705092 import *
import sys

input_root = "."
model_filename = "1705092_model.pickle"
prediction_filename = "1705092_prediction.csv"

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_1705092.py <test_data_directory>")
        exit(1)

    test_data_dir = sys.argv[1]

    # Load model
    testing_model = load_model(
        load_root=input_root, load_filename=model_filename)

    # load image paths
    test_image_filenames = load_image_paths(test_data_dir)

    # load images
    test_images = load_images(image_path_root=test_data_dir,
                              image_paths=test_image_filenames, output_dim=image_output_dim)

    # predict
    predictions = test_predict_model(model=testing_model, X=test_images)

    # pdf
    pred_df = pd.DataFrame(test_image_filenames, columns=['FileName'])
    pred_df['Digit'] = predictions
    save_csv_path = Path(test_data_dir) / prediction_filename
    pred_df.to_csv(save_csv_path, index=False)
    print(f'Prediction saved to {save_csv_path}')
