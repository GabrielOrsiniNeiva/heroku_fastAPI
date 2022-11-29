"""
Script to train machine learning model.

Author: Udacity + Gabriel
Date: 11/2022
"""
import logging
import argparse

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from starter import process_data, inference, compute_model_metrics, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    data = pd.read_csv(args.data_path)

    logger.info("Processing data")
    train, test = train_test_split(
        data, test_size=args.test_size, random_state=args.random_seed)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    X_test, y_test, _encoder, _lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    encoder = {
        'encoder': encoder,
        'features': cat_features
    }

    # Trainning
    logger.info("Fitting predictor")
    model = train_model(X_train, y_train, args.random_seed)

    # Predict, save and log metrics
    y_pred = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, y_pred)
    logger.info(
        f"Precision: {precision:0.2f}, Recall: {recall:0.2f}, F1: {f1:0.2f}")

    logger.info(f"Saving models on {args.model_output_path}")
    joblib.dump(encoder, f'{args.model_output_path}/encoder.pkl')
    joblib.dump(model, f'{args.model_output_path}/model.pkl')
    joblib.dump(lb, f'{args.model_output_path}/lb.pkl')

    # Saving metrics on sliced data
    for feature in cat_features:
        slice_performance(
            test,
            y_test,
            y_pred,
            feature,
            args.results_output_path)


def slice_performance(df, y_true, y_pred, feature, path):
    """
    Calculate model metrics on slices of the dataset and save on results

    Inputs
    ------
    df : pd.dataframe
        Full dataframe, for filtering the slice
    y_true : np.array
        y array with real values
    y_pred : np.array
        y array with predicted values
    feature : str
        Name of the feature for performing slice
    path: str
        Path to save the file
    """

    # Setting positions for Plot
    len_features = len(df[feature].unique())
    title_pos = (0.1 * len_features + 0.1)
    x_pos = 0

    # Plot Titles
    plt.rc('figure', figsize=(5, 5), facecolor='white')
    plt.text(0.0,
             title_pos,
             "Value",
             {'fontsize': 10},
             fontproperties='monospace',
             weight='bold')
    plt.text(0.5,
             title_pos,
             "Precision",
             {'fontsize': 10},
             fontproperties='monospace',
             weight='bold')
    plt.text(1.0,
             title_pos,
             "Recall",
             {'fontsize': 10},
             fontproperties='monospace',
             weight='bold')
    plt.text(1.5,
             title_pos,
             "F1",
             {'fontsize': 10},
             fontproperties='monospace',
             weight='bold')

    for slice_val in df[feature].unique():
        # Filter dataframe
        df_temp = df.reset_index()
        df_temp = df_temp[df_temp[feature] == slice_val]

        # Filter numpy array, based on index from filtered dataframe
        np_index = df_temp.index.to_list()

        # Calculate metric
        precision, recall, f1 = compute_model_metrics(
            y_true[np_index],
            y_pred[np_index]
        )

        # Plotting text
        plt.text(
            0.0, x_pos, f"{slice_val}", {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.5, x_pos, f"{precision:0.2f}", {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            1.0, x_pos, f"{recall:0.2f}", {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            1.5, x_pos, f"{f1:0.2f}", {
                'fontsize': 10}, fontproperties='monospace')

        # Adding 0.1 to the X position for the next slice
        x_pos += 0.1

    plt.axis('off')
    plt.savefig(f'{path}/categorical_{feature}.png', bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Perform data process and training")

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data in CSV",
        default='data/census.csv',
        required=False,
    )

    parser.add_argument(
        "--model_output_path",
        type=str,
        help="Path to the trained model",
        default='model/',
        required=False,
    )

    parser.add_argument(
        "--results_output_path",
        type=str,
        help="Path to the results images",
        default='results/',
        required=False,
    )

    parser.add_argument(
        "--test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset",
        default=0.3,
        required=False,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    args = parser.parse_args()

    go(args)
