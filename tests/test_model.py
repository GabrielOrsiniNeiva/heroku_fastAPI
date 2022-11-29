"""
Module for testing model.py

Author: Gabriel
Date: 11/2022
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from starter.data import process_data
from starter.model import inference, train_model


def test_train_model():
    """
    Tests model can fit

    """
    # Creating random uniform array for training
    np.random.seed(42)
    X_rtrain = np.random.uniform(0, 50, size=(100, 108))
    y_rtrain = np.random.randint(2, size=100).astype(bool)

    assert train_model(X_rtrain, y_rtrain, 42)


def test_inference(model_path, data_test_true):
    """
    Tests if inference returns expected values

    """
    df_test = pd.DataFrame(data_test_true, index=[0])

    try:
        model = joblib.load(f'{model_path}/model.pkl')
        encoder = joblib.load(f'{model_path}/encoder.pkl')
        lb = joblib.load(f'{model_path}/lb.pkl')

    except FileNotFoundError as e:
        raise FileNotFoundError(
            e,
            "Pickles not founded, try running it from root folder \
            or changing the path on conftest.py"
        )

    X, _y, _encoder, _lb = process_data(
        df_test,
        categorical_features=encoder['features'],
        encoder=encoder['encoder'],
        label='salary',
        lb=lb,
        training=False
    )

    assert inference(model, X) == 1
