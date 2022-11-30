"""
Module for testing data.py

Author: Gabriel
Date: 11/2022
"""
import os
import sys

import sklearn
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from starter.data import process_data


def test_process_data_training_true(data_test_true, encoder):
    """
    Tests if data returns as expected performing process_data for training

    """
    df_test = pd.DataFrame(data_test_true, index=[0])

    X_train, y_train, encoder, lb = process_data(
        df_test,
        categorical_features=encoder['features'],
        label="salary",
        training=True
    )

    assert X_train.shape == (1, 14)
    assert y_train.shape == (1,)
    assert isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder)
    assert isinstance(lb, sklearn.preprocessing._label.LabelBinarizer)


def test_process_data_training_false(data_test_true, encoder, lb, model):
    """
    Tests if data returns as expected performing process_data for inference

    """
    df_test = pd.DataFrame(data_test_true, index=[0])

    X, _y, _encoder, _lb = process_data(
        df_test,
        categorical_features=encoder['features'],
        encoder=encoder['encoder'],
        label="salary",
        lb=lb,
        training=False
    )

    assert X.shape == (1, 108)
