import sklearn
import pandas as pd

from starter.data import process_data

def test_process_data_training_true(df_test, encoder):
    """
    Tests if data returns as expected performing process_data for training

    """
    X_train, y_train, encoder, lb = process_data(
        df_test,
        categorical_features=encoder['features'],
        label="salary",
        training=True
    )

    assert (X_train.shape == (1, 14), 'X Shape not expected')
    assert (y_train.shape == (1,), 'Y Shape not expected')
    assert type(encoder) == sklearn.preprocessing._encoders.OneHotEncoder
    assert type(lb) == sklearn.preprocessing._label.LabelBinarizer

def test_process_data_training_false(df_test, encoder, lb, model):
    """
    Tests if data returns as expected performing process_data for inference

    """
    X, _y, _encoder, _lb = process_data(
        df_test,
        categorical_features=encoder['features'],
        encoder=encoder['encoder'],
        label="salary",
        lb=lb,
        training=False
    )
    
    assert (X.shape == (1, 108), 'X Shape not expected')