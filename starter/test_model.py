import joblib
import numpy as np
import pandas as pd

from starter.data import process_data
from starter.model import inference, compute_model_metrics, train_model

def test_train_model():
    """
    Tests model can fit

    """
    # Creating random uniform array for training
    np.random.seed(42)
    X_rtrain = np.random.uniform(0, 50, size=(100, 108))
    y_rtrain = np.random.randint(2, size=100).astype(bool)

    assert (
        train_model(X_rtrain, y_rtrain, 42),
        'Model not fitted'
    )

def test_inference(model_path, df_test):
    """
    Tests if inference returns expected values

    """
    try:
        model = joblib.load(f'{model_path}/model.pkl')
        encoder = joblib.load(f'{model_path}/encoder.pkl')
        lb = joblib.load(f'{model_path}/lb.pkl')

    except FileNotFoundError as e:
        raise FileNotFoundError(e, 'Pickles not founded, try running it from root folder or changing the path on conftest.py')

    X, _y, _encoder, _lb = process_data(
        df_test,
        categorical_features=encoder['features'],
        encoder=encoder['encoder'],
        label='salary',
        lb=lb,
        training=False
    )
 
    assert (inference(model, X) == 1, 'Predicted value not expected')