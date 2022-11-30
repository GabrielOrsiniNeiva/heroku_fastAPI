import os
import pytest
import joblib


@pytest.fixture(scope='session')
def data_test_true(request):
    data_test_true = {
        'age': 33,
        'workclass': 'Private',
        'fnlgt': 185908,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'Black',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 55,
        'native-country': 'United-States',
        'salary': '<=50K'
    }

    return data_test_true


@pytest.fixture(scope='session')
def data_test_false(request):
    data_test_false = {
        'age': 45,
        'workclass': 'State-gov',
        'fnlgt': 50567,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Wife',
        'race': 'White',
        'sex': 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States',
        'salary': '>50K'
    }

    return data_test_false


@pytest.fixture(scope='session')
def encoder(request):
    encoder = joblib.load(
        f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/model/encoder.pkl'
    )

    return encoder


@pytest.fixture(scope='session')
def model(request):
    model = joblib.load(
        f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/model/model.pkl'
    )

    return model


@pytest.fixture(scope='session')
def lb(request):
    lb = joblib.load(
        f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/model/lb.pkl'
    )

    return lb
