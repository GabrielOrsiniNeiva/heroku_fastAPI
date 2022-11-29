import json

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ["Welcome to my API"]


def test_model_inference_true():
    data = {
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
        'native-country': 'United-States'
    }

    r = client.post(
        "/model_inference", 
        data=json.dumps(data)
    )

    assert r.status_code == 200
    assert r.json() == {"Result": "Salary > 50k"}

def test_model_inference_false():
    data = {
        'age':45,
        'workclass':'State-gov',
        'fnlgt':50567,
        'education':'HS-grad',
        'education-num':9,
        'marital-status':'Married-civ-spouse',
        'occupation':'Exec-managerial',
        'relationship':'Wife',
        'race':'White',
        'sex':'Female',
        'capital-gain':0,
        'capital-loss':0,
        'hours-per-week':40,
        'native-country':'United-States'
    }
    r = client.post(
        "/model_inference", 
        data=json.dumps(data)
    )

    assert r.status_code == 200
    assert r.json() == {"Result": "Salary <= 50k"}