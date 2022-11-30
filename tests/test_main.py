import os
import sys
import json

from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ["Welcome to my API"]


def test_model_inference_true(data_test_true):

    data_pop = data_test_true.copy()
    data_pop.pop('salary', None)

    r = client.post(
        "/model_inference",
        data=json.dumps(data_pop)
    )

    assert r.status_code == 200
    assert r.json() == {"Result": "Salary > 50k"}


def test_model_inference_false(data_test_false):

    data_pop = data_test_false.copy()
    data_pop.pop('salary', None)

    r = client.post(
        "/model_inference",
        data=json.dumps(data_pop)
    )

    assert r.status_code == 200
    assert r.json() == {"Result": "Salary <= 50k"}
