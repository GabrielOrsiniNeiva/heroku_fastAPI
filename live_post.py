import json
import requests

data_test = {
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

r = requests.post(
    'https://gabrors-udacity-api.herokuapp.com/model_inference',
    data=json.dumps(data_test)
)

print(r.status_code)
print(r.json())
