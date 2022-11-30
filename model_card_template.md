# Model Card

Prediction model applied on the Census Income Data Set, provided in the following link:
https://archive.ics.uci.edu/ml/datasets/census+income

## Model Details
The model consist in a RandomForestClassfier, in combination with OneHotEnconding applied on categorical features. The target has been binarized with LabelBinarizer.

## Intended Use
This model is made only for study purpose.

## Training and Evaluation Data
Training data was generated from the Census Data Set with the train_test_split method from sklearn. The test_size wast settled to 30%.

## Metrics
Precision: 0.72
Recall: 0.62
F1: 0.67

Sliced data metrics can be found on the slice_output folder.

## Ethical Considerations
The data used in the model is sensitive and the prediction results may reinforce stereotypes and discriminations.

## Caveats and Recommendations
Based on the ethical considerations, this model should only be used for studying purpose.