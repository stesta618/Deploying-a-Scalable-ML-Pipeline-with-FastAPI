# Model Card


## Model Details

This binary classification model was created using a random forest classifier. Hyperparameter tuning was not performed on this model.

## Intended Use

The model was developed using various demographic and employment inputs to predict whether an individual's income is either above $50,000 or below/equal to $50,000. The model can be extended and is intended for use in scenarios where income prediction is used in decision-making, such as targeted marketing or financial planning.

## Training Data

The training data contains income census data gathered by UC Irvine Machine Learning Repository in a csv file. The data contains 32,561 rows and 15 columns. Feature information can be found at the following link: https://archive.ics.uci.edu/dataset/20/census+income. The 8 categorical columns were preprocessed with OneHotEncoder and LabelBinarizer. The categorical features used for training included workclass, education, marital-status, occupation, relationship, race, sex, and native-country.

## Evaluation Data
The data was split at a ratio of 85:15, 85% of the data was used for training and 15% of the data was used for testing. The testing and training data were preprocessed in the same manner.

## Metrics
The model was evaluated on Precision, Recall and F1-score.
Results:
Precision: 0.7539 | Recall: 0.6448 | F1: 0.6951

## Ethical Considerations
The model may exhibit bias if the training data contains biases in its attributes. Care should be taken as the baises may impact model predictions. There are no privacy concerns since all personally identifiable information is left out of the dataset for it to remain publically available.

## Caveats and Recommendations
This model was trained on census data from 1994. Consideration should be used as training data is old and socioecoonmic changes may have occured since then.

Address any potential bias in the training data to ensure fair predictions. 
