import pytest
# TODO: add necessary import
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import compute_model_metrics, train_model
from sklearn.ensemble import RandomForestClassifier

# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    Test if the train_model function is returning a model of the expected type
    """
    
    # Create a controlled dataset for testing
    data = pd.DataFrame(
        {
            "column_1": [1, 2, 3, 4, 5],
            "column_2": [0, 1, 0, 1, 0],
            "label": [0, 1, 0, 1, 0]
        }
    )

    # Process data
    X_train, y_train, _, _ = process_data(
        data,
        categorical_features=["column_2"],
        label="label",
        training=True
    )

    # Train model
    model = train_model(X_train, y_train)

    # Check if model returned by train_model is of the expected type
    expected_model_type = RandomForestClassifier

    assert isinstance(model, expected_model_type)


# TODO: implement the second test. Change the function name and input as needed
def test_model_algorithm():
    """
    Test the model's algorithm
    """
    
    # Create a controlled dataset for testing
    data = pd.DataFrame(
        {
            "column_1": [1, 2, 3, 4, 5],
            "column_2": [0, 1, 0, 1, 0],
            "label": [0, 1, 0, 1, 0],
        }
    )

    # Process data
    X_train, y_train, _, _ = process_data(
        data,
        categorical_features=["column_2"],
        label="label",
        training=True,
    )

    # Train model
    model = train_model(X_train, y_train)

    # Check if the model uses the expected algorithm
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    #Test if compute_model_metrics function returns the expected values
    """
    
    # Define a test set with controlled values
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])

    # Compute the metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Check if the computed metrics match the expected values
    expected_precision = 0.6667
    expected_recall = 1.0
    expected_fbeta = 0.8

    assert round(precision, 4) == expected_precision
    assert round(recall, 4) == expected_recall
    assert round(fbeta, 4) == expected_fbeta
