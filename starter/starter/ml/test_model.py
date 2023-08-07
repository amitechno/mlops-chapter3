import numpy as np
import pytest
from ml.model import train_model, compute_model_metrics, inference

# Mock data for testing
X_train = np.array([[1, 2], [3, 4], [5, 6]])
y_train = np.array([0, 1, 0])
X_test = np.array([[7, 8], [9, 10]])
y_test = np.array([1, 0])

# Test train_model function
def test_train_model():
    model = train_model(X_train, y_train)
    assert hasattr(model, "predict")

# Test compute_model_metrics function
def test_compute_model_metrics():
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

# Test inference function
def test_inference():
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert len(preds) == len(X_test)




@pytest.mark.parametrize(
    "y_true, y_pred, expected_precision, expected_recall, expected_fbeta",
    [
        (np.array([0, 1, 0]), np.array([0, 1, 1]), 0.5, 1.0, 0.667),
        # Add more test cases as needed
    ],
)
def test_compute_model_metrics_parametrized(y_true, y_pred, expected_precision, expected_recall, expected_fbeta):
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert pytest.approx(precision, abs=1e-3) == expected_precision
    assert pytest.approx(recall, abs=1e-3) == expected_recall
    assert pytest.approx(fbeta, abs=1e-3) == expected_fbeta
