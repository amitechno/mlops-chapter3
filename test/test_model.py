import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from starter.ml.model import train_model, compute_model_metrics, inference


def generate_data():
    X, y = make_classification(
        n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def test_train_model():
    X_train, _, y_train, _ = generate_data()
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)


def test_compute_model_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_inference():
    X_train, X_test, y_train, _ = generate_data()
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert len(preds) == X_test.shape[0]
