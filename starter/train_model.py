import pandas as pd
from sklearn.model_selection import train_test_split
from .ml.data import process_data
from .ml.model import train_model, inference, compute_model_metrics
import joblib


def get_data(data_path):
    # Load the data
    data = pd.read_csv(data_path)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(
        data, test_size=0.20)

    return train_data, test_data


def train_and_save_model(
        train_data,
        model_path,
        cat_features,
        label_column='salary'):

    # Process the training data
    X_train, y_train, encoder, lb = process_data(
        train_data, categorical_features=cat_features,
        label=label_column,
        training=True)

    # Train the model
    model = train_model(X_train, y_train)

    # Save the trained model
    joblib.dump((model, encoder, lb), model_path)


def custom_prediction(test_data, model_path, cat_features,
                      label_column='salary'):
    # Load the model from `model_path`
    model, encoder, lb = joblib.load(model_path)

    # Process the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test_data,
        categorical_features=cat_features,
        label=label_column,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Evaluate model
    preds = inference(model=model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(
        y_test, preds)
    print('Precision:\t', precision)
    print('Recall:\t', recall)
    print('F-beta score:\t', fbeta)

    return precision, recall, fbeta
