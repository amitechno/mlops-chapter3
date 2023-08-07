import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
import joblib

# Load the data
csv_file_path = "starter/data/census.csv"
data = pd.read_csv(csv_file_path)

# Optional enhancement: Use K-fold cross-validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train the model
model = train_model(X_train, y_train)

# Save the trained model
model_filename = "starter/model/trained_model.joblib"
joblib.dump(model, model_filename)

# Optionally, you can load the model later using:
# loaded_model = joblib.load(model_filename)

# Evaluate the model
y_pred = model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

# Print the evaluation results
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", fbeta)
