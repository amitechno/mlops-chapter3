import pandas as pd
from main import CAT_FEATURES
from starter.train_model import batch_inference
from tabulate import tabulate


def compute_model_slices(feature_name, data_path, cat_features, model_path):
    # Load the data
    data = pd.read_csv(data_path)

    # Get distinct values for the chosen feature
    slices_values = data[feature_name].unique()

    # Compute model slices and store the metrics in a list
    model_slices_output = []
    for value in slices_values:
        # Filter the data for the current value of the feature
        slice_data = data[data[feature_name] == value]

        # Compute the metrics using batch_inference
        precision, recall, f_beta = batch_inference(
            slice_data, model_path, cat_features)

        # Append the metrics to the output list as a dictionary
        model_slices_output.append({
            "Value": value,
            "Precision": precision,
            "Recall": recall,
            "F-beta": f_beta
        })

    return model_slices_output


if __name__ == "__main__":
    data_path = 'data/census.csv'
    feature_name = "education"
    model_path = 'model/census_model.pkl'

    model_slices_output = compute_model_slices(
        feature_name, data_path, CAT_FEATURES, model_path)

    # Write the output to slice_output.txt in a tabular format
    with open("slice_output.txt", "w") as f:
        f.write(f"Feature: {feature_name}\n\n")
        f.write(tabulate(model_slices_output, headers="keys", tablefmt="grid"))
