<<<<<<< HEAD
from starter.train_model import train_and_save_model, get_data, batch_inference


CAT_FEATURES = [
=======
from starter.train_model import train_and_save_model, get_data, custom_prediction


cat_columns = [
>>>>>>> 7735632 (finalising code)
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

if __name__ == '__main__':
    data_path = 'data/census.csv'
    model_path = "model/census_model.pkl"
    print(model_path)

    # Get the splitted data
    train_data, test_data = get_data(data_path)
    # Training the model on the train data
    train_and_save_model(train_data, model_path, cat_columns)
    # evaluating the model on the test data
    precision, recall, f_beta = custom_prediction(test_data,
                                                model_path,
                                                cat_columns)
