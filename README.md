# Project 3: Census Income Prediction

This project aims to predict whether an individual's income is greater than $50K per year or not using a machine learning model. It uses a FastAPI-based web service to accept input data and provide predictions, along with various features such as data validation, model explanation, and more.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- - [DVC Installation](#dvc-installation)
- [Usage](#usage)
- - [Data Slicing](#data-slicing)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Continuous Integration and Deployment](#continuous-integration-and-deployment)
- [Code Coverage](#code-coverage)
- [Data Version Control](#data-version-control)
- [License](#license)

## Project Overview

The project involves the following main components:

- **app.py**: The main FastAPI application that provides the web service for making predictions.
- **train_model.py**: Module for training the machine learning model.
- **data**: Directory containing the dataset used for training the model.
- **tests**: Directory containing test scripts for testing the application.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/project3.git
   cd mlops-chapter3
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   conda create -n project3-env python=3.8
   conda activate project3-env
   conda install --file requirements.txt
   ```

### DVC Installation

Install DVC and configure it to use the S3 remote for data version control:

```bash
pip install dvc dvc[s3]
dvc remote add -d s3remote s3://your-bucket-name/path/to/data
```

## Usage

1. Start the FastAPI application:

   ```bash
   uvicorn app:app --reload
   ```

2. Open a web browser and go to `http://localhost:8000` to access the API.

### Data Slicing

Run the data slicing script to compute performance on model slices:

```bash
python compute_slices_script.py
```

## API Documentation

- API documentation and interactive Swagger UI can be accessed at `http://localhost:8000/docs`.

## Testing

Run tests using the following command:

```bash
pytest test
```

## Continuous Integration and Deployment

The project includes GitHub Actions for continuous integration. Commits to the `main` branch are automatically tested and deployed.
Render is used for the continuous deployment.

## Code Coverage

To check code coverage, run:

```bash
coverage run -m pytest test
coverage report -m
```



## License

This project is licensed under the [MIT License](LICENSE.txt).
```
