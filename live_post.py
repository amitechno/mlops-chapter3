import requests

# URL of your FastAPI app
url = "https://project3-census-income.onrender.com/predict"

# Sample data for the request
data = {
    "age": 32,
    "workclass": "Private",
    "fnlgt": 205019,
    "education": "Assoc-acdm",
    "education_num": 12,
    "marital_status": "Never-married",
    "occupation": "Sales",
    "relationship": "Not-in-family",
    "race": "Black",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"
}

# Make the POST request
response = requests.post(url, json=data)

# Get the status code and response content
status_code = response.status_code
result = response.json()

# Print the result and status code
print("Status Code:", status_code)
print("Result:", result)
