# Model Card

## Model Details
- Developer: Amit Sharma
- Model Date: 7 Aug 23
- Model Version: 1.1.0
- Model Type: Classification
- Training Algorithm: Logistic Regression
- Training Parameters:
    - Regularization Penalty: L2
    - Regularization Strength (C): 1.0
    - Optimization Algorithm: lbfgs
    - Maximum Iterations: 100
- Features: age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,salary



## Intended Use
- Primary Intended Uses: Predicting income level based on Census Bureau data
- Primary Intended Users: Policymakers, researchers
- Out-of-Scope Use Cases: Other non-income-related predictions

## Factors
- Relevant Factors: Age, Education, Marital Status, Occupation, etc.
- Evaluation Factors: Demographic factors such as Age, Gender, and Ethnicity

## Metrics
- Model Performance Measures: The model was evaluated using F1 score,Recall and Precision for which values are as below:
   - Precision: 0.722117202268431
   - Recall:  0.24773022049286642
   - F1 score: 0.36890391115403187


## Evaluation Data
- Datasets: Census Bureau data
- Motivation: Publicly available dataset for evaluating income predictions
- Preprocessing: Data cleaning, handling missing values, and one-hot encoding categorical features

## Training Data
- Datasets: Census Bureau data
- Distribution over Factors: Details not provided due to data privacy constraints

## Quantitative Analyses
- Unitary Results: Model performance metrics on the evaluation dataset
- Intersectional Results: Model performance metrics for different demographic groups (e.g., age groups, genders)

## Ethical Considerations
- Data Bias: Potential bias in Census Bureau data regarding underrepresented groups
- Fairness: Ensuring fairness and minimizing disparate impact during model development
- Privacy: No personal identifiers used in the model
- Security: [Considerations regarding model security, if applicable]

## Caveats and Recommendations
- Limitations: Model performance may vary based on data quality and representation
- Recommendations: Regularly update the model with new data, consider fairness-aware techniques

