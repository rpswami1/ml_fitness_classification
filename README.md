# ml_fitness_classification
This project implements an end-to-end Machine Learning workflow to predict whether an individual is "Fit" or "Not Fit" based on health and lifestyle metrics. The implementation includes data cleaning, feature engineering, and a comparative analysis of multiple classification models through an interactive Streamlit dashboard.

# Fitness Classification Project

## a. Problem Statement
The goal of this project is to develop a machine learning application that classifies individuals into "Fit" or "Unfit" categories based on various health and lifestyle attributes. This binary classification problem aims to assist in automated health assessment using synthetic data that mimics real-world scenarios, including data quality challenges such as noise, missing values, and mixed data types.

## b. Dataset Description
The dataset used is a synthetic fitness classification dataset designed to simulate real-world data quality issues.

- **Source**: Synthetic dataset (muhammedderric/fitness-classification-dataset-synthetic).
- **Target Variable**: `is_fit` (renamed from `fitness_category`), indicating if an individual is considered fit.
- **Key Features**:
    - `age`: Age of the individual.
    - `weight_kg`: Weight in kilograms (contained outliers).
    - `height_m`: Height in meters.
    - `bmi`: Body Mass Index (calculated feature).
    - `gender`: Categorical gender feature.
    - `smokes`: Smoking status (contained mixed types like "yes"/1).
    - `sleep_hours`: Average daily sleep (contained missing values).
    - Other lifestyle metrics.
- **Data Quality Handling**:
    - **Mixed Data Types**: Standardized the 'smokes' column to binary 0/1.
    - **Missing Values**: Imputed 'sleep_hours' using the median.
    - **Outliers**: Capped extreme values in 'weight_kg' using the IQR method.
    - **Feature Engineering**: Calculated BMI from weight and height.
    - **Encoding**: One-hot encoded categorical variables.
    - **Scaling**: Standardized numerical features for model compatibility.

## c. Models Used

The following six machine learning models were implemented and evaluated.

### Comparison Table of Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* |
| **Decision Tree** | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* |
| **kNN** | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* |
| **Naive Bayes** | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* |
| **Random Forest (Ensemble)** | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* |
| **XGBoost (Ensemble)** | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* | *[Value]* |

*(Note: Please replace `*[Value]*` with the actual results obtained from the Streamlit app dashboard after running the evaluation.)*

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Typically provides a solid baseline. It performs well if the decision boundary is linear but may struggle with complex non-linear relationships in the fitness data. |
| **Decision Tree** | Offers high interpretability but is prone to overfitting on the training data if not pruned. It captures non-linear patterns well but might have high variance. |
| **kNN** | Performance depends heavily on the scale of features (handled via standardization). It can be computationally expensive at prediction time and sensitive to noisy features. |
| **Naive Bayes** | Assumes feature independence, which might not hold true for health metrics (e.g., weight and BMI are correlated). Often serves as a fast, simple baseline. |
| **Random Forest (Ensemble)** | Generally outperforms single decision trees by reducing variance through bagging. It handles non-linearities and interactions between features (like Age and BMI) effectively. |
| **XGBoost (Ensemble)** | Often achieves the highest performance by sequentially correcting errors (boosting). It is robust to outliers and handles complex patterns efficiently, though it requires careful tuning. |

## How to Run the Project

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```
3.  **Workflow**:
    - The app will automatically download/load the dataset.
    - It performs data cleaning, feature engineering, and scaling.
    - Models are trained automatically.
    - You can download the Test Set CSV.
    - Upload the Test Set CSV to view the evaluation metrics and populate the table above.
