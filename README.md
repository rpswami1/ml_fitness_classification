# 🏋️‍♂️ Fitness Classification Project

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge)

This project implements an **end-to-end Machine Learning workflow** to predict whether an individual is **"Fit"** or **"Not Fit"** based on health and lifestyle metrics. The implementation includes robust data cleaning, feature engineering, and a comparative analysis of multiple classification models through an interactive **Streamlit dashboard**.

---

## 📌 a. Problem Statement

The primary goal of this project is to develop a machine learning application that classifies individuals into **"Fit"** or **"Unfit"** categories based on various health and lifestyle attributes. 

This binary classification problem aims to assist in **automated health assessment** using synthetic data that mimics real-world scenarios. The project specifically addresses real-world data quality challenges, including:
*   🔊 **Noise** in feature data.
*   ❓ **Missing values** in critical columns.
*   🔀 **Mixed data types** requiring standardization.

---

## 📊 b. Dataset Description

The dataset used is a **synthetic fitness classification dataset** designed to simulate real-world data quality issues.

*   **Source**: Synthetic dataset (`muhammedderric/fitness-classification-dataset-synthetic`).
*   **Target Variable**: `is_fit` (renamed from `fitness_category`), indicating fitness status.

### 🔑 Key Features

| Feature | Description |
| :--- | :--- |
| `age` | Age of the individual. |
| `weight_kg` | Weight in kilograms (contained outliers). |
| `height_m` | Height in meters. |
| `bmi` | **Body Mass Index** (calculated feature). |
| `gender` | Categorical gender feature. |
| `smokes` | Smoking status (contained mixed types like "yes"/1). |
| `sleep_hours` | Average daily sleep (contained missing values). |

### 🛠️ Data Quality Handling

| Issue | Strategy Implemented |
| :--- | :--- |
| **Mixed Data Types** | Standardized the `smokes` column to binary `0`/`1`. |
| **Missing Values** | Imputed `sleep_hours` using the **median**. |
| **Outliers** | Capped extreme values in `weight_kg` using the **IQR method**. |
| **Feature Engineering** | Calculated **BMI** from weight and height. |
| **Encoding** | **One-hot encoded** categorical variables. |
| **Scaling** | **Standardized** numerical features for model compatibility. |

---

## 🤖 c. Models Used

The following six machine learning models were implemented and evaluated to solve the classification problem.

### 🏆 Comparison Table of Evaluation Metrics

The table below summarizes the performance of each model on the test dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8000 | 0.9341 | 0.6667 | 0.8571 | 0.7500 | 0.6005 |
| **Decision Tree** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **kNN** | 0.9500 | 0.9725 | 1.0000 | 0.8571 | 0.9231 | 0.8921 |
| **Naive Bayes** | 0.8500 | 0.9231 | 0.7500 | 0.8571 | 0.8000 | 0.6847 |
| **Random Forest (Ensemble)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **XGBoost (Ensemble)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

### 🔍 Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Typically provides a solid baseline. It performs well if the decision boundary is linear but may struggle with complex non-linear relationships in the fitness data. |
| **Decision Tree** | Offers high interpretability but is prone to overfitting on the training data if not pruned. It captures non-linear patterns well but might have high variance. |
| **kNN** | Performance depends heavily on the scale of features (handled via standardization). It can be computationally expensive at prediction time and sensitive to noisy features. |
| **Naive Bayes** | Assumes feature independence, which might not hold true for health metrics (e.g., weight and BMI are correlated). Often serves as a fast, simple baseline. |
| **Random Forest (Ensemble)** | Generally outperforms single decision trees by reducing variance through bagging. It handles non-linearities and interactions between features (like Age and BMI) effectively. |
| **XGBoost (Ensemble)** | Often achieves the highest performance by sequentially correcting errors (boosting). It is robust to outliers and handles complex patterns efficiently, though it requires careful tuning. |

---

## 🚀 How to Run the Project

Follow these steps to set up and run the application locally.

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
Launch the interactive dashboard:
```bash
streamlit run app.py
```

### 3. Application Workflow
1.  **Dataset Loading**: The app automatically downloads or loads the dataset.
2.  **Preprocessing**: It performs data cleaning, feature engineering, and scaling.
3.  **Training**: Models are trained automatically on the processed data.
4.  **Testing**:
    *   Download the **Test Set CSV** provided in the app.
    *   **Upload** the Test Set CSV back into the app to view the evaluation metrics and populate the performance charts.
