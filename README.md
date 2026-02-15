# 🏋️‍♂️ Fitness Classification Project

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge)

In this project, we've built a complete Machine Learning pipeline designed to determine if someone is "Fit" or "Not Fit" by analyzing their health and lifestyle data. We didn't just throw data at a model; we carefully cleaned it, engineered meaningful features, and then put several classification models to the test. Everything is brought together in an interactive Streamlit dashboard, making it easy to visualize and compare the results.

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
| **Logistic Regression** | • **Baseline Performance**: Achieved the lowest accuracy (80%), indicating the relationship between fitness and features is non-linear.<br>• **Feature Interaction**: Struggled to capture complex dependencies between `BMI`, `age`, and lifestyle factors.<br>• **Precision Issue**: Lower precision (0.67) implies a higher rate of false positives compared to other models.<br>• **Confusion Matrix**: Showed significant misclassification of "Unfit" individuals as "Fit". |
| **Decision Tree** | • **Perfect Fit**: Achieved 100% accuracy, suggesting the synthetic dataset follows clear, hierarchical rules.<br>• **Non-Linearity**: Successfully modeled non-linear decision boundaries that Logistic Regression missed.<br>• **Overfitting Risk**: While perfect here, a single tree is prone to memorizing noise; pruning would be necessary on real-world data.<br>• **Confusion Matrix**: Perfect diagonal, zero false positives or false negatives. |
| **kNN** | • **Strong Results**: High accuracy (95%) demonstrates the effectiveness of the `StandardScaler` applied during preprocessing.<br>• **Local Clustering**: Successfully identified clusters of "fit" individuals based on proximity in the feature space.<br>• **Noise Sensitivity**: Slightly lower performance than trees suggests sensitivity to boundary noise in features like `weight_kg`.<br>• **Confusion Matrix**: Very few misclassifications, mostly near the decision boundary. |
| **Naive Bayes** | • **Independence Violation**: Performance (85%) was limited because features like `weight_kg` and `BMI` are highly correlated, violating the model's core assumption.<br>• **Robust Baseline**: Despite theoretical limitations, it outperformed linear regression, proving useful as a quick, probabilistic baseline.<br>• **Recall**: Maintained decent recall (0.86), identifying most fit individuals but with less precision.<br>• **Confusion Matrix**: Higher false positive rate due to the independence assumption. |
| **Random Forest (Ensemble)** | • **Variance Reduction**: Achieved perfect scores (100%) by aggregating multiple trees, effectively neutralizing the risk of individual tree errors.<br>• **Feature Handling**: Excellent at managing the mix of continuous (`bmi`) and categorical (`smokes`) features without bias.<br>• **Stability**: More robust to potential overfitting than a single Decision Tree, making it a reliable choice for this classification task.<br>• **Confusion Matrix**: Flawless classification across all classes. |
| **XGBoost (Ensemble)** | • **Gradient Boosting Power**: Matched the perfect performance (100%) by iteratively correcting errors, ideal for the complex patterns in this dataset.<br>• **Outlier Robustness**: The algorithm's tree-building process is naturally resistant to the outliers present in `weight_kg`.<br>• **Optimization**: Likely converged faster on the optimal solution due to its advanced regularization parameters.<br>• **Confusion Matrix**: Perfect prediction accuracy, confirming its status as a top-tier classifier. |

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
python -m streamlit run app.py
```

### 3. Application Workflow
1.  **Dataset Loading**: The app automatically downloads or loads the dataset.
2.  **Preprocessing**: It performs data cleaning, feature engineering, and scaling.
3.  **Training**: Models are trained automatically on the processed data.
4.  **Testing**:
    *   Download the **Test Set CSV** provided in the app.
    *   **Upload** the Test Set CSV back into the app to view the evaluation metrics and populate the performance charts.
