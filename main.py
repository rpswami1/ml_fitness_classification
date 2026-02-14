import os
import subprocess
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------------------------
# 1. SETUP & DATASET DOWNLOAD
# -------------------------------------------------
def get_code_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

CODE_DIR = get_code_dir()
DATASET_SLUG = "muhammedderric/fitness-classification-dataset-synthetic"
MAIN_ZIP = "fitness-classification-dataset-synthetic.zip"
MAIN_ZIP_PATH = os.path.join(CODE_DIR, MAIN_ZIP)

print("Preparing Fitness Classification dataset (Kaggle)...")

if not os.path.exists(MAIN_ZIP_PATH):
    print("Downloading Kaggle ZIP into:", CODE_DIR)
    # Ensure kaggle.json is set up or environment variables are set for this to work
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", CODE_DIR],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have the Kaggle API installed and configured.")
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it using 'pip install kaggle'.")

# Extracting CSV from ZIP
if os.path.exists(MAIN_ZIP_PATH):
    with zipfile.ZipFile(MAIN_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(CODE_DIR)

# Find the CSV file
csv_file = os.path.join(CODE_DIR, "fitness_class_data.csv")
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    print("Dataset loaded. Shape:", df.shape)

    # -------------------------------------------------
    # 2. DATA PREPROCESSING & CLEANING
    # -------------------------------------------------
    print("Starting Preprocessing...")

    # Check for missing values
    if df.isnull().values.any():
        df = df.fillna(df.median(numeric_only=True))
        df = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == "object" else x)

    # -------------------------------------------------
    # 3. DATA ENGINEERING & FEATURE ENGINEERING
    # -------------------------------------------------
    # Feature Engineering: Creating new useful features
    # Example: BMI calculation if weight and height are present
    if 'weight' in df.columns and 'height' in df.columns:
        df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

    # One-Hot Encoding for categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Remove target variable from encoding if it is categorical
    target_col = 'fitness_category' # Assuming this is the target based on dataset name
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        # Label encode the target
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    # Perform One-Hot Encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # -------------------------------------------------
    # 4. VISUALIZATION & FEATURE SELECTION
    # -------------------------------------------------
    # Plotting Distribution of Features
    print("Plotting Feature Graphs...")
    # Using non-blocking show or saving figures might be better for an app, but keeping as is for now
    df.hist(figsize=(15, 10), bins=20)
    plt.tight_layout()
    plt.show()

    # Heatmap for correlation analysis
    print("Generating Heatmap...")
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Identify and remove unnecessary features (Low correlation with target or High Multicollinearity)
    # Threshold for feature removal (e.g., correlation with target < 0.1)
    threshold = 0.1
    if target_col in correlation_matrix.columns:
        target_corr = correlation_matrix[target_col].abs()
        unnecessary_features = target_corr[target_corr < threshold].index.tolist()

        print(f"Removing low-impact features: {unnecessary_features}")
        df.drop(columns=unnecessary_features, inplace=True)

    # -------------------------------------------------
    # 5. FINAL DATA PREPARATION
    # -------------------------------------------------
    print("Final Dataset Columns:", df.columns.tolist())
    print("✅ DONE: Data is ready for Model Implementation.")
else:
    print(f"CSV file not found at {csv_file}. Please check the download/extraction process.")
