import streamlit as st
import os
import subprocess
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(page_title="Fitness Classification App", layout="wide")

st.title("Fitness Classification App")

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

st.header("1. Dataset Setup")
st.write("Preparing Fitness Classification dataset (Kaggle)...")

if not os.path.exists(MAIN_ZIP_PATH):
    st.write(f"Downloading Kaggle ZIP into: {CODE_DIR}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", CODE_DIR],
            check=True
        )
        st.success("Download successful.")
    except Exception as e:
        st.error(f"Error downloading dataset: {e}")
        st.info("Please ensure you have the Kaggle API installed and configured (kaggle.json).")

# Extracting CSV from ZIP
if os.path.exists(MAIN_ZIP_PATH):
    with zipfile.ZipFile(MAIN_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(CODE_DIR)

# Find the CSV file
csv_file = os.path.join(CODE_DIR, "fitness_class_data.csv")
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    st.write(f"Dataset loaded. Shape: {df.shape}")
    st.dataframe(df.head())

    # -------------------------------------------------
    # 2. DATA PREPROCESSING & CLEANING
    # -------------------------------------------------
    st.header("2. Preprocessing & Cleaning")
    st.write("Starting Preprocessing...")

    # Check for missing values
    if df.isnull().values.any():
        st.write("Missing values found. Filling with median/mode.")
        df = df.fillna(df.median(numeric_only=True))
        df = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == "object" else x)
    else:
        st.write("No missing values found.")

    # -------------------------------------------------
    # 3. DATA ENGINEERING & FEATURE ENGINEERING
    # -------------------------------------------------
    st.header("3. Feature Engineering")
    
    # Example: BMI calculation if weight and height are present
    if 'weight' in df.columns and 'height' in df.columns:
        df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
        st.write("Added BMI column.")

    # One-Hot Encoding for categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    target_col = 'fitness_category' 
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        st.write(f"Label Encoded target column: {target_col}")

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    st.write("Performed One-Hot Encoding.")
    st.dataframe(df.head())

    # -------------------------------------------------
    # 4. VISUALIZATION & FEATURE SELECTION
    # -------------------------------------------------
    st.header("4. Visualization")
    
    st.subheader("Feature Distributions")
    # Create a figure explicitly to avoid issues with st.pyplot()
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca()
    df.hist(figsize=(15, 10), bins=20, ax=ax)
    # Since df.hist creates its own figure/axes, we might need to capture the current figure
    # But df.hist(ax=ax) is not standard pandas. 
    # Let's use the standard way and pass the figure.
    plt.clf() # Clear current figure
    df.hist(figsize=(15, 10), bins=20)
    st.pyplot(plt.gcf())

    st.subheader("Correlation Heatmap")
    fig_corr = plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    st.pyplot(fig_corr)

    st.header("5. Feature Selection")
    threshold = 0.1
    if target_col in correlation_matrix.columns:
        target_corr = correlation_matrix[target_col].abs()
        unnecessary_features = target_corr[target_corr < threshold].index.tolist()

        st.write(f"Removing low-impact features (correlation < {threshold}): {unnecessary_features}")
        df.drop(columns=unnecessary_features, inplace=True)

    # -------------------------------------------------
    # 5. FINAL DATA PREPARATION
    # -------------------------------------------------
    st.header("Final Data Preparation")
    st.write("Final Dataset Columns:", df.columns.tolist())
    st.success("✅ DONE: Data is ready for Model Implementation.")

else:
    st.error("CSV file not found.")
