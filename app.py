import sys
import types

# -------------------------------------------------
# PATCH FOR PYTHON 3.13 (Missing imghdr)
# -------------------------------------------------
# imghdr was removed in Python 3.13, but Streamlit still imports it.
# We inject a dummy module into sys.modules to prevent ModuleNotFoundError.
if sys.version_info >= (3, 13):
    if 'imghdr' not in sys.modules:
        imghdr_mock = types.ModuleType('imghdr')
        imghdr_mock.what = lambda file, h=None: None
        imghdr_mock.tests = []
        sys.modules['imghdr'] = imghdr_mock

import subprocess
import importlib.util
import os

# -------------------------------------------------
# 0. AUTO-INSTALL DEPENDENCIES FROM requirements.txt
# -------------------------------------------------
def install_requirements():
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"Warning: {requirements_file} not found. Skipping auto-installation.")
        return

    print(f"Checking dependencies from {requirements_file}...")
    
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    missing_packages = []
    for req in requirements:
        # Handle version specifiers if present (e.g., pandas>=1.0)
        package_name = req.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0]
        
        # Mapping for packages where import name differs from install name
        import_name = package_name
        if package_name == "scikit-learn":
            import_name = "sklearn"
        elif package_name == "altair":
            import_name = "altair"

        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(req)

    if missing_packages:
        print(f"Missing packages found: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("All missing packages installed successfully.")
            
            if "streamlit" in [pkg.split("==")[0] for pkg in missing_packages]:
                print("\nStreamlit has been installed. Please run the app again using:\nstreamlit run app.py")
                sys.exit(0)
                
        except subprocess.CalledProcessError as e:
            print(f"Failed to install packages. Error: {e}")
            sys.exit(1)
    else:
        print("All dependencies are already satisfied.")

install_requirements()

# -------------------------------------------------
# APP IMPORTS
# -------------------------------------------------
try:
    import streamlit as st
except ImportError:
    print("Streamlit installed but import failed. Please restart the script.")
    sys.exit(0)

import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
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

if os.path.exists(MAIN_ZIP_PATH):
    with zipfile.ZipFile(MAIN_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(CODE_DIR)

csv_file = os.path.join(CODE_DIR, "fitness_class_data.csv")

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    st.write(f"Dataset loaded. Shape: {df.shape}")
    
    with st.expander("Raw Data Preview"):
        st.dataframe(df.head())

    # -------------------------------------------------
    # 2. DATA PREPROCESSING & CLEANING
    # -------------------------------------------------
    st.header("2. Preprocessing & Cleaning")
    
    # 2.1 Handling Corrupted Data
    numeric_keywords = ['weight', 'height', 'age', 'bmi', 'score', 'income', 'rate', 'duration']
    potential_numeric_cols = [c for c in df.columns if any(k in c.lower() for k in numeric_keywords)]
    
    for col in potential_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2.2 Handling Missing Values
    if df.isnull().values.any():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        st.success("Imputed missing values.")

    # -------------------------------------------------
    # 3. FEATURE ENGINEERING
    # -------------------------------------------------
    st.header("3. Feature Engineering")
    
    # 3.1 BMI Calculation
    if 'weight' in df.columns and 'height' in df.columns:
        height_mean = df['height'].mean()
        if height_mean > 3:
            df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
        else:
            df['BMI'] = df['weight'] / (df['height'] ** 2)
        st.write("✅ Created feature: `BMI`")

    # 3.2 One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    target_col = 'fitness_category' 
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        st.write(f"✅ Label Encoded target: `{target_col}`")

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    st.write("✅ Performed One-Hot Encoding.")

    # -------------------------------------------------
    # 4. FEATURE SELECTION
    # -------------------------------------------------
    st.header("4. Feature Selection")
    
    st.subheader("Correlation Heatmap")
    fig_corr = plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    st.pyplot(fig_corr)

    # Automatic Feature Selection based on Correlation Threshold
    threshold = 0.1
    st.write(f"Applying Correlation Threshold: {threshold}")
    
    if target_col in correlation_matrix.columns:
        target_corr = correlation_matrix[target_col].abs()
        unnecessary_features = target_corr[target_corr < threshold].index.tolist()
        
        if unnecessary_features:
            st.write(f"Dropping low-impact features (< {threshold} correlation): {unnecessary_features}")
            df.drop(columns=unnecessary_features, inplace=True)
            st.success(f"Dropped {len(unnecessary_features)} features.")
        else:
            st.info("No features dropped (all meet the correlation threshold).")

    st.write(f"Final Dataset Shape: {df.shape}")

    # -------------------------------------------------
    # 5. MODEL IMPLEMENTATION
    # -------------------------------------------------
    st.header("5. Model Implementation & Evaluation")

    # Splitting
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Model Dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    if st.button("Train All Models"):
        st.write("Training models... please wait.")
        results = []
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # AUC Score (Handle multi-class if needed)
            auc = "N/A"
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    # Check if binary or multi-class
                    if len(np.unique(y)) == 2:
                        auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            except Exception:
                pass

            results.append({
                "Model": name,
                "Accuracy": acc,
                "AUC Score": auc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            })

        # Display Results Table
        results_df = pd.DataFrame(results)
        st.subheader("Model Comparison")
        st.dataframe(results_df.style.format({
            "Accuracy": "{:.4f}",
            "AUC Score": "{:.4f}", 
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}"
        }))

        # Visualization of Accuracy
        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis", ax=ax)
        plt.xlim(0, 1.0)
        st.pyplot(fig)

else:
    st.error("CSV file not found. Please check the download process.")
