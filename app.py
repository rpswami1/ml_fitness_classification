import sys
import subprocess
import importlib.util
import os

# -------------------------------------------------
# 0. AUTO-INSTALL DEPENDENCIES FROM requirements.txt
# -------------------------------------------------
def install_requirements():
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        return

    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    missing_packages = []
    for req in requirements:
        package_name = req.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0]
        import_name = package_name
        if package_name == "scikit-learn":
            import_name = "sklearn"
        
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(req)

    if missing_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            if "streamlit" in [pkg.split("==")[0] for pkg in missing_packages]:
                print("\nStreamlit has been installed. Please run the app again using:\nstreamlit run app.py")
                sys.exit(0)
        except subprocess.CalledProcessError:
            sys.exit(1)

install_requirements()

# -------------------------------------------------
# APP IMPORTS
# -------------------------------------------------
try:
    import streamlit as st
except ImportError:
    sys.exit(0)

import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------
# CUSTOM MODEL FROM SCRATCH (KNN)
# -------------------------------------------------
class KNNScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        # Get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

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
    # Load with basic cleaning for corrupted text in numeric columns
    # We read as object first to inspect, or just read normally and coerce later
    df = pd.read_csv(csv_file)
    st.write(f"Dataset loaded. Shape: {df.shape}")
    
    with st.expander("Raw Data Preview"):
        st.dataframe(df.head())

    # -------------------------------------------------
    # 2. DATA PREPROCESSING & CLEANING
    # -------------------------------------------------
    st.header("2. Preprocessing & Cleaning")
    
    # 2.1 Handling Corrupted Data (Non-numeric in numeric columns)
    # Heuristic: If a column name implies numeric (weight, height, age, income, score), force numeric
    numeric_keywords = ['weight', 'height', 'age', 'bmi', 'score', 'income', 'rate', 'duration']
    potential_numeric_cols = [c for c in df.columns if any(k in c.lower() for k in numeric_keywords)]
    
    for col in potential_numeric_cols:
        # Coerce to numeric, turning errors (strings) into NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2.2 Handling Missing Values
    if df.isnull().values.any():
        st.warning("Missing/Corrupted values detected.")
        st.write(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        
        # Fill numeric with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill categorical with mode
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        st.success("Imputed missing values with Median (Numeric) and Mode (Categorical).")
    else:
        st.info("No missing values found after cleaning.")

    # -------------------------------------------------
    # 3. FEATURE ENGINEERING
    # -------------------------------------------------
    st.header("3. Feature Engineering")
    
    # 3.1 BMI Calculation
    if 'weight' in df.columns and 'height' in df.columns:
        # Ensure height is in meters for BMI (assuming input is cm if > 3)
        # Simple heuristic: if mean height > 3, likely cm.
        height_mean = df['height'].mean()
        if height_mean > 3:
            df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
        else:
            df['BMI'] = df['weight'] / (df['height'] ** 2)
        st.write("✅ Created feature: `BMI`")
        
        # 3.2 BMI Categorization (Binning)
        # Underweight < 18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese >= 30
        bins = [0, 18.5, 25, 30, 100]
        labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
        df['BMI_Category'] = pd.cut(df['BMI'], bins=bins, labels=labels)
        st.write("✅ Created feature: `BMI_Category` (Binned from BMI)")

    # 3.3 One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    target_col = 'fitness_category' # Adjust if your dataset has a different target name
    
    # Verify target column exists
    if target_col not in df.columns:
        # Fallback or user selection could go here, but assuming dataset structure
        pass
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        st.write(f"✅ Label Encoded target: `{target_col}`")
        
        # Show mapping
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        st.json(mapping)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    st.write("✅ Performed One-Hot Encoding on categorical features.")
    
    with st.expander("Processed Data Preview"):
        st.dataframe(df.head())

    # -------------------------------------------------
    # 4. VISUALIZATION & SELECTION
    # -------------------------------------------------
    st.header("4. Visualization & Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Heatmap")
        fig_corr = plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
        st.pyplot(fig_corr)

    with col2:
        st.subheader("Feature Selection")
        threshold = st.slider("Correlation Threshold for Removal", 0.0, 0.5, 0.05)
        if target_col in correlation_matrix.columns:
            target_corr = correlation_matrix[target_col].abs()
            unnecessary_features = target_corr[target_corr < threshold].index.tolist()
            st.write(f"Features with correlation < {threshold}:")
            st.write(unnecessary_features)
            
            if st.checkbox("Drop these features?", value=True):
                df.drop(columns=unnecessary_features, inplace=True)
                st.success(f"Dropped {len(unnecessary_features)} features.")

    # -------------------------------------------------
    # 5. MODEL IMPLEMENTATION & TUNING
    # -------------------------------------------------
    st.header("5. Model Implementation")

    # Splitting
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting", "KNN (From Scratch)"]
    )
    
    enable_tuning = st.checkbox("Enable Hyperparameter Tuning (GridSearch)")

    if st.button("Train & Evaluate"):
        model = None
        best_params = None
        
        if model_choice == "KNN (From Scratch)":
            st.info("Training Custom KNN implementation...")
            # Tuning for custom model manually implemented for demo
            k_val = 5
            if enable_tuning:
                st.write("Simple tuning for KNN: Testing k=3, 5, 7...")
                best_acc = 0
                best_k = 3
                for k in [3, 5, 7]:
                    temp_model = KNNScratch(k=k)
                    temp_model.fit(X_train, y_train)
                    # Use a subset for speed if needed, but dataset is likely small
                    acc = accuracy_score(y_test, temp_model.predict(X_test))
                    if acc > best_acc:
                        best_acc = acc
                        best_k = k
                k_val = best_k
                best_params = {"k": best_k}
            
            model = KNNScratch(k=k_val)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        else:
            # Sklearn Models
            base_model = None
            param_grid = {}
            
            if model_choice == "Logistic Regression":
                base_model = LogisticRegression(max_iter=1000)
                param_grid = {'C': [0.1, 1, 10]}
            elif model_choice == "Random Forest":
                base_model = RandomForestClassifier(random_state=42)
                param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
            elif model_choice == "SVM":
                base_model = SVC()
                param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            elif model_choice == "Gradient Boosting":
                base_model = GradientBoostingClassifier(random_state=42)
                param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}

            if enable_tuning:
                with st.spinner(f"Tuning {model_choice}..."):
                    grid = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                    best_params = grid.best_params_
            else:
                model = base_model
                model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)

        # Results
        acc = accuracy_score(y_test, y_pred)
        st.subheader(f"Results for {model_choice}")
        st.metric("Accuracy", f"{acc:.4f}")
        
        if best_params:
            st.write("Best Hyperparameters found:", best_params)
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
        
        with col_res2:
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            st.pyplot(fig_cm)

else:
    st.error("CSV file not found. Please check the download process.")
