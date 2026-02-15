import sys
import subprocess
import importlib.util
import os
import io

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
        package_name = req.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0]
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

# Import models from the 'model' directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

try:
    from model.logistic_regression import train_logistic_regression
    from model.decision_tree import train_decision_tree
    from model.knn import train_knn
    from model.naive_bayes import train_naive_bayes
    from model.random_forest import train_random_forest
    from model.xgboost_model import train_xgboost
except ImportError as e:
    st.error(f"Error importing model modules: {e}")
    st.stop()

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def clean_and_engineer_features(df_in):
    """
    Applies cleaning and feature engineering to a dataframe.
    Returns the modified dataframe.
    """
    df_out = df_in.copy()
    
    # 1. Standardize 'smokes'
    if 'smokes' in df_out.columns:
        df_out['smokes'] = df_out['smokes'].astype(str).str.lower().str.strip()
        smokes_map = {'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
        df_out['smokes'] = df_out['smokes'].map(smokes_map)
        if df_out['smokes'].isnull().any():
             df_out['smokes'] = df_out['smokes'].fillna(df_out['smokes'].mode()[0])

    # 2. Impute 'sleep_hours'
    if 'sleep_hours' in df_out.columns:
        df_out['sleep_hours'] = df_out['sleep_hours'].fillna(df_out['sleep_hours'].median())

    # 3. Handle Outliers in 'weight_kg' (Capping)
    weight_col = 'weight_kg' if 'weight_kg' in df_out.columns else ('weight' if 'weight' in df_out.columns else None)
    if weight_col:
        df_out[weight_col] = pd.to_numeric(df_out[weight_col], errors='coerce')
        Q1 = df_out[weight_col].quantile(0.25)
        Q3 = df_out[weight_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_out[weight_col] = np.clip(df_out[weight_col], lower_bound, upper_bound)

    # 4. General Numeric Cleaning
    numeric_keywords = ['weight', 'height', 'age', 'bmi', 'score', 'income', 'rate', 'duration']
    potential_numeric_cols = [c for c in df_out.columns if any(k in c.lower() for k in numeric_keywords)]
    for col in potential_numeric_cols:
        if col != 'smokes':
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')

    # 5. General Imputation
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[numeric_cols] = df_out[numeric_cols].fillna(df_out[numeric_cols].median())
    
    object_cols = df_out.select_dtypes(include=['object']).columns
    for col in object_cols:
        if not df_out[col].mode().empty:
            df_out[col] = df_out[col].fillna(df_out[col].mode()[0])

    # 6. Feature Engineering: BMI
    w_col = 'weight_kg' if 'weight_kg' in df_out.columns else ('weight' if 'weight' in df_out.columns else None)
    h_col = 'height_m' if 'height_m' in df_out.columns else ('height' if 'height' in df_out.columns else None)

    if w_col and h_col:
        height_mean = df_out[h_col].mean()
        if height_mean > 3: # Assume cm
            df_out['BMI'] = df_out[w_col] / ((df_out[h_col] / 100) ** 2)
        else: # Assume m
            df_out['BMI'] = df_out[w_col] / (df_out[h_col] ** 2)
            
    return df_out

def prepare_data_for_model(df_in, target_col, train_columns=None, scaler=None):
    """
    Encodes, aligns columns, and scales data.
    If train_columns and scaler are provided, it aligns/scales to match training data.
    Otherwise, it prepares training data and returns columns/scaler.
    """
    df_proc = df_in.copy()
    
    # Label Encode Target
    if target_col in df_proc.columns:
        le = LabelEncoder()
        df_proc[target_col] = le.fit_transform(df_proc[target_col])
    
    # One-Hot Encoding
    categorical_cols = df_proc.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        
    df_proc = pd.get_dummies(df_proc, columns=categorical_cols, drop_first=True)
    
    # Separate X and y
    if target_col in df_proc.columns:
        X = df_proc.drop(columns=[target_col])
        y = df_proc[target_col]
    else:
        X = df_proc
        y = None

    # Align Columns
    if train_columns is not None:
        # Add missing columns with 0
        for col in train_columns:
            if col not in X.columns:
                X[col] = 0
        # Drop extra columns
        X = X[train_columns]
        # Ensure order
        X = X[train_columns]
    else:
        train_columns = X.columns.tolist()

    # Scaling
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
        
    return X_scaled, y, train_columns, scaler

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

# Option to upload dataset
uploaded_file = st.file_uploader("Upload your Fitness Dataset (CSV)", type=["csv"])

df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded dataset '{uploaded_file.name}' loaded successfully!")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
else:
    csv_file = os.path.join(CODE_DIR, "fitness_class_data.csv")
    user_provided_csv = os.path.join(CODE_DIR, "fitness_dataset.csv")

    if os.path.exists(user_provided_csv):
        csv_file = user_provided_csv
        st.success(f"Found local dataset: {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file)
    elif os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
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
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

if df is not None:
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Rename target if needed
    if 'fitness_category' in df.columns and 'is_fit' not in df.columns:
        df.rename(columns={'fitness_category': 'is_fit'}, inplace=True)
        st.info("Renamed 'fitness_category' column to 'is_fit'.")
    
    st.write(f"Dataset loaded. Shape: {df.shape}")
    st.subheader("Raw Data Preview")
    st.dataframe(df)

    target_col_raw = 'is_fit'
    if target_col_raw not in df.columns:
        st.error(f"Target column '{target_col_raw}' not found.")
        likely_targets = [c for c in df.columns if 'fitness' in c.lower() or 'category' in c.lower() or 'class' in c.lower()]
        if likely_targets:
            target_col_raw = st.selectbox("Select Target Column", likely_targets)

    # -------------------------------------------------
    # 2. VISUALIZATION
    # -------------------------------------------------
    st.header("2. Initial Data Visualization")
    
    numeric_cols_viz = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols_viz = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.subheader("2.1 Single Feature Distribution")
    selected_feature = st.selectbox("Select Feature to Visualize", numeric_cols_viz + categorical_cols_viz)
    
    if selected_feature:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Distribution of {selected_feature}**")
            fig_hist, ax_hist = plt.subplots()
            if selected_feature in numeric_cols_viz:
                sns.histplot(df[selected_feature].dropna(), kde=True, ax=ax_hist)
            else:
                sns.countplot(y=df[selected_feature], ax=ax_hist, palette="viridis")
            st.pyplot(fig_hist)
        with col2:
            if target_col_raw in df.columns:
                st.write(f"**{selected_feature} vs Target**")
                fig_box, ax_box = plt.subplots()
                if selected_feature in numeric_cols_viz:
                    sns.boxplot(x=target_col_raw, y=selected_feature, data=df, ax=ax_box)
                else:
                    sns.countplot(x=selected_feature, hue=target_col_raw, data=df, ax=ax_box)
                st.pyplot(fig_box)

    st.subheader("2.2 Feature Comparisons")
    if len(numeric_cols_viz) > 1:
        st.write("**Correlation Heatmap**")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols_viz].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)
    
    if target_col_raw in df.columns:
        st.write("**All Features vs Target**")
        all_features = [c for c in df.columns if c != target_col_raw]
        if all_features:
            cols_per_row = 2
            rows = (len(all_features) + cols_per_row - 1) // cols_per_row
            for i in range(rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j
                    if idx < len(all_features):
                        col_name = all_features[idx]
                        with cols[j]:
                            fig, ax = plt.subplots()
                            if col_name in numeric_cols_viz:
                                sns.barplot(x=target_col_raw, y=col_name, data=df, ax=ax, palette="viridis")
                                plt.title(f"Mean {col_name} by Target")
                            else:
                                sns.countplot(x=col_name, hue=target_col_raw, data=df, ax=ax, palette="Set2")
                                plt.title(f"{col_name} by Target")
                                plt.xticks(rotation=45)
                            st.pyplot(fig)

    # -------------------------------------------------
    # 3. DATA SPLIT (RAW) & DOWNLOAD
    # -------------------------------------------------
    st.header("3. Data Split & Test Set Download")
    
    # Split Raw Data first to allow downloading raw test set
    train_df_raw, test_df_raw = train_test_split(df, test_size=0.2, random_state=42)
    
    st.write(f"Data split into Training ({train_df_raw.shape[0]} samples) and Test ({test_df_raw.shape[0]} samples) sets.")
    
    # Convert Test Set to CSV for download
    csv_buffer = io.StringIO()
    test_df_raw.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Test Set (CSV)",
        data=csv_data,
        file_name="test_dataset.csv",
        mime="text/csv"
    )

    # -------------------------------------------------
    # 4. PREPROCESSING & TRAINING
    # -------------------------------------------------
    st.header("4. Preprocessing & Model Training")
    
    st.write("Processing Training Data...")
    
    # 1. Clean & Engineer Features (Train)
    train_df_proc = clean_and_engineer_features(train_df_raw)
    
    # 2. Feature Selection (Correlation based on Train)
    # We do this before encoding to keep it simple, or after? 
    # Let's do it after encoding in prepare_data_for_model or manually here.
    # For simplicity, we'll stick to the previous logic: Drop low correlation features.
    # But we need to do it on the encoded data.
    
    # Prepare Train Data (Encode, Scale)
    X_train_scaled, y_train, final_features, scaler = prepare_data_for_model(train_df_proc, target_col_raw)
    
    # Feature Selection Logic (re-implemented to be robust)
    # We need a temporary DF to calculate correlation
    temp_train_df = pd.DataFrame(X_train_scaled, columns=final_features)
    if y_train is not None:
        temp_train_df[target_col_raw] = y_train.values
        
    corr_matrix = temp_train_df.corr()
    threshold = 0.1
    
    features_to_drop = []
    if target_col_raw in corr_matrix.columns:
        target_corr = corr_matrix[target_col_raw].abs()
        features_to_drop = target_corr[target_corr < threshold].index.tolist()
        if target_col_raw in features_to_drop:
            features_to_drop.remove(target_col_raw)
    
    if features_to_drop:
        st.write(f"Dropping low correlation features: {features_to_drop}")
        # Update final_features list
        final_features = [f for f in final_features if f not in features_to_drop]
        # Update X_train_scaled (need to re-slice, but X_train_scaled is numpy array)
        # It's easier to re-run prepare_data_for_model with reduced columns? 
        # Or just slice the array.
        # Let's slice the array.
        indices_to_keep = [i for i, col in enumerate(temp_train_df.columns) if col in final_features]
        X_train_scaled = X_train_scaled[:, indices_to_keep]
    
    st.success("Training Data Processed & Scaled.")

    # Train Models
    st.write("Training Models...")
    models = {}
    
    # Logistic Regression
    model_lr, _, _ = train_logistic_regression(X_train_scaled, X_train_scaled, y_train, y_train)
    models["Logistic Regression"] = model_lr
    
    # Decision Tree
    model_dt, _, _ = train_decision_tree(X_train_scaled, X_train_scaled, y_train, y_train)
    models["Decision Tree"] = model_dt
    
    # KNN
    model_knn, _, _ = train_knn(X_train_scaled, X_train_scaled, y_train, y_train)
    models["K-Nearest Neighbors"] = model_knn
    
    # Naive Bayes
    model_nb, _, _ = train_naive_bayes(X_train_scaled, X_train_scaled, y_train, y_train)
    models["Naive Bayes (Gaussian)"] = model_nb
    
    # Random Forest
    model_rf, _, _ = train_random_forest(X_train_scaled, X_train_scaled, y_train, y_train)
    models["Random Forest"] = model_rf
    
    # XGBoost
    model_xgb, _, _ = train_xgboost(X_train_scaled, X_train_scaled, y_train, y_train)
    models["XGBoost"] = model_xgb
    
    st.success("All Models Trained Successfully.")

    # -------------------------------------------------
    # 5. EVALUATION (TEST SET)
    # -------------------------------------------------
    st.header("5. Evaluation on Test Data")
    
    # Option to upload separate test file
    test_file_upload = st.file_uploader("Upload a separate Test CSV (Optional)", type=["csv"])
    
    if test_file_upload is not None:
        try:
            df_test_eval = pd.read_csv(test_file_upload)
            # Clean column names
            df_test_eval.columns = df_test_eval.columns.str.strip()
            if 'fitness_category' in df_test_eval.columns and 'is_fit' not in df_test_eval.columns:
                df_test_eval.rename(columns={'fitness_category': 'is_fit'}, inplace=True)
            st.info(f"Using uploaded test file: {test_file_upload.name}")
        except Exception as e:
            st.error(f"Error reading test file: {e}")
            df_test_eval = test_df_raw
    else:
        st.info("Using the default Test split from the dataset.")
        df_test_eval = test_df_raw

    if df_test_eval is not None:
        # Process Test Data using SAME logic as Train
        test_df_proc = clean_and_engineer_features(df_test_eval)
        
        # Prepare Test Data (Align columns to Train, Scale using Train scaler)
        X_test_scaled, y_test, _, _ = prepare_data_for_model(
            test_df_proc, 
            target_col_raw, 
            train_columns=final_features, # Use features AFTER selection
            scaler=scaler
        )
        
        if y_test is None:
            st.warning("Test dataset does not contain the target column. Cannot calculate evaluation metrics.")
        else:
            # Evaluate
            from model.logistic_regression import calculate_metrics
            
            results = []
            for name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None
                
                metrics = calculate_metrics(y_test, y_pred, y_proba)
                metrics["Model"] = name
                results.append(metrics)
            
            # Display Results
            results_df = pd.DataFrame(results)
            cols = ["Model", "Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC"]
            results_df = results_df[cols]
            
            st.subheader("Evaluation Metrics")
            st.dataframe(results_df.style.format("{:.4f}"))
            
            # Visualizations
            st.subheader("Model Performance Comparison")
            metrics_to_plot = ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC"]
            cols_plot = st.columns(2)
            
            for i, metric in enumerate(metrics_to_plot):
                with cols_plot[i % 2]:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(x=metric, y="Model", data=results_df, palette="viridis", ax=ax)
                    plt.title(f"{metric} Comparison")
                    plt.xlim(0, 1.0)
                    st.pyplot(fig)

else:
    st.info("Please upload a dataset or ensure the default dataset is available.")
