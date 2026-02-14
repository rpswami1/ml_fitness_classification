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

# Import models from the 'model' directory
# Ensure the directory is in the path
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

# Check if the CSV already exists directly in the folder (from previous manual download or extraction)
csv_file = os.path.join(CODE_DIR, "fitness_class_data.csv")
# Also check for the file provided by the user: fitness_dataset.csv
user_provided_csv = os.path.join(CODE_DIR, "fitness_dataset.csv")

if os.path.exists(user_provided_csv):
    csv_file = user_provided_csv
    st.success(f"Found local dataset: {os.path.basename(csv_file)}")
elif not os.path.exists(csv_file):
    # Only try to download if neither file exists
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
    
    # Clean column names to avoid whitespace issues
    df.columns = df.columns.str.strip()
    
    # Rename fitness_category to is_fit if present
    if 'fitness_category' in df.columns and 'is_fit' not in df.columns:
        df.rename(columns={'fitness_category': 'is_fit'}, inplace=True)
        st.info("Renamed 'fitness_category' column to 'is_fit'.")
    
    st.write(f"Dataset loaded. Shape: {df.shape}")
    
    with st.expander("Raw Data Preview"):
        st.dataframe(df.head())

    # -------------------------------------------------
    # 2. INITIAL DATA VISUALIZATION (Before Preprocessing)
    # -------------------------------------------------
    st.header("2. Initial Data Visualization")
    st.write("Visualizing raw features before any cleaning or engineering.")
    
    target_col_raw = 'is_fit' 
    
    # Check if target column exists
    if target_col_raw not in df.columns:
        st.error(f"Target column '{target_col_raw}' not found in dataset. Available columns: {df.columns.tolist()}")
        # Try to find a likely candidate
        likely_targets = [c for c in df.columns if 'fitness' in c.lower() or 'category' in c.lower() or 'class' in c.lower()]
        if likely_targets:
            st.info(f"Did you mean one of these? {likely_targets}")
            target_col_raw = st.selectbox("Select Target Column", likely_targets)
    
    # Select numeric columns for visualization (attempting to parse numeric even if object for viz)
    # We create a temporary copy for visualization to not affect the main df yet
    df_viz = df.copy()
    
    # Try to convert potential numeric columns just for visualization purposes
    numeric_keywords = ['weight', 'height', 'age', 'bmi', 'score', 'income', 'rate', 'duration']
    potential_numeric_cols = [c for c in df_viz.columns if any(k in c.lower() for k in numeric_keywords)]
    for col in potential_numeric_cols:
        df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce')
        
    numeric_cols_viz = df_viz.select_dtypes(include=[np.number]).columns.tolist()
    
    selected_feature = st.selectbox("Select Feature to Visualize (Raw Data)", numeric_cols_viz)
    
    if selected_feature:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Distribution of {selected_feature}")
            fig_hist, ax_hist = plt.subplots()
            # Drop NA for plotting
            sns.histplot(df_viz[selected_feature].dropna(), kde=True, ax=ax_hist)
            st.pyplot(fig_hist)
            
        with col2:
            st.subheader(f"{selected_feature} vs Target")
            fig_box, ax_box = plt.subplots()
            if target_col_raw in df_viz.columns:
                sns.boxplot(x=target_col_raw, y=selected_feature, data=df_viz, ax=ax_box)
                st.pyplot(fig_box)
            else:
                st.warning(f"Target column '{target_col_raw}' not found.")

    # -------------------------------------------------
    # 3. DATA PREPROCESSING & CLEANING
    # -------------------------------------------------
    st.header("3. Preprocessing & Cleaning")
    
    st.subheader("3.1 Handling Specific Data Quality Issues")
    
    # 1. Handle Mixed Data Types in 'smokes'
    if 'smokes' in df.columns:
        st.write("Standardizing 'smokes' column (mixed types detected)...")
        # Convert to string first to handle mixed types, then map
        df['smokes'] = df['smokes'].astype(str).str.lower().str.strip()
        # Map yes/no/1/0
        smokes_map = {'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
        df['smokes'] = df['smokes'].map(smokes_map)
        # Fill any NaNs (unexpected values) with mode
        if df['smokes'].isnull().any():
             df['smokes'] = df['smokes'].fillna(df['smokes'].mode()[0])
        st.write("✅ 'smokes' column standardized to binary (0/1).")
        
    # 2. Handle Missing Values in 'sleep_hours'
    if 'sleep_hours' in df.columns:
        missing_sleep = df['sleep_hours'].isnull().sum()
        if missing_sleep > 0:
            st.write(f"Imputing {missing_sleep} missing values in 'sleep_hours'...")
            df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].median())
            st.write("✅ 'sleep_hours' imputed with median.")
            
    # 3. Outlier Detection in 'weight_kg'
    # Check for weight_kg or weight
    weight_col = 'weight_kg' if 'weight_kg' in df.columns else ('weight' if 'weight' in df.columns else None)
    
    if weight_col:
        st.write(f"Handling outliers in '{weight_col}' (IQR Method)...")
        # Ensure numeric
        df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
        
        Q1 = df[weight_col].quantile(0.25)
        Q3 = df[weight_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[weight_col] < lower_bound) | (df[weight_col] > upper_bound)).sum()
        if outliers > 0:
            # Cap/Clip outliers instead of removing to preserve data size if small
            df[weight_col] = np.clip(df[weight_col], lower_bound, upper_bound)
            st.write(f"✅ Capped {outliers} outliers in '{weight_col}' to IQR bounds.")
        else:
            st.write(f"No outliers detected in '{weight_col}'.")

    st.subheader("3.2 General Cleaning")
    
    # 3.1 Handling Corrupted Data (General)
    # Now we apply changes to the main 'df'
    for col in potential_numeric_cols:
        # Skip columns we already handled specifically if needed, but safe to re-run
        if col != 'smokes': 
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3.2 Handling Missing Values (General)
    if df.isnull().values.any():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        st.success("Imputed remaining missing values.")

    # -------------------------------------------------
    # 4. FEATURE ENGINEERING
    # -------------------------------------------------
    st.header("4. Feature Engineering")
    
    # 4.1 BMI Calculation
    # Check for weight_kg/weight and height/height_m
    w_col = 'weight_kg' if 'weight_kg' in df.columns else ('weight' if 'weight' in df.columns else None)
    h_col = 'height_m' if 'height_m' in df.columns else ('height' if 'height' in df.columns else None)

    if w_col and h_col:
        # Check if height is likely in cm (> 3) or m (< 3)
        height_mean = df[h_col].mean()
        if height_mean > 3:
            # Assume cm, convert to m
            df['BMI'] = df[w_col] / ((df[h_col] / 100) ** 2)
        else:
            # Assume m
            df['BMI'] = df[w_col] / (df[h_col] ** 2)
        st.write(f"✅ Created feature: `BMI` from `{w_col}` and `{h_col}`")

    # 4.2 One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    target_col = target_col_raw # Use the identified target column
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        st.write(f"✅ Label Encoded target: `{target_col}`")

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    st.write("✅ Performed One-Hot Encoding.")

    # -------------------------------------------------
    # 5. FEATURE SELECTION
    # -------------------------------------------------
    st.header("5. Feature Selection")
    
    st.subheader("Correlation Heatmap")
    fig_corr = plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    st.pyplot(fig_corr)

    # Automatic Feature Selection based on Correlation Threshold
    threshold = 0.1
    st.write(f"Applying Correlation Threshold: {threshold}")
    
    # Check if target_col is still in correlation_matrix
    if target_col in correlation_matrix.columns:
        target_corr = correlation_matrix[target_col].abs()
        unnecessary_features = target_corr[target_corr < threshold].index.tolist()
        
        # SAFETY CHECK: Ensure target column is NOT dropped
        # We use list comprehension to be absolutely sure we filter it out
        unnecessary_features = [f for f in unnecessary_features if f != target_col]
        
        if unnecessary_features:
            st.write(f"Dropping low-impact features (< {threshold} correlation): {unnecessary_features}")
            df.drop(columns=unnecessary_features, inplace=True)
            st.success(f"Dropped {len(unnecessary_features)} features.")
        else:
            st.info("No features dropped (all meet the correlation threshold).")
    else:
        st.warning(f"Target column '{target_col}' not found in correlation matrix. Skipping feature selection based on target correlation.")

    st.write(f"Final Dataset Shape: {df.shape}")

    # -------------------------------------------------
    # 6. MODEL IMPLEMENTATION
    # -------------------------------------------------
    st.header("6. Model Implementation & Evaluation")

    # Splitting
    # Ensure target_col is in df before dropping
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        if st.button("Train All Models"):
            st.write("Training models... please wait.")
            results = []
            
            # Train Logistic Regression
            _, metrics = train_logistic_regression(X_train, X_test, y_train, y_test)
            metrics["Model"] = "Logistic Regression"
            results.append(metrics)
            
            # Train Decision Tree
            _, metrics = train_decision_tree(X_train, X_test, y_train, y_test)
            metrics["Model"] = "Decision Tree"
            results.append(metrics)
            
            # Train KNN
            _, metrics = train_knn(X_train, X_test, y_train, y_test)
            metrics["Model"] = "K-Nearest Neighbors"
            results.append(metrics)
            
            # Train Naive Bayes
            _, metrics = train_naive_bayes(X_train, X_test, y_train, y_test)
            metrics["Model"] = "Naive Bayes (Gaussian)"
            results.append(metrics)
            
            # Train Random Forest
            _, metrics = train_random_forest(X_train, X_test, y_train, y_test)
            metrics["Model"] = "Random Forest"
            results.append(metrics)
            
            # Train XGBoost
            _, metrics = train_xgboost(X_train, X_test, y_train, y_test)
            metrics["Model"] = "XGBoost"
            results.append(metrics)

            # Display Results Table
            results_df = pd.DataFrame(results)
            # Reorder columns
            cols = ["Model", "Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC"]
            results_df = results_df[cols]
            
            st.subheader("Model Comparison Table")
            st.dataframe(results_df.style.format({
                "Accuracy": "{:.4f}",
                "AUC Score": "{:.4f}", 
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "MCC": "{:.4f}"
            }))

            # Visualization of Metrics
            st.subheader("Deep Comparison of Models")
            
            metrics_to_plot = ["Accuracy", "AUC Score", "F1 Score", "MCC"]
            
            for metric in metrics_to_plot:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x=metric, y="Model", data=results_df, palette="viridis", ax=ax)
                plt.title(f"{metric} Comparison")
                plt.xlim(0, 1.0)
                st.pyplot(fig)

    else:
        st.error(f"Target column '{target_col}' not found in the final dataset. It might have been dropped during processing.")

else:
    st.error("CSV file not found. Please check the download process.")
