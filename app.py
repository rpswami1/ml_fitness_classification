import sys
import subprocess
import importlib.util
import os
import io

# -------------------------------------------------
# 0. AUTO-INSTALL DEPENDENCIES FROM requirements.txt
# -------------------------------------------------
def install_requirements():

    # Upgrade pip first
    try:
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except subprocess.CalledProcessError as e:
        print(f"pip upgrade not needed.")

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
from sklearn.metrics import confusion_matrix, classification_report

# Import models from the 'model' directory
# Ensure the directory is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

try:
    from model.logistic_regression import train_logistic_regression, calculate_metrics
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

# Option to upload dataset
uploaded_file = st.file_uploader("Upload your Fitness Dataset (CSV) for training the models (Default it will download this file if not provided manually)", type=["csv"])

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

    # -------------------------------------------------
    # FUNCTION: Ensure ID column exists in physical CSV
    # -------------------------------------------------
    def ensure_id_column(file_path):
        if os.path.exists(file_path):
            temp_df = pd.read_csv(file_path)
            temp_df.columns = temp_df.columns.str.strip()

            if "id" not in temp_df.columns:
                temp_df.insert(0, "id", range(1, len(temp_df) + 1))
                temp_df.to_csv(file_path, index=False)

            return temp_df
        return None

    # -------------------------------------------------
    # CASE 1: Local user dataset exists
    # -------------------------------------------------
    if os.path.exists(user_provided_csv):

        df = ensure_id_column(user_provided_csv)
        st.success(f"Found local dataset: {os.path.basename(user_provided_csv)}")

    # -------------------------------------------------
    # CASE 2: Default CSV already exists
    # -------------------------------------------------
    elif os.path.exists(csv_file):

        df = ensure_id_column(csv_file)

    # -------------------------------------------------
    # CASE 3: Need to download from Kaggle
    # -------------------------------------------------
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
                st.info("Please ensure Kaggle API is configured.")

        # Extract ZIP
        if os.path.exists(MAIN_ZIP_PATH):
            with zipfile.ZipFile(MAIN_ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(CODE_DIR)

        # After extraction update CSV with ID
        if os.path.exists(csv_file):
            df = ensure_id_column(csv_file)

if df is not None:
    # Clean column names to avoid whitespace issues
    df.columns = df.columns.str.strip()
    
    # Rename fitness_category to is_fit if present
    if 'fitness_category' in df.columns and 'is_fit' not in df.columns:
        df.rename(columns={'fitness_category': 'is_fit'}, inplace=True)
        st.info("Renamed 'fitness_category' column to 'is_fit'.")
    
    st.write(f"Dataset loaded. Shape: {df.shape}")

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10))

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
    categorical_cols_viz = df_viz.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2.1 Single Feature Visualization
    st.subheader("2.1 Single Feature Distribution")
    selected_feature = st.selectbox("Select Feature to Visualize", numeric_cols_viz + categorical_cols_viz)
    
    if selected_feature:
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Distribution of {selected_feature}**")
            fig_hist, ax_hist = plt.subplots()
            if selected_feature in numeric_cols_viz:
                sns.histplot(df_viz[selected_feature].dropna(), kde=True, ax=ax_hist)
            else:
                sns.countplot(y=df_viz[selected_feature], ax=ax_hist, palette="viridis")
            st.pyplot(fig_hist)

        with col2:
            if target_col_raw in df_viz.columns:
                st.write(f"**{selected_feature} vs Target ({target_col_raw})**")
                fig_box, ax_box = plt.subplots()
                if selected_feature in numeric_cols_viz:
                    sns.boxplot(x=target_col_raw, y=selected_feature, data=df_viz, ax=ax_box)
                else:
                    sns.countplot(x=selected_feature, hue=target_col_raw, data=df_viz, ax=ax_box)
                st.pyplot(fig_box)

    # 2.2 Feature Comparisons (Bar Charts & Heatmaps)
    st.subheader("2.2 Feature Comparisons & Relationships")

    # Correlation Heatmap (Numerical)
    st.write("**Correlation Heatmap (Numerical Features)**")
    if len(numeric_cols_viz) > 1:
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_viz[numeric_cols_viz].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)
    
    # Bar Charts for All Features vs Target
    if target_col_raw in df_viz.columns:
        st.write("**All Features vs Target (Bar Charts)**")

        # Get all features excluding target
        all_features = [c for c in df_viz.columns if c != target_col_raw]

        if all_features:
            # Create a grid layout
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
                                # Bar chart of mean value vs target
                                sns.barplot(x=target_col_raw, y=col_name, data=df_viz, ax=ax, palette="viridis")
                                plt.title(f"Mean {col_name} by Target")
                            else:
                                # Count plot for categorical
                                sns.countplot(x=col_name, hue=target_col_raw, data=df_viz, ax=ax, palette="Set2")
                                plt.title(f"{col_name} Distribution by Target")
                                plt.xticks(rotation=45)
                            st.pyplot(fig)

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
        if "label_encoder" not in st.session_state:
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col])
            st.session_state.label_encoder = le
        else:
            le = st.session_state.label_encoder
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
# 6. MODEL IMPLEMENTATION & PERSISTENT TRAINING
# -------------------------------------------------
st.header("6. Model Implementation & Evaluation")

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "baseline_metrics" not in st.session_state:
    st.session_state.baseline_metrics = {}

if "le" not in st.session_state:
    st.session_state.le = None

if "target_col" not in st.session_state:
    st.session_state.target_col = target_col

# -------------------------------------------------
# TRAIN ONLY ONCE
# -------------------------------------------------
if not st.session_state.trained_models:

    st.write("Training models (this runs only once)...")

    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Encode target safely
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        st.session_state.le = le
    else:
        st.session_state.le = None

    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Persist objects
    st.session_state.scaler = scaler
    st.session_state.feature_columns = X.columns
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.df = df
    st.session_state.target_col = target_col

    results = []

    model_functions = {
        "Logistic Regression": train_logistic_regression,
        "Decision Tree": train_decision_tree,
        "K-Nearest Neighbors": train_knn,
        "Naive Bayes (Gaussian)": train_naive_bayes,
        "Random Forest": train_random_forest,
        "XGBoost": train_xgboost
    }

    for model_name, train_func in model_functions.items():

        model, train_m, test_m = train_func(X_train, X_test, y_train, y_test)

        st.session_state.trained_models[model_name] = model

        st.session_state.baseline_metrics[model_name] = {
            "Accuracy": train_m["Accuracy"],
            "Precision": train_m["Precision"],
            "Recall": train_m["Recall"],
            "F1 Score": train_m["F1 Score"],
            "AUC Score": train_m["AUC Score"],
            "MCC": train_m["MCC"]
        }

        train_m["Model"] = model_name
        train_m["Set"] = "Train"
        test_m["Model"] = model_name
        test_m["Set"] = "Test"

        results.append(train_m)
        results.append(test_m)

    results_df = pd.DataFrame(results)

    st.success("Models trained and stored successfully.")

else:
    st.success("Models already trained — using persisted models.")

# -------------------------------------------------
# DISPLAY TRAINING BASELINE METRICS
# -------------------------------------------------
results_display = pd.DataFrame([
    {
        "Model": name,
        **st.session_state.baseline_metrics[name]
    }
    for name in st.session_state.trained_models.keys()
])

st.subheader("Training Baseline Metrics")

numeric_cols = results_display.select_dtypes(include=["number"]).columns

st.dataframe(
    results_display.style.format(
        {col: "{:.4f}" for col in numeric_cols}
    ),
    use_container_width=True
)

# -------------------------------------------------
# 7. TESTING & ANALYTICS DASHBOARD
# -------------------------------------------------
st.header("7. Testing & Analytics Dashboard")

if not st.session_state.trained_models:
    st.warning("No trained models found.")
    st.stop()

scaler = st.session_state.scaler
feature_columns = st.session_state.feature_columns
le = st.session_state.le
target_col = st.session_state.target_col
baseline_metrics = st.session_state.baseline_metrics
train_df = st.session_state.df

# Download template
template_csv = train_df.sample(min(20, len(train_df))).to_csv(index=False)
st.download_button("📥 Download Test CSV Template",
                   template_csv,
                   "test_template.csv",
                   "text/csv")

# Model selection
selected_models = st.multiselect(
    "Select models to evaluate:",
    list(st.session_state.trained_models.keys()),
    default=list(st.session_state.trained_models.keys())
)

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file and selected_models:

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    # Align features
    missing_cols = set(feature_columns) - set(test_df.columns)
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()

    # -------------------------------------------------
    # TEST DATA PREPROCESSING & EDA
    # -------------------------------------------------
    st.subheader("Test Data Analysis (EDA)")

    # Preprocess Test Data (Cleaning & Engineering)
    # We need to apply the same cleaning steps as training data
    # Re-using the logic from Section 3 & 4 but applied to test_df

    # 1. Standardize 'smokes'
    if 'smokes' in test_df.columns:
        test_df['smokes'] = test_df['smokes'].astype(str).str.lower().str.strip()
        smokes_map = {'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
        test_df['smokes'] = test_df['smokes'].map(smokes_map)
        if test_df['smokes'].isnull().any():
             test_df['smokes'] = test_df['smokes'].fillna(test_df['smokes'].mode()[0])

    # 2. Impute 'sleep_hours'
    if 'sleep_hours' in test_df.columns:
        test_df['sleep_hours'] = test_df['sleep_hours'].fillna(test_df['sleep_hours'].median())

    # 3. Handle Outliers in 'weight_kg' (Capping)
    weight_col = 'weight_kg' if 'weight_kg' in test_df.columns else ('weight' if 'weight' in test_df.columns else None)
    if weight_col:
        test_df[weight_col] = pd.to_numeric(test_df[weight_col], errors='coerce')
        Q1 = test_df[weight_col].quantile(0.25)
        Q3 = test_df[weight_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        test_df[weight_col] = np.clip(test_df[weight_col], lower_bound, upper_bound)

    # 4. General Numeric Cleaning
    numeric_keywords = ['weight', 'height', 'age', 'bmi', 'score', 'income', 'rate', 'duration']
    potential_numeric_cols = [c for c in test_df.columns if any(k in c.lower() for k in numeric_keywords)]
    for col in potential_numeric_cols:
        if col != 'smokes':
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    # 5. General Imputation
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns
    test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].median())

    object_cols = test_df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if not test_df[col].mode().empty:
            test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

    # 6. Feature Engineering: BMI
    w_col = 'weight_kg' if 'weight_kg' in test_df.columns else ('weight' if 'weight' in test_df.columns else None)
    h_col = 'height_m' if 'height_m' in test_df.columns else ('height' if 'height' in test_df.columns else None)

    if w_col and h_col:
        height_mean = test_df[h_col].mean()
        if height_mean > 3: # Assume cm
            test_df['BMI'] = test_df[w_col] / ((test_df[h_col] / 100) ** 2)
        else: # Assume m
            test_df['BMI'] = test_df[w_col] / (test_df[h_col] ** 2)

    # Visualizations for Test Data
    test_numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()

    # Heatmap
    if len(test_numeric_cols) > 1:
        st.write("**Test Data Correlation Heatmap**")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(test_df[test_numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

    # Bar Charts vs Target (if target exists)
    if target_col in test_df.columns:
        st.write("**Test Data Features vs Target**")
        all_features = [c for c in test_df.columns if c != target_col]
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
                            if col_name in test_numeric_cols:
                                sns.barplot(x=target_col, y=col_name, data=test_df, ax=ax, palette="viridis")
                                plt.title(f"Mean {col_name} by Target (Test)")
                            else:
                                sns.countplot(x=col_name, hue=target_col, data=test_df, ax=ax, palette="Set2")
                                plt.title(f"{col_name} Distribution by Target (Test)")
                                plt.xticks(rotation=45)
                            st.pyplot(fig)

    # Prepare for Prediction
    # One-Hot Encoding
    categorical_cols = test_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    test_df_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

    X_test = test_df_encoded.drop(columns=[target_col]) if target_col in test_df_encoded.columns else test_df_encoded

    # Align columns with training data
    for col in feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_columns] # Ensure order and selection matches training

    X_test_scaled = scaler.transform(X_test)

    if target_col in test_df.columns:
        if le is not None:
            # Handle potential unseen labels in target
            try:
                y_true = le.transform(test_df[target_col])
            except:
                 # Fallback: if test data has labels not in training, we can't evaluate properly
                 # For binary classification 0/1 or 'Fit'/'Unfit', this is less likely if cleaned
                 # Re-fitting a temporary encoder just to get numbers for metrics (assuming same classes)
                 le_temp = LabelEncoder()
                 y_true = le_temp.fit_transform(test_df[target_col])
        else:
             y_true = test_df[target_col]
    else:
        st.error("Target column missing in test data.")
        st.stop()

    from sklearn.metrics import *
    import plotly.express as px
    import plotly.graph_objects as go

    results = []

    st.subheader("Detailed Model Evaluation")

    for name in selected_models:

        model = st.session_state.trained_models[name]

        y_pred = model.predict(X_test_scaled)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = y_pred

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "AUC Score": roc_auc_score(y_true, y_prob),
            "MCC": matthews_corrcoef(y_true, y_pred)
        }

        results.append(metrics)

        # Confusion Matrix & Report per Model
        with st.expander(f"Detailed Report: {name}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_true, y_pred)

                # Extract TN, FP, FN, TP
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    st.write(f"True Negatives: {tn}")
                    st.write(f"False Positives: {fp}")
                    st.write(f"False Negatives: {fn}")
                    st.write(f"True Positives: {tp}")

                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig_cm)

            with col2:
                st.write("**Classification Report**")
                report = classification_report(y_true, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

    test_results_df = pd.DataFrame(results)

    st.subheader("Test Performance Comparison")

    numeric_cols = test_results_df.select_dtypes(include=["number"]).columns
    st.dataframe(
        test_results_df.style.format(
            {col: "{:.4f}" for col in numeric_cols}
        ),
        use_container_width=True
    )

    # ----------------------------
    # BAR CHART
    # ----------------------------
    fig_bar = px.bar(test_results_df, x="Model", y="F1 Score", color="Model")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ----------------------------
    # LINE CHART
    # ----------------------------
    fig_line = px.line(test_results_df.set_index("Model"))
    st.plotly_chart(fig_line, use_container_width=True)

    # ----------------------------
    # SCATTER PRECISION vs RECALL
    # ----------------------------
    fig_scatter = px.scatter(test_results_df,
                             x="Recall",
                             y="Precision",
                             size="F1 Score",
                             color="Model")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ----------------------------
    # ROC CURVE
    # ----------------------------
    fig_roc = go.Figure()

    for name in selected_models:

        model = st.session_state.trained_models[name]

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                     mode='lines',
                                     name=name))

    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                 line=dict(dash='dash'),
                                 name="Baseline"))

    fig_roc.update_layout(title="ROC Curve Comparison")
    st.plotly_chart(fig_roc, use_container_width=True)

    # ----------------------------
    # MODEL DRIFT ANALYSIS
    # ----------------------------
    st.subheader("📉 Model Drift Analysis")

    drift_results = []

    for name in selected_models:

        baseline_f1 = baseline_metrics[name]["F1 Score"]
        current_f1 = test_results_df.loc[test_results_df["Model"] == name, "F1 Score"].values[0]

        drift_percent = ((current_f1 - baseline_f1) / baseline_f1) * 100 if baseline_f1 != 0 else 0

        drift_results.append({
            "Model": name,
            "Baseline F1": baseline_f1,
            "Current F1": current_f1,
            "Drift %": drift_percent
        })

    drift_df = pd.DataFrame(drift_results)

    st.dataframe(
        drift_df.style.format(
            {"Baseline F1": "{:.4f}",
             "Current F1": "{:.4f}",
             "Drift %": "{:.2f}"}
        ),
        use_container_width=True
    )

    fig_drift = px.line(drift_df, x="Model",
                        y=["Baseline F1", "Current F1"],
                        markers=True)

    st.plotly_chart(fig_drift, use_container_width=True)

    # ----------------------------
    # FINAL CONCLUSIONS (Train vs Test)
    # ----------------------------
    st.header("8. Final Conclusions: Train vs Test Analysis")

    # Combine Train (Baseline) and Test metrics for comparison
    comparison_data = []

    for name in selected_models:
        # Get Train metrics
        train_m = baseline_metrics[name]
        for metric, value in train_m.items():
            comparison_data.append({
                "Model": name,
                "Set": "Train",
                "Metric": metric,
                "Value": value
            })

        # Get Test metrics
        test_row = test_results_df[test_results_df["Model"] == name].iloc[0]
        for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score", "MCC"]:
             comparison_data.append({
                "Model": name,
                "Set": "Test",
                "Metric": metric,
                "Value": test_row[metric]
            })

    comp_df = pd.DataFrame(comparison_data)

    st.write("### Comparative Analysis Graphs")

    # Faceted plot for all metrics
    fig_comp = px.bar(comp_df, x="Model", y="Value", color="Set", barmode="group",
                      facet_col="Metric", facet_col_wrap=2,
                      title="Train vs Test Performance across Metrics")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.write("""
    ### Conclusion
    
    *   **Overfitting Check**: If the 'Train' bars are significantly higher than the 'Test' bars (especially for Accuracy and F1 Score), the model might be overfitting.
    *   **Generalization**: Models where Train and Test scores are close demonstrate good generalization to new data.
    *   **Best Performer**: Look for the model with the highest consistent scores across both sets, particularly in F1 Score and MCC for imbalanced datasets.
    """)

else:
    st.info("Please upload a dataset or ensure the default dataset is available.")
