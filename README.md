# Real-World Style Fitness Classification Dataset (Synthetic)

## Dataset Description

This synthetic dataset simulates a real-world binary classification problem where the goal is to predict whether a person is fit (is_fit = 1) or not fit (is_fit = 0) based on various health and lifestyle features.

The dataset contains 2000 samples with a mixture of numerical and categorical features, some of which include noisy, inconsistent, or missing values to reflect real-life data challenges. This design enables users, especially beginners, to practice data preprocessing, feature engineering, and building classification models such as neural networks.

Features have both linear and non-linear relationships with the target variable. Some features have complex interactions and the target is generated using a sigmoid-like function with added noise, making it a challenging but realistic task. The dataset also includes mixed data types (e.g., the "smokes" column contains both numeric and string values) and some outliers are present.

This dataset is ideal for users wanting to improve skills in cleaning messy data, encoding categorical variables, handling missing values, detecting outliers, and training classification models including neural networks.

## Column Descriptions

| Column Name | Description |
|-------------|-------------|
| age | Age of the individual in years (integer) |
| height_cm | Height in centimeters (integer) |
| weight_kg | Weight in kilograms (integer, contains some outliers) |
| heart_rate | Resting heart rate in beats per minute (float) |
| blood_pressure | Systolic blood pressure in mmHg (float) |
| sleep_hours | Average hours of sleep per day (float, may contain NaNs) |
| nutrition_quality | Daily nutrition quality score between 0 and 10 (float) |
| activity_index | Physical activity level score between 1 and 5 (float) |
| smokes | Smoking status (mixed types: 0, 1, "yes", "no") |
| gender | Gender of individual, either 'M' or 'F' |
| is_fit | Target variable: 1 if the person is fit, 0 otherwise |

## Dataset Statistics

- **Total samples**: 2000
- **Features**: 10 (9 predictive features + 1 target)
- **Target distribution**: Approximately 60% not fit (0), 40% fit (1)
- **Missing values**: ~8% missing values in sleep_hours column
- **Data types**: Mixed (integers, floats, strings)
- **Outliers**: Present in weight_kg column (~2% of samples)

## Data Quality Issues (Intentional)

This dataset intentionally includes several data quality issues to simulate real-world scenarios:

1. **Mixed data types**: The 'smokes' column contains both numeric (0, 1) and string ("yes", "no") values
2. **Missing values**: The 'sleep_hours' column has approximately 8% missing values
3. **Outliers**: The 'weight_kg' column contains some extreme values (very low or very high weights)
4. **Noise**: All features contain some level of noise to make the classification task more realistic

## Suggested Data Preprocessing Steps

1. **Handle mixed data types**: Convert the 'smokes' column to a consistent format
2. **Deal with missing values**: Impute or remove missing values in 'sleep_hours'
3. **Outlier detection**: Identify and handle outliers in 'weight_kg'
4. **Feature engineering**: Consider creating BMI from height and weight
5. **Encoding**: One-hot encode categorical variables like 'gender'
6. **Scaling**: Normalize or standardize numerical features for neural networks

## Potential Use Cases

- **Binary classification**: Predict fitness status
- **Data preprocessing practice**: Clean and prepare messy data
- **Feature engineering**: Create new meaningful features
- **Model comparison**: Compare different classification algorithms
- **Neural network training**: Practice building and tuning neural networks
- **Exploratory data analysis**: Understand relationships between health metrics

## Model Performance Expectations

Due to the synthetic nature and intentional noise, expect:
- **Baseline accuracy**: ~60% (majority class)
- **Good models**: 75-85% accuracy
- **Excellent models**: 85-90% accuracy

The dataset is designed to be challenging but achievable, making it perfect for learning and experimentation.

## License

This dataset is provided under the **CC0 Public Domain** license, making it suitable for educational and research purposes without restrictions.

## Acknowledgments

This is a synthetic dataset created for educational purposes. It does not contain real personal health information and is designed to help users practice data science skills in a safe, privacy-compliant environment.

