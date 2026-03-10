import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# 1. Load the dataset
# Make sure the filename matches exactly what is in your folder
df = pd.read_csv('online_shoppers_intention.csv')

# Part 1 a) Display the first and last 12 rows 
print("--- First 12 Rows ---")
print(df.head(12))

print("\n--- Last 12 Rows ---")
print(df.tail(12))
# b) Identify and print the total number of rows and columns 
print("\n--- 1b) Total Rows and Columns ---")
rows, cols = df.shape
print(f"Total rows: {rows}")
print(f"Total columns: {cols}")

# c) List all column names and their corresponding data types 
print("\n--- 1c) Column Names and Data Types ---")
print(df.dtypes)

# d) Generate a full dataset summary (non-null counts and data types) 
print("\n--- 1d) Full Dataset Summary ---")
df.info() # Note: info() automatically prints to the console

# e) Identify the target variable and analyze its class distribution 
print("\n--- 1e) Target Variable Class Distribution ('Revenue') ---")
print(df['Revenue'].value_counts(dropna=False))
print("\nPercentage distribution:")
print(df['Revenue'].value_counts(normalize=True) * 100)

# f) Select one categorical feature and display distinct/most frequent values 
print("\n--- 1f) Categorical Feature Analysis ('VisitorType') ---")
print("Distinct values:")
print(df['VisitorType'].unique())
print(f"Most frequent value: {df['VisitorType'].mode()[0]}")

# g) Compute mean, median, standard dev, and percentiles (20%, 50%, 75%) 
print("\n--- 1g) Numerical Features Statistics ---")
# The describe method calculates mean, std, and we can specify exact percentiles (50% is the median)
stats = df.describe(percentiles=[.20, .50, .75])
print(stats)

# h) Detect missing values and report their counts per column 
print("\n--- 1h) Missing Values per Column ---")
print(df.isnull().sum())

# i) Identify duplicate records and remove them if found 
print("\n--- 1i) Duplicate Records ---")
duplicates_count = df.duplicated().sum()
print(f"Number of duplicate records found: {duplicates_count}")

if duplicates_count > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. New dataset shape: {df.shape}")
else:
    print("No duplicates to remove.")

# Part 2: Data Preparation
print("\n=== Part 2: Data Preparation ===")

# 2a) Meaningful filtering condition
# We filter out sessions where the user visited 0 product-related pages.
# Justification: If a user doesn't look at a single product, their purchasing intention is fundamentally zero. 
# Removing these "instant bounce" or purely administrative visits helps our future models focus on actual shopper behavior.
initial_shape = df.shape
df_filtered = df[df['ProductRelated'] > 0].copy()
print("\n--- 2a) Filtering ---")
print(f"Removed {initial_shape[0] - df_filtered.shape[0]} rows where ProductRelated was 0.")
print(f"New dataset shape: {df_filtered.shape}")

# 2b) Encode categorical variables using an appropriate technique
print("\n--- 2b) Encoding Categorical Variables ---")
# We use Label Encoding for binary (True/False) targets so they become 1 and 0.
le = LabelEncoder()
df_filtered['Weekend'] = le.fit_transform(df_filtered['Weekend'])
df_filtered['Revenue'] = le.fit_transform(df_filtered['Revenue'])

# We use One-Hot Encoding (pd.get_dummies) for multi-class categoricals to prevent the model 
# from assuming false mathematical hierarchies (e.g., assuming Month 12 is "greater" than Month 1).
# We drop the first column to avoid the dummy variable trap (perfect multicollinearity).
categorical_cols = ['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
df_encoded = pd.get_dummies(df_filtered, columns=categorical_cols, drop_first=True)
print(f"Data shape after One-Hot Encoding: {df_encoded.shape}")

# 2c) Normalize numerical features using StandardScaler
print("\n--- 2c) Normalization ---")
numerical_cols = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
                  'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

scaler = StandardScaler()
# StandardScaler shifts the distribution to have a mean of 0 and a standard deviation of 1.
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
print("StandardScaler successfully applied to numerical features.")

# 2d) Divide one numerical feature into 5 equal-width bins and report bin distribution
print("\n--- 2d) Bin Distribution ---")
# Using pd.cut to create 5 equal-width bins on the scaled 'ProductRelated' feature.
df_encoded['ProductRelated_Binned'] = pd.cut(df_encoded['ProductRelated'], bins=5)
print("Bin distribution for ProductRelated (Scaled):")
print(df_encoded['ProductRelated_Binned'].value_counts().sort_index())

# 2e) Identify and handle missing values
print("\n--- 2e) Missing Values Handling ---")
missing_values_after_preparation = df_filtered.isnull().sum()
print("Missing values per column after filtering:")
print(missing_values_after_preparation)
print(
    "\nMedian would be used for numerical columns and mode for categorical columns "
    "because median is robust to outliers while mode preserves the most common category."
)
print("This dataset has no missing values, so no imputation step was required.")

# 2f) Use correlation analysis to detect numerical features that carry nearly the same information.
# When two numerical columns are very strongly correlated, keeping both can add redundancy without improving the analysis.
print("\n--- 2f) Correlation Analysis (Numerical Features) ---")
numerical_cols_for_selection = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
correlation_matrix = df[numerical_cols_for_selection].corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

correlation_threshold = 0.85
high_correlation_pairs = []
removed_numerical_features = []

for column in upper_triangle.columns:
    for row_name, corr_value in upper_triangle[column].dropna().items():
        if corr_value > correlation_threshold:
            high_correlation_pairs.append((row_name, column, corr_value))
            removed_numerical_features.append(column)

if high_correlation_pairs:
    print(f"Highly correlated pairs found (threshold > {correlation_threshold}):")
    for feature_a, feature_b, corr_value in high_correlation_pairs:
        print(f"- {feature_a} vs {feature_b}: {corr_value:.3f}")
else:
    print("No highly correlated numerical features found.")

removed_numerical_features = sorted(set(removed_numerical_features))

# Still in 2f, use the chi-square test to measure whether each categorical feature has a meaningful relationship with the target.
# Features with weak statistical association can be removed because they contribute little predictive value.
print("\n--- 2f) Chi-square Test (Categorical Features) ---")
categorical_features = ['Month', 'VisitorType', 'Weekend']
chi_square_results = []
removed_categorical_features = []

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(df['Revenue'])

for column in categorical_features:
    feature_encoder = LabelEncoder()
    encoded_feature = feature_encoder.fit_transform(df[column]).reshape(-1, 1)
    chi_square_stat, p_value = chi2(encoded_feature, y_encoded)
    chi_square_stat = float(chi_square_stat[0])
    p_value = float(p_value[0])

    chi_square_results.append((column, chi_square_stat, p_value))

    if p_value > 0.05:
        removed_categorical_features.append(column)

for feature, statistic, p_value in chi_square_results:
    print(f"- {feature}: chi2 = {statistic:.3f}, p-value = {p_value:.6f}")

selected_features = [
    column for column in df.columns
    if column not in removed_numerical_features and column not in removed_categorical_features
]

# 2g) Report exactly which features were removed and include the statistical evidence behind each removal.
# This keeps the feature-selection result explicit and avoids hiding the important answer behind generic messages.
print("\n--- 2g) Removed Features and Justification ---")
if removed_numerical_features:
    print("Removed numerical features:")
    for removed_feature in removed_numerical_features:
        matching_pair = next(
            pair for pair in high_correlation_pairs if pair[1] == removed_feature
        )
        kept_feature, _, corr_value = matching_pair
        print(
            f"- {removed_feature}: removed because it is highly correlated with "
            f"{kept_feature} (correlation = {corr_value:.3f})"
        )
else:
    print("Removed numerical features: None")

if removed_categorical_features:
    print("\nRemoved categorical features:")
    for removed_feature in removed_categorical_features:
        matching_result = next(
            result for result in chi_square_results if result[0] == removed_feature
        )
        _, chi_square_stat, p_value = matching_result
        print(
            f"- {removed_feature}: removed because chi2 = {chi_square_stat:.3f} "
            f"and p-value = {p_value:.6f} (> 0.05)"
        )
else:
    print("\nRemoved categorical features: None")

print("\nSelected features after feature selection:")
print(selected_features)

# 2h) Measure class imbalance in the target and then apply simple random undersampling.
# The goal is to reduce the dominance of the majority class so both classes are represented equally.
print("\n--- 2h) Class Imbalance Analysis and Sampling ---")
class_distribution_before = df['Revenue'].value_counts()
class_percentage_before = df['Revenue'].value_counts(normalize=True) * 100

print("Class distribution before sampling:")
print(class_distribution_before)
print("\nClass percentage before sampling:")
print(class_percentage_before.round(2))

majority_class = df[df['Revenue'] == False]
minority_class = df[df['Revenue'] == True]

balanced_majority = resample(
    majority_class,
    replace=False,
    n_samples=len(minority_class),
    random_state=42,
)

df_sampled = pd.concat([balanced_majority, minority_class]).sample(
    frac=1,
    random_state=42,
).reset_index(drop=True)

# 2i) Compare the class distribution before and after sampling to confirm that the balancing step worked.
print("\n--- 2i) Class Distribution Before vs After Sampling ---")
class_distribution_after = df_sampled['Revenue'].value_counts()
class_percentage_after = df_sampled['Revenue'].value_counts(normalize=True) * 100

print("Before sampling:")
print(class_distribution_before)
print("\nAfter simple random undersampling:")
print(class_distribution_after)

print("\nPercentage before sampling:")
print(class_percentage_before.round(2))
print("\nPercentage after sampling:")
print(class_percentage_after.round(2))
