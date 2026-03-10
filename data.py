import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
    # PART 2 : Data Preparation




    
    # 2a) Meaningful filtering condition
# We filter out sessions where the user visited 0 product-related pages.
# Justification: If a user doesn't look at a single product, their purchasing intention is fundamentally zero. 
# Removing these "instant bounce" or purely administrative visits helps our future models focus on actual shopper behavior.
initial_shape = df.shape
df_filtered = df[df['ProductRelated'] > 0].copy()
print(f"2a) Filtering: Removed {initial_shape[0] - df_filtered.shape[0]} rows where ProductRelated was 0.")
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
print("As verified in Part 1(h), the dataset contains 0 missing values. No imputation (median/mode) was required.")