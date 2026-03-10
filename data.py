import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
# Make sure the filename matches exactly what is in your folder
df = pd.read_csv('online_shoppers_intention.csv')

# Part 1 a) Display the first and last 12 rows 
print("--- First 12 Rows ---")
print(df.head(12))

print("\n--- Last 12 Rows ---")
print(df.tail(12))
# b) Identify and print the total number of rows and columns 
print("\n--- b) Total Rows and Columns ---")
rows, cols = df.shape
print(f"Total rows: {rows}")
print(f"Total columns: {cols}")

# c) List all column names and their corresponding data types 
print("\n--- c) Column Names and Data Types ---")
print(df.dtypes)

# d) Generate a full dataset summary (non-null counts and data types) 
print("\n--- d) Full Dataset Summary ---")
df.info() # Note: info() automatically prints to the console

# e) Identify the target variable and analyze its class distribution 
print("\n--- e) Target Variable Class Distribution ('Revenue') ---")
print(df['Revenue'].value_counts(dropna=False))
print("\nPercentage distribution:")
print(df['Revenue'].value_counts(normalize=True) * 100)

# f) Select one categorical feature and display distinct/most frequent values 
print("\n--- f) Categorical Feature Analysis ('VisitorType') ---")
print("Distinct values:")
print(df['VisitorType'].unique())
print(f"Most frequent value: {df['VisitorType'].mode()[0]}")

# g) Compute mean, median, standard dev, and percentiles (20%, 50%, 75%) 
print("\n--- g) Numerical Features Statistics ---")
# The describe method calculates mean, std, and we can specify exact percentiles (50% is the median)
stats = df.describe(percentiles=[.20, .50, .75])
print(stats)

# h) Detect missing values and report their counts per column 
print("\n--- h) Missing Values per Column ---")
print(df.isnull().sum())

# i) Identify duplicate records and remove them if found 
print("\n--- i) Duplicate Records ---")
duplicates_count = df.duplicated().sum()
print(f"Number of duplicate records found: {duplicates_count}")

if duplicates_count > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. New dataset shape: {df.shape}")
else:
    print("No duplicates to remove.")