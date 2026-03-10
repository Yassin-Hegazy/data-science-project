import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

##################################################
# Part 1: Data Exploration
##################################################

df = pd.read_csv('online_shoppers_intention.csv')

print('--- First 12 Rows ---')
print(df.head(12))

print('\n--- Last 12 Rows ---')
print(df.tail(12))

print('\n--- 1b) Total Rows and Columns ---')
rows, cols = df.shape
print(f'Total rows: {rows}')
print(f'Total columns: {cols}')

print('\n--- 1c) Column Names and Data Types ---')
print(df.dtypes)

print('\n--- 1d) Full Dataset Summary ---')
df.info()

print("\n--- 1e) Target Variable Class Distribution ('Revenue') ---")
print(df['Revenue'].value_counts(dropna=False))
print('\nPercentage distribution:')
print(df['Revenue'].value_counts(normalize=True) * 100)

print("\n--- 1f) Categorical Feature Analysis ('VisitorType') ---")
print('Distinct values:')
print(df['VisitorType'].unique())
print(f"Most frequent value: {df['VisitorType'].mode()[0]}")

print('\n--- 1g) Numerical Features Statistics ---')
stats = df.describe(percentiles=[.20, .50, .75])
print(stats)

print('\n--- 1h) Missing Values per Column ---')
print(df.isnull().sum())

print('\n--- 1i) Duplicate Records ---')
duplicates_count = df.duplicated().sum()
print(f'Number of duplicate records found: {duplicates_count}')

if duplicates_count > 0:
    df = df.drop_duplicates()
    print(f'Duplicates removed. New dataset shape: {df.shape}')
else:
    print('No duplicates to remove.')
