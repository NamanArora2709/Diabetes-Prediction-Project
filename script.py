# load_and_preprocess.py

import pandas as pd

# Load the CSV file you already have
df = pd.read_csv("diabetes_dataset.csv")

# Show first few rows
print("First 5 rows:")
print(df.head())

# Show basic info
print("\nInfo:")
print(df.info())

# Describe the data
print("\nDescription:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())