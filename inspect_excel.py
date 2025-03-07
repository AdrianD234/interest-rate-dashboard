import pandas as pd

# Load the Excel file
print("Loading Excel file...")
df = pd.read_excel('Rev_IR.xlsx')

# Display general information
print("\nFile Information:")
print(f"- Total rows: {len(df)}")
print(f"- Total columns: {len(df.columns)}")
print(f"- Column names: {df.columns.tolist()}")

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Display data types
print("\nData types:")
print(df.dtypes)

# Analyze missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Wait for user input before closing
input("\nPress Enter to exit...")