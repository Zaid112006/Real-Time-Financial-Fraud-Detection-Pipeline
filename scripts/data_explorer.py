# ============================================
# DATA EXPLORER SCRIPT - Fraud Detection Project
# ============================================
# What this script does (in plain English):
# 1. Opens the transactions file
# 2. Shows basic info: how many rows, what columns exist
# 3. Counts how many transactions are fraud vs not fraud
# 4. Makes a few simple charts
#
# You don't need to understand every line yet.
# Just run it, look at the output, and ask questions about
# anything that doesn't make sense.
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- STEP 1: Load the data ----
# This reads the CSV file into a table (called a "DataFrame")
df = pd.read_csv("transactions_train.csv")

print("=" * 50)
print("STEP 1: BASIC INFO")
print("=" * 50)

# How many rows and columns?
print(f"Number of transactions (rows): {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Column names: {list(df.columns)}")

# Show the first 5 rows so we can see what the data looks like
print("\nFirst 5 rows of data:")
print(df.head())

# ---- STEP 2: Check for missing data ----
print("\n" + "=" * 50)
print("STEP 2: MISSING VALUES CHECK")
print("=" * 50)
print(df.isnull().sum())
print("(If all zeros, there's no missing data - good!)")

# ---- STEP 3: Fraud count ----
print("\n" + "=" * 50)
print("STEP 3: HOW MANY TRANSACTIONS ARE FRAUD?")
print("=" * 50)
fraud_counts = df["isFraud"].value_counts()
print(fraud_counts)
print(f"\nFraud percentage: {df['isFraud'].mean() * 100:.4f}%")
print("(Fraud is usually a tiny fraction of all transactions - this is normal)")

# ---- STEP 4: Which transaction TYPES are fraud? ----
print("\n" + "=" * 50)
print("STEP 4: FRAUD BY TRANSACTION TYPE")
print("=" * 50)
fraud_by_type = df.groupby("type")["isFraud"].sum()
print(fraud_by_type)
print("\n(You'll likely see fraud ONLY happens in certain types, e.g. TRANSFER/CASH_OUT)")

# ---- STEP 5: Simple charts ----
print("\n" + "=" * 50)
print("STEP 5: MAKING CHARTS (saved as image files)")
print("=" * 50)

# Chart 1: Fraud vs Not Fraud count
plt.figure(figsize=(6, 4))
sns.countplot(x="isFraud", data=df)
plt.title("Fraud vs Not Fraud Count")
plt.xlabel("0 = Not Fraud, 1 = Fraud")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("chart_fraud_count.png")
print("Saved: chart_fraud_count.png")
plt.close()

# Chart 2: Transaction type distribution
plt.figure(figsize=(8, 4))
sns.countplot(x="type", data=df)
plt.title("Number of Transactions by Type")
plt.tight_layout()
plt.savefig("chart_transaction_types.png")
print("Saved: chart_transaction_types.png")
plt.close()

# Chart 3: Fraud count by type
plt.figure(figsize=(8, 4))
fraud_by_type.plot(kind="bar", color="crimson")
plt.title("Fraud Count by Transaction Type")
plt.ylabel("Number of Fraud Cases")
plt.tight_layout()
plt.savefig("chart_fraud_by_type.png")
print("Saved: chart_fraud_by_type.png")
plt.close()

print("\n" + "=" * 50)
print("DONE! Check the folder for 3 chart images.")
print("=" * 50)
