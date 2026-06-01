import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)

# Load data
df_churn = pd.read_csv("synthetic_customer_churn_100k.csv")

# Display first few rows
print(df_churn.head())

# ============ STEP 1: EDA ============

# Dataset shape
print("Rows:", df_churn.shape[0])
print("Columns:", df_churn.shape[1])

# Data info
print(df_churn.info())

# Check for null values
print(df_churn.isnull().sum())

# Check for duplicates
print(df_churn.duplicated().sum())

# Churn distribution
print(df_churn['Churn'].value_counts())
print(df_churn['Churn'].value_counts(normalize=True))

# Visualize churn distribution
plt.figure(figsize=(5, 4))
sns.countplot(x='Churn', data=df_churn, palette=['#06d6a0', '#ef476f'])
plt.title("Distribution of Churn")
plt.show()

# Drop CustomerID
df_churn = df_churn.drop(columns=['CustomerID'])

# Identify numerical columns
numerical_cols = df_churn.select_dtypes(include=['int64', 'float64']).columns
print(numerical_cols)

# ============ Univariate Analysis ============

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 17,
    "axes.titleweight": "bold"
})

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Univariate Distribution of Numerical Features", fontweight="bold")
axes = axes.flatten()

sns.histplot(df_churn["Age"], bins=range(0, 100, 5), kde=True, ax=axes[0])
axes[0].set_title("Age (yrs)")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Number of Customers")

sns.histplot(df_churn["Tenure"], bins=range(0, 80, 5), kde=True, ax=axes[1], color="#5f9ea0")
axes[1].set_title("Tenure Distribution")
axes[1].set_xlabel("Tenure (months)")
axes[1].set_ylabel("Number of Customers")

sns.histplot(df_churn["MonthlyCharges"], bins=20, kde=True, ax=axes[2], color="#8fbc8f")
axes[2].set_title("Monthly Charges Distribution")
axes[2].set_xlabel("Monthly Charges")
axes[2].set_ylabel("Number of Customers")

sns.histplot(df_churn["TotalCharges"], bins=20, kde=True, ax=axes[3], color="#ef476f")
axes[3].set_title("Total Charges Distribution")
axes[3].set_xlabel("Total Charges")
axes[3].set_ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ============ Boxplots ============

plt.figure(figsize=(16, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x=[col]*len(df_churn), y=df_churn[col], hue=[col]*len(df_churn), 
                palette="viridis", legend=False)
    plt.title(col)
plt.tight_layout()
plt.show()

# ============ Bivariate Analysis ============

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Churn', y=col, data=df_churn, palette=["#06d6a0", "#ef476f"])
    plt.title(f"{col} vs Churn")
    plt.show()

# Summary statistics by churn
print(df_churn.groupby('Churn')[numerical_cols].mean())
