import pandas as pd

# ==========================================
# 1. LOAD THE ORIGINAL DATASET
# ==========================================
df = pd.read_csv(r"E:\ca2_predictiveanalysis\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Original dataset shape:", df.shape)
print(df.head())

# ==========================================
# 2. CHECK MISSING VALUES AND DUPLICATES
# ==========================================
print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows before cleaning:")
print(df.duplicated().sum())

# ==========================================
# 3. CONVERT TotalCharges TO NUMERIC
# ==========================================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nMissing TotalCharges after conversion:",df["TotalCharges"].isnull().sum())

# New customers with zero tenure have no accumulated
# total charges in this dataset.
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# ==========================================
# 4. STANDARDIZE SERVICE VALUES
# ==========================================
cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup","DeviceProtection", "TechSupport", "StreamingTV","StreamingMovies"]

for c in cols:
    df[c] = df[c].replace({"No internet service": "No","No phone service": "No"})

# Convert SeniorCitizen into readable categories
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No",1: "Yes"})

# ==========================================
# 5. CREATE THE TARGET AND REMOVE CUSTOMER ID
# ==========================================
df["Churn_num"] = df["Churn"].map({"No": 0,"Yes": 1})

df = df.drop(columns=["customerID"])

# ==========================================
# 6. REMOVE DUPLICATE ROWS
# ==========================================
print("\nDuplicate rows before removal:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Duplicate rows after removal:", df.duplicated().sum())

# ==========================================
# 7. VERIFY THE CLEANED DATASET
# ==========================================
print("\nFinal dataset shape:", df.shape)
print("Total missing values:", df.isnull().sum().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nSenior citizen distribution:")
print(df["SeniorCitizen"].value_counts())

print("\nColumn names:")
print(df.columns.tolist())

# ==========================================
# 8. SAVE THE FINAL CLEANED DATASET
# ==========================================
df.to_csv(r"E:\ca2_predictiveanalysis\telco_cleaned.csv",index=False)

print("\nFinal cleaned dataset saved successfully!")


import pandas as pd

df = pd.read_csv(r"E:\ca2_predictiveanalysis\telco_cleaned.csv")

print("Dataset shape:", df.shape)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("Total missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts())


#obj 1
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))

ax = sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Churn")
plt.ylabel("Number of Customers")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()
plt.savefig(r"E:\ca2_predictiveanalysis\churn_distribution.png",dpi=300)
plt.show()

print("\nChurn percentages:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))


# Objective 2: Analyze Churn by Contract Type

plt.close("all")

fig, ax = plt.subplots(figsize=(9, 5))

sns.countplot(data=df, x="Contract", hue="Churn", ax=ax)

ax.set_title("Customer Churn by Contract Type")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Number of Customers")
ax.legend(title="Churn")

fig.tight_layout()

fig.savefig(r"E:\ca2_predictiveanalysis\churn_by_contract.png",dpi=300,bbox_inches="tight")

plt.show()

print("\nChurn percentage by contract type:")

contract_churn = pd.crosstab(df["Contract"],df["Churn"],normalize="index") * 100

print(contract_churn.round(2))


# Objective 3: Analyze Tenure by Churn

plt.close("all")

fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(data=df, x="Churn", y="tenure", ax=ax)

ax.set_title("Customer Tenure by Churn Status")
ax.set_xlabel("Customer Churn")
ax.set_ylabel("Tenure (Months)")

fig.tight_layout()

fig.savefig(r"E:\ca2_predictiveanalysis\tenure_by_churn.png",dpi=300,bbox_inches="tight")

plt.show()

print("\nAverage tenure by churn status:")
print(df.groupby("Churn")["tenure"].mean().round(2))


# Objective 4: Analyze Monthly Charges by Churn

plt.close("all")

fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)

ax.set_title("Monthly Charges by Churn Status")
ax.set_xlabel("Customer Churn")
ax.set_ylabel("Monthly Charges")

fig.tight_layout()

fig.savefig(r"E:\ca2_predictiveanalysis\monthly_charges_by_churn.png",dpi=300,bbox_inches="tight")

plt.show()

print("\nAverage monthly charges by churn status:")
print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))


# Objective 5: Analyze Churn by Internet Service

plt.close("all")

fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(data=df, x="InternetService",hue="Churn", ax=ax)

ax.set_title("Customer Churn by Internet Service")
ax.set_xlabel("Internet Service Type")
ax.set_ylabel("Number of Customers")
ax.legend(title="Churn")

fig.tight_layout()

fig.savefig( r"E:\ca2_predictiveanalysis\churn_by_internet_service.png",dpi=300,bbox_inches="tight")

plt.show()

print("\nChurn percentage by internet service:")

internet_churn = pd.crosstab(df["InternetService"],df["Churn"], normalize="index") * 100

print(internet_churn.round(2))


# Correlation Matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical columns
numeric_df = df.select_dtypes(include=["number"])

# Calculate correlation
correlation = numeric_df.corr()

# Plot correlation matrix
plt.close("all")
fig, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

ax.set_title("Correlation Matrix of Numerical Features")
plt.tight_layout()

plt.savefig(
    r"E:\ca2_predictiveanalysis\correlation_matrix.png",
    dpi=300
)

plt.show()
