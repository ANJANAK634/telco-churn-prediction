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



