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


# Feature Engineering: Create Tenure Groups

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49+ months"
    ]
)

print("\nCustomer count by tenure group:")
print(df["TenureGroup"].value_counts().sort_index())

# Visualize the tenure groups
plt.close("all")

fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(
    data=df,
    x="TenureGroup",
    order=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49+ months"
    ],
    ax=ax
)

ax.set_title("Customer Distribution by Tenure Group")
ax.set_xlabel("Tenure Group")
ax.set_ylabel("Number of Customers")
ax.tick_params(axis="x", rotation=15)

fig.tight_layout()

fig.savefig(
    r"E:\ca2_predictiveanalysis\tenure_groups.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Correlation Matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical columns
numeric_df = df.select_dtypes(include=["number"])


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
print("\nCorrelation Matrix:")
print(correlation.round(2))


# ==========================================
# MACHINE LEARNING MODEL IMPLEMENTATION
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Create target variable
df["Churn_num"] = df["Churn"].map({"No": 0, "Yes": 1})

# Separate features and target
X = df.drop(columns=["Churn", "Churn_num"], errors="ignore")
y = df["Churn_num"]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

# Preprocess numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define four models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
}

results = []
trained_models = {}


# Train and evaluate each model
for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

    trained_models[name] = pipeline

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(classification_report(
        y_test, y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    ))

    # Confusion matrix for each model
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        rf"E:\ca2_predictiveanalysis\confusion_matrix_{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


# ==========================================
# MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

# Sort models by F1 Score
results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

# Display results
print("\n" + "=" * 60)
print("MACHINE LEARNING MODEL COMPARISON")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

# Save comparison table
results_df.to_csv(
    r"E:\ca2_predictiveanalysis\model_comparison.csv",
    index=False
)


# ==========================================
# GRAPH: COMPARISON OF FIVE METRICS
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

plot_df = results_df.set_index("Model")[metrics]

plt.close("all")

fig, ax = plt.subplots(figsize=(12, 6))

plot_df.plot(
    kind="bar",
    ax=ax,
    width=0.8
)

ax.set_title(
    "Comparison of Machine Learning Models",
    fontsize=14
)
ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)

ax.legend(
    title="Evaluation Metrics",
    loc="lower right"
)

ax.tick_params(axis="x", rotation=20)

fig.tight_layout()

# Save the graph
fig.savefig(
    r"E:\ca2_predictiveanalysis\model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ==========================================
# DECISION TREE VISUALIZATION
# ==========================================

from sklearn.tree import plot_tree

# Get the trained Decision Tree model
dt_model = trained_models["Decision Tree"]

# Get the preprocessed training data
preprocessor_fitted = dt_model.named_steps["preprocessor"]
classifier = dt_model.named_steps["classifier"]

X_train_transformed = preprocessor_fitted.transform(X_train)
feature_names = preprocessor_fitted.get_feature_names_out()

# Plot the decision tree
plt.figure(figsize=(24, 12))

plot_tree(
    classifier,
    feature_names=feature_names,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=7,
    max_depth=3
)

plt.title("Decision Tree Visualization - Customer Churn")
plt.tight_layout()

plt.savefig(
    r"E:\ca2_predictiveanalysis\decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Decision Tree diagram saved successfully!")



# ==========================================
# IDENTIFY THE BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

print("\nBest model based on F1 Score:", best_model_name)

print("\nFiles saved successfully:")
print("- model_comparison.csv")
print("- model_comparison.png")




# ==========================================
# FEATURE IMPORTANCE ANALYSIS
# ==========================================

# Get the trained Random Forest model
rf_model = trained_models["Random Forest"]

# Get the feature names after preprocessing
preprocessor_fitted = rf_model.named_steps["preprocessor"]

feature_names = preprocessor_fitted.get_feature_names_out()

# Get feature importance values
importance_values = (
    rf_model.named_steps["classifier"].feature_importances_
)

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Select the top 15 important features
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(15)

print("\n" + "=" * 60)
print("TOP 15 FEATURES INFLUENCING CUSTOMER CHURN")
print("=" * 60)

print(importance_df.round(4).to_string(index=False))

# Save feature importance results
importance_df.to_csv(
    r"E:\ca2_predictiveanalysis\feature_importance.csv",
    index=False
)

# Plot feature importance
plt.figure(figsize=(12, 7))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Features Influencing Customer Churn")
plt.xlabel("Feature Importance")
plt.ylabel("Customer Features")
plt.tight_layout()

plt.savefig(
    r"E:\ca2_predictiveanalysis\feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nFeature importance files saved successfully:")
print("- feature_importance.csv")
print("- feature_importance.png")



# ==========================================
# FINAL PROJECT SUMMARY AND CONCLUSION
# ==========================================

print("\n" + "=" * 60)
print("TELCO CUSTOMER CHURN PREDICTION")
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print("\n1. DATASET SUMMARY")
print("Total customers analyzed:", len(df))
print("Customers who left:", (df["Churn"] == "Yes").sum())
print("Customers who stayed:", (df["Churn"] == "No").sum())

churn_rate = (df["Churn"] == "Yes").mean() * 100
print(f"Overall customer churn rate: {churn_rate:.2f}%")

print("\n2. MACHINE LEARNING SUMMARY")
print(results_df.round(4).to_string(index=False))

print("\n3. BEST MODEL")
print("Best model based on F1 Score:", best_model_name)

best_row = results_df.iloc[0]

print(f"Accuracy:  {best_row['Accuracy']:.4f}")
print(f"Precision: {best_row['Precision']:.4f}")
print(f"Recall:    {best_row['Recall']:.4f}")
print(f"F1 Score:  {best_row['F1 Score']:.4f}")
print(f"ROC-AUC:   {best_row['ROC-AUC']:.4f}")

print("\n4. PROJECT CONCLUSION")
print(
    "This project analyzed customer churn in a telecommunications "
    "dataset using data cleaning, exploratory data analysis, "
    "visualization, and machine learning."
)

print(
    "Four classification models were compared using accuracy, "
    "precision, recall, F1 score, and ROC-AUC."
)

print(
    "The results can help identify customers who may be at risk "
    "of leaving and support customer retention strategies."
)

print("\n5. PROJECT COMPLETED")
print("Check the project folder for the saved graphs and CSV files.")
print("=" * 60)

# Save final model comparison and summary
with open(
    r"E:\ca2_predictiveanalysis\project_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("TELCO CUSTOMER CHURN PREDICTION\n")
    file.write("FINAL PROJECT SUMMARY\n\n")

    file.write(f"Total customers analyzed: {len(df)}\n")
    file.write(
        f"Customers who left: {(df['Churn'] == 'Yes').sum()}\n"
    )
    file.write(
        f"Customers who stayed: {(df['Churn'] == 'No').sum()}\n"
    )
    file.write(f"Overall churn rate: {churn_rate:.2f}%\n\n")

    file.write("MODEL COMPARISON:\n")
    file.write(results_df.round(4).to_string(index=False))

    file.write(f"\n\nBest model based on F1 Score: {best_model_name}\n")

print("\nSummary saved as project_summary.txt")
