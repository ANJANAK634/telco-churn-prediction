# Telco Customer Churn Prediction Using Machine Learning

## Project Overview

Telco Customer Churn Prediction is a Data Science and Machine Learning project that aims to analyse customer behaviour and predict whether a customer is likely to leave a telecommunications company.

Customer churn is an important challenge for telecom companies because losing customers can affect business growth and revenue. This project uses historical customer data to identify patterns associated with customer churn and build machine learning classification models to predict customer retention.

## Objectives

The main objectives of this project are:

1. To analyse customer data and understand customer behaviour.
2. To clean and preprocess the dataset.
3. To perform Exploratory Data Analysis (EDA).
4. To identify important factors influencing customer churn.
5. To build machine learning classification models.
6. To evaluate and compare the performance of different models.
7. To identify customers who may be at risk of leaving the service.

## Dataset Description

**Dataset:** Telco Customer Churn Dataset

**Source:** IBM Telco Customer Churn Dataset (available through Kaggle and other public dataset platforms).

The dataset contains information about 7,043 telecom customers and 21 original columns.

Some important attributes include:

- CustomerID
- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- PhoneService
- InternetService
- Contract
- MonthlyCharges
- TotalCharges
- Churn

The target variable is **Churn**, which indicates whether a customer has left the telecom service.

- Yes – Customer has left.
- No – Customer has stayed.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Spyder
- GitHub

## Data Preprocessing

The following preprocessing steps were performed:

- Checked and handled missing values.
- Converted TotalCharges into numeric format.
- Filled missing TotalCharges values with 0.
- Checked duplicate records.
- Converted SeniorCitizen values into categorical labels.
- Standardized service-related categorical values.
- Encoded the target variable Churn into numerical values.
- Created a new feature called TenureGroup.
- Removed unnecessary customer ID information from model input features.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand customer churn patterns.

The following visualizations were created:

1. Churn Distribution – Shows the number of customers who stayed and left.
2. Churn by Contract – Analyses churn across different contract types.
3. Tenure by Churn – Compares customer tenure between churned and retained customers.
4. Monthly Charges by Churn – Examines monthly charges in relation to churn.
5. Churn by Internet Service – Analyses churn across internet service categories.
6. Tenure Groups – Displays customer distribution across tenure categories.
7. Correlation Matrix – Examines relationships between numerical features.

## Machine Learning Algorithms

Four classification algorithms were implemented:

### 1. Logistic Regression
A supervised classification algorithm used to estimate the probability of customer churn.

### 2. K-Nearest Neighbors (KNN)
A classification algorithm that predicts customer churn based on the nearest data points.

### 3. Decision Tree Classifier
A tree-based algorithm that makes predictions using decision rules and feature conditions.

### 4. Random Forest Classifier
An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

## Model Evaluation

The models were evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

Confusion matrices were generated to understand correct and incorrect predictions for each model.

A model comparison graph was also created to compare the performance of all four algorithms.

## Feature Importance Analysis

Feature importance analysis was performed using the Random Forest model to identify the customer attributes that contribute most to its predictions.

The top 15 important features were visualized using a bar chart.

## Decision Tree Visualization

A Decision Tree diagram was generated to understand how the Decision Tree Classifier makes predictions based on customer attributes.

## Project Results

The project compares four machine learning models to identify the best-performing model based on F1-score.

The analysis helps identify customer characteristics associated with churn and provides insights that can support customer retention strategies.

**Note:** Actual model performance values are available in the generated `model_comparison.csv` file.

## Project Structure

```text
telco-churn-prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── telco_cleaned.csv
├── ca2_project_predictiveanly.py
├── README.md
│
├── churn_distribution.png
├── churn_by_contract.png
├── tenure_by_churn.png
├── monthly_charges_by_churn.png
├── churn_by_internet_service.png
├── tenure_groups.png
├── correlation_matrix.png
│
├── model_comparison.csv
├── model_comparison.png
├── feature_importance.csv
├── feature_importance.png
├── decision_tree.png
│
└── confusion_matrix_*.png
```

*The project structure above lists the expected files. Include only the files that are actually present in your GitHub repository.*

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/telco-churn-prediction.git
```

### Step 2: Navigate to the Project Folder

```bash
cd telco-churn-prediction
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Step 4: Run the Python File

```bash
python ca2_project_predictiveanly.py
```

Alternatively, open the Python file in Spyder or run the notebook in Jupyter Notebook.

## Conclusion

This project demonstrates how Data Science and Machine Learning techniques can be applied to predict customer churn in the telecommunications industry.

Through data preprocessing, exploratory data analysis, visualization, and machine learning, the project identifies patterns associated with customer churn and compares multiple classification algorithms.

The insights obtained can help telecom companies understand customer behaviour and develop strategies to improve customer retention.

## Future Scope

- Improve prediction performance through hyperparameter tuning.
- Experiment with additional machine learning algorithms.
- Develop an interactive dashboard for churn analysis.
- Build a web application for real-time customer churn prediction.
- Provide personalized customer retention recommendations.

## Acknowledgement

This project was developed as part of academic learning in Data Science and Predictive Analytics. It provided practical experience in data preprocessing, exploratory data analysis, machine learning, model evaluation, and visualization.



