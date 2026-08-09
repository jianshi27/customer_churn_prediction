# Customer Churn Prediction & Retention Analytics

An end-to-end customer analytics project that uses Python, SQL-oriented business analysis, machine learning, SHAP explainability, and customer risk segmentation to identify customers at risk of churn and translate predictions into retention strategies.

---

## Business Problem

Customer churn can significantly affect recurring revenue in the telecommunications industry.

The objective of this project is to analyze customer behavior, identify factors associated with churn, build a predictive classification model, explain model predictions, and translate customer risk into actionable retention priorities.

The project addresses five key questions:

1. Which customer characteristics are associated with churn?
2. Which customers are most likely to churn?
3. Which classification model performs best?
4. Why does the model identify a customer as high risk?
5. How can churn predictions be translated into practical retention actions?

---

## Dataset

The project uses the **Telcom Customer Churn Dataset** available on Kaggle.

The dataset contains 7,043 customer records and 21 columns covering:

- Customer demographics
- Tenure
- Services subscribed
- Contract type
- Billing information
- Payment method
- Monthly and total charges
- Churn status

The dataset is adapted from the IBM Sample Data Sets and is intended for educational and research use.

---

## Project Workflow

```text
Customer Data
     ↓
Data Cleaning & Validation
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Preprocessing Pipeline
     ↓
Model Comparison
     ↓
Final Logistic Regression Model
     ↓
SHAP Explainability
     ↓
Customer Churn Probability
     ↓
Risk Segmentation
     ↓
Revenue Exposure Analysis
     ↓
Retention Prioritization
