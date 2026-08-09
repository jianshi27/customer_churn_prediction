# Customer Churn Prediction & Retention Analytics

An end-to-end customer churn analysis project using Python, exploratory data analysis, machine learning, SHAP explainability, customer risk segmentation, and retention prioritization.

The project analyzes customer behavior, predicts churn probability, explains model predictions, and translates high-risk customers into business-focused retention priorities.

---

## Business Problem

Customer churn can significantly affect recurring revenue in the telecommunications industry.

The objective of this project is to analyze customer behavior and build a machine learning model that can identify customers who are at higher risk of churn.

The project focuses on five key questions:

1. Which customer characteristics are associated with churn?
2. Which customers are most likely to churn?
3. Which classification model performs best?
4. What factors contribute to a customer's predicted churn risk?
5. How can churn predictions be translated into practical retention priorities?

---

## Dataset

The project uses the **Telcom Customer Churn Dataset** available on Kaggle.

The dataset contains **7,043 customer records and 21 columns**, covering:

- Customer demographics
- Tenure
- Services subscribed
- Contract type
- Billing information
- Payment method
- Monthly charges
- Total charges
- Churn status

The dataset is described as a fictional telecommunications customer dataset and is adapted from the IBM Sample Data Sets for educational and research use.

The raw dataset is not included in this repository.

---

## Project Workflow

```text
Customer Data
     ↓
Data Cleaning
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
Churn Probability
     ↓
Customer Risk Segmentation
     ↓
Revenue Exposure Analysis
     ↓
Retention Prioritization
```

## Data Cleaning

The original dataset contained `TotalCharges` as an object/string column rather than a numeric column.

The column was converted to numeric values using coercion for invalid entries. This identified **11 invalid `TotalCharges` values**, which were removed during cleaning.

The `Churn` target variable was also converted from categorical values into binary values:

- `No` → `0`
- `Yes` → `1`

After cleaning:

- Dataset size: **7,032 rows × 21 columns**
- Missing values: **0**
- Target values: **0** and **1**

---

## Exploratory Data Analysis

The exploratory analysis examined customer behavior and churn patterns across demographic, service, contract, billing, and tenure-related variables.

The analysis included:

- Overall churn distribution
- Tenure groups
- Contract type
- Internet service
- Payment method
- Monthly charges
- Total charges
- Service adoption
- Customer service count

### Overall Churn

After cleaning, the dataset contained:

| Customer Status | Customers | Percentage |
|---|---:|---:|
| Stayed | 5,163 | 73.42% |
| Churned | 1,869 | 26.58% |

Churn represents a minority class in the dataset. Therefore, model evaluation considers precision, recall, F1-score, and ROC-AUC alongside accuracy.

---

## Key EDA Findings

### Tenure and Churn

Shorter-tenure customers showed substantially higher observed churn rates.

| Tenure Group | Churn Rate |
|---|---:|
| 0–12 months | 47.68% |
| 13–24 months | 28.71% |
| 25–48 months | 20.39% |
| 49–60 months | 14.42% |
| 61–72 months | 6.61% |

The observed churn rate decreases substantially as customer tenure increases. This suggests that the early customer lifecycle is an important period to consider when designing retention strategies.

---

### Contract Type and Churn

Contract type showed a strong difference in observed churn.

| Contract | Customers | Churn Rate |
|---|---:|---:|
| Month-to-month | 3,875 | 42.71% |
| One year | 1,472 | 11.28% |
| Two year | 1,685 | 2.85% |

Month-to-month customers had substantially higher observed churn than customers on longer-term contracts.

---

### Internet Service and Churn

| Internet Service | Customers | Churn Rate |
|---|---:|---:|
| Fiber optic | 3,096 | 41.89% |
| DSL | 2,416 | 19.00% |
| No internet | 1,520 | 7.43% |

Fiber optic customers showed the highest observed churn rate among the three internet service groups.

---

### Payment Method and Churn

| Payment Method | Customers | Churn Rate |
|---|---:|---:|
| Electronic check | 2,365 | 45.29% |
| Mailed check | 1,604 | 19.20% |
| Bank transfer (automatic) | 1,542 | 16.73% |
| Credit card (automatic) | 1,521 | 15.25% |

Electronic check customers showed the highest observed churn rate among the payment methods.

These findings describe associations observed in the dataset and should not be interpreted as causal relationships.

---

## Numerical Analysis

Churned customers showed different numerical characteristics compared with customers who stayed.

| Churn Status | Median Tenure | Median Monthly Charges | Median Total Charges |
|---|---:|---:|---:|
| Stayed | 38 months | 64.45 | 1,683.60 |
| Churned | 10 months | 79.65 | 703.55 |

Churned customers had substantially lower tenure and total charges but higher monthly charges compared with customers who stayed.

This pattern is consistent with the earlier tenure analysis, where newer customers showed substantially higher observed churn.

---

## Feature Engineering

A `service_count` feature was created to represent the number of subscribed service-related features for each customer.

The feature was derived from the available service-related variables and provides a simple measure of service adoption.

The resulting `service_count` ranged from **0 to 6**.

The customer identifier, `customerID`, was excluded from modeling because it is an identifier rather than a meaningful predictive feature.

---

## Machine Learning Preparation

The target variable was separated from the predictor variables before model training.

The final predictors consisted of:

### Numerical Features

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `service_count`

### Categorical Features

- `gender`
- `Partner`
- `Dependents`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`

---

## Train/Test Split

The dataset was split into training and testing sets using an **80/20 split**.

Stratification was used to preserve the proportion of churned customers in both datasets.

| Dataset | Rows | Churn Rate |
|---|---:|---:|
| Training | 5,625 | 26.58% |
| Testing | 1,407 | 26.58% |

The matching churn rates indicate that stratification preserved the target distribution across the two sets.

---

## Preprocessing

A scikit-learn preprocessing pipeline was used to prepare the data for machine learning.

### Numerical Variables

Numerical features were:

1. Imputed using the median
2. Standardized using `StandardScaler`

### Categorical Variables

Categorical features were:

1. Imputed using the most frequent value
2. One-hot encoded using `OneHotEncoder`

The preprocessing transformed the original **20 predictors into 46 processed features**.

No missing values remained in either the processed training or testing data.

---

## Machine Learning Models

Three classification models were evaluated:

1. **Logistic Regression**
2. **XGBoost**
3. **Random Forest**

Logistic Regression was used as an interpretable baseline, while Random Forest and XGBoost were included to evaluate nonlinear ensemble approaches.

Because churn is the minority class, accuracy was not treated as the only measure of model performance.

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Model Comparison

The models were evaluated on the held-out test set.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.45% | 64.95% | 57.49% | 60.99% | **83.59%** |
| XGBoost | 73.63% | 50.26% | **78.07%** | **61.15%** | 83.16% |
| Random Forest | 78.25% | 61.97% | 47.06% | 53.50% | 81.42% |

---

## Model Evaluation

### Logistic Regression

Logistic Regression achieved:

- Accuracy: **80.45%**
- Precision: **64.95%**
- Recall: **57.49%**
- F1-score: **60.99%**
- ROC-AUC: **83.59%**

The model correctly classified a large proportion of customers while achieving the highest ROC-AUC among the three evaluated models.

---

### XGBoost

XGBoost achieved:

- Accuracy: **73.63%**
- Precision: **50.26%**
- Recall: **78.07%**
- F1-score: **61.15%**
- ROC-AUC: **83.16%**

XGBoost produced substantially higher recall than Logistic Regression, identifying more of the actual churners.

However, this came with lower precision and more false positives.

---

### Random Forest

Random Forest achieved:

- Accuracy: **78.25%**
- Precision: **61.97%**
- Recall: **47.06%**
- F1-score: **53.50%**
- ROC-AUC: **81.42%**

Random Forest performed reasonably well on accuracy and precision but had the lowest recall and ROC-AUC of the three models.

---

## Final Model Selection

**Logistic Regression** was selected as the final model.

It achieved the highest ROC-AUC among the evaluated models while also providing the highest accuracy and precision.

Final test performance:

- **Accuracy:** 80.45%
- **Precision:** 64.95%
- **Recall:** 57.49%
- **F1-score:** 60.99%
- **ROC-AUC:** 83.59%

Although XGBoost achieved slightly higher F1-score and substantially higher recall, it generated considerably more false positives.

The final model selection therefore considered the overall balance between predictive performance, precision, recall, and interpretability.

---

## SHAP Explainability

SHAP was used to provide model explainability and identify the features that contributed most strongly to the model's predictions.

The SHAP analysis was performed on the processed test data and evaluated feature contributions using mean absolute SHAP values.

The top features included:

1. `tenure`
2. `TotalCharges`
3. `MonthlyCharges`
4. `Contract_Month-to-month`
5. `InternetService_DSL`
6. `InternetService_Fiber optic`
7. `Contract_Two year`
8. `MultipleLines_No`
9. `service_count`
10. `PaperlessBilling_No`

SHAP provides an additional interpretability layer beyond aggregate model metrics. It helps explain which features have the greatest influence on the model's churn predictions.

---

## Customer Risk Segmentation

The final prediction pipeline was used to generate churn probabilities for the **1,407 customers in the test set**.

Customers were grouped into three risk categories based on predicted churn probability:

- **Low Risk:** probability < 40%
- **Medium Risk:** probability between 40% and 70%
- **High Risk:** probability ≥ 70%

| Risk Segment | Customers | Percentage |
|---|---:|---:|
| Low Risk | 965 | 68.59% |
| Medium Risk | 359 | 25.52% |
| High Risk | 83 | 5.90% |

This segmentation provides a simple way to translate model probabilities into actionable customer groups.

---

## High-Risk Customer Analysis

The **83 high-risk customers** had:

- **Average predicted churn probability:** 75.00%
- **Average monthly charge:** 84.14
- **Average tenure:** 6.43 months
- **Total monthly revenue exposure:** 6,983.85

All 83 high-risk customers were on **month-to-month contracts**.

The combined monthly charges of these customers represent approximately **6,983.85 in current monthly revenue exposure**.

This figure represents potential exposure based on the current customer portfolio. It should not be interpreted as guaranteed future revenue loss.

---

## Retention Prioritization

A simple retention-priority score was created by combining predicted churn probability with monthly charges.

The purpose of this score is to prioritize customers where higher churn risk overlaps with greater recurring revenue exposure.

### Early-Tenure Customers

Customers with short tenure and high predicted churn probability could be prioritized for proactive onboarding and engagement.

### Month-to-Month Customers

Customers with high churn probability and month-to-month contracts could be considered for targeted incentives encouraging longer-term contracts.

### High-Value, High-Risk Customers

Customers with both high predicted churn probability and higher monthly charges could receive higher-priority retention attention.

### Personalized Interventions

SHAP explanations can be used to understand the major factors contributing to an individual customer's predicted risk and support more targeted retention actions.

These recommendations are analytical suggestions and would need to be validated through actual retention campaigns or controlled experiments.

---

## Model Deployment

A complete scikit-learn prediction pipeline was created containing:

- Numerical preprocessing
- Categorical preprocessing
- Missing-value imputation
- Standardization
- One-hot encoding
- Logistic Regression

The complete pipeline was serialized using `joblib`.

The saved model is:

```text
models/churn_prediction_pipeline.pkl
```
The saved pipeline was successfully reloaded and tested.

The reloaded pipeline produced predictions and churn probabilities that matched the original pipeline outputs.

This allows the trained model to be reused without repeating the training and preprocessing steps.

---

## Streamlit Dashboard

A basic Streamlit dashboard was created as a presentation layer for the churn prediction project.

The dashboard is designed to provide an interface for exploring:

- Customer churn probability
- Customer risk segment
- Customer characteristics
- Retention priority
- High-risk customer information

The dashboard code is located at:

```text
app/app.py
```
The current dashboard is a basic prototype and can be expanded with additional visualizations, customer-level explanations, and deployment functionality.

---

## Technology Stack

- **Python** — analysis and machine learning
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **Seaborn** — exploratory visualization
- **Scikit-learn** — preprocessing, modeling, and evaluation
- **XGBoost** — gradient boosting model
- **SHAP** — model explainability
- **Joblib** — model serialization
- **Streamlit** — dashboard prototype
- **Git / GitHub** — version control and project sharing

---

## Project Structure

```text
customer-churn-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── README.md
│
├── models/
│   └── churn_prediction_pipeline.pkl
│
├── notebooks/
│   └── Customer_Churn_Analysis.ipynb
│
├── reports/
│   └── README.md
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Limitations

This project uses historical observational customer data.

The model identifies predictive patterns and associations but does not establish causal relationships.

Revenue exposure is calculated using current monthly charges and should not be interpreted as guaranteed future revenue loss.

Retention recommendations have not been evaluated through randomized experiments or actual retention campaigns.

The model was evaluated using a single held-out test set. Further validation, cross-validation, and model calibration could provide additional confidence in generalization.

The current Streamlit dashboard is a basic prototype rather than a production deployment.

---

## Future Improvements

Potential improvements include:

- Hyperparameter tuning
- Cross-validation
- Probability threshold optimization
- Model calibration
- Additional customer segmentation
- Automated data-quality checks
- Time-based validation with future customer data
- A/B testing of retention strategies
- Production database integration
- Model monitoring
- Expanded Streamlit dashboard
- Automated prediction pipelines

---

## Disclaimer

This project is intended for educational, portfolio, and analytical demonstration purposes.

The churn predictions and revenue exposure calculations should be treated as analytical outputs rather than guaranteed business outcomes.

Retention strategies are proposed recommendations and should be validated using actual customer behavior and controlled experiments.
