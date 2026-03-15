# Customer Churn Prediction

An end-to-end machine learning project for predicting telecom customer churn using the Telco Customer Churn dataset.

## Project Overview

The goal of this project is to build a complete production-style machine learning workflow that:

- cleans and preprocesses customer data
- trains churn prediction models
- evaluates classification performance with business-focused metrics
- explains model behavior using SHAP
- serves predictions through a Streamlit web app

This project was built to practice a full ML pipeline from raw data to deployment.

## Dataset

This project uses the **Telco Customer Churn** dataset.

- Rows: 7,043
- Columns: 21 before cleaning
- Target variable: `Churn`

After cleaning:

- `customerID` was dropped
- `TotalCharges` was converted to numeric
- rows with missing `TotalCharges` were removed
- `Churn` was mapped to:
  - `0` = No churn
  - `1` = Churn

## Problem Type

Binary classification:

> Predict whether a customer is likely to churn.

Because churn prediction is an imbalanced business problem, this project focuses on metrics like **recall**, **precision**, **F1-score**, and **ROC-AUC**, not just accuracy.

## Project Workflow

### 1. Data Cleaning

Cleaning steps performed:

- removed `customerID`
- converted `TotalCharges` to numeric with coercion
- dropped rows with missing `TotalCharges`
- encoded `Churn` as 0/1

### 2. Exploratory Data Analysis

EDA included:

- churn class distribution
- correlation heatmap for numeric variables
- churn rate by categorical features such as:
  - contract type
  - internet service
  - payment method
- visualizations such as:
  - churn rate by contract
  - tenure distribution by churn
  - monthly charges vs total charges
  - correlation heatmap

### 3. Preprocessing Pipeline

A `ColumnTransformer` + `Pipeline` workflow was used to prevent data leakage.

#### Numeric features
- median imputation
- standard scaling

#### Categorical features
- most frequent imputation
- one-hot encoding with `handle_unknown="ignore"`

### 4. Models

Models trained:

- Logistic Regression baseline
- Random Forest classifier

The final saved model is a tuned Random Forest pipeline.

### 5. Threshold Tuning

Instead of using the default classification threshold of `0.50`, the project uses a tuned threshold of:

```python
CHURN_THRESHOLD = 0.35
```

This increases recall for churners, which is important when missing a true churner is more costly than flagging extra false positives.

### 6. Explainability

SHAP was used to understand which features most influence churn predictions.

Key patterns found from SHAP:

- **Month-to-month contracts** strongly increase churn risk
- **Shorter tenure** is associated with higher churn risk
- **Fiber optic internet** tends to increase churn risk
- **Two-year contracts** reduce churn risk
- customers with **no online security** or **no tech support** are more likely to churn
- **Electronic check** payment method is associated with higher churn risk

Generated figures:

- `outputs/figures/shap_summary_beeswarm.png`
- `outputs/figures/shap_summary_bar.png`

## Model Performance

Example Random Forest confusion matrix:

```text
[[920 113]
 [179 195]]
```

This corresponds to:

- True Negatives = 920
- False Positives = 113
- False Negatives = 179
- True Positives = 195

Approximate default-threshold results:

- Accuracy: ~79%
- Recall for churn: ~52%

After threshold tuning to `0.35`, recall for churn improved substantially, making the model more useful for retention-focused business scenarios.

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── raw/
│       └── telco_churn.csv
├── models/
│   └── churn_model.pkl
├── notebooks/
│   └── 01_eda.ipynb
├── outputs/
│   ├── figures/
│   │   ├── shap_summary_bar.png
│   │   └── shap_summary_beeswarm.png
│   └── metrics/
├── src/
│   ├── data_prep.py
│   ├── explain.py
│   ├── predict.py
│   └── train.py
├── app.py
├── README.md
└── requirements.txt
```

## How to Run

### 1. Clone the repository

```bash
git clone <https://github.com/ParamShelar/customer-churn-prediction>
cd customer-churn-prediction
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python -m src.train
```

### 5. Generate SHAP plots

```bash
python -m src.explain
```

### 6. Run the Streamlit app

```bash
python -m streamlit run app.py
```

## Streamlit App

The app:

- loads the saved churn model
- accepts customer information as input
- predicts churn probability
- applies the tuned business threshold
- displays whether the customer is likely to churn

## Technologies Used

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- SHAP
- joblib
- Streamlit

## Portfolio Highlights

This project demonstrates:

- end-to-end ML workflow design
- leakage-safe preprocessing with pipelines
- model evaluation beyond accuracy
- decision-threshold tuning for business use cases
- model explainability with SHAP
- deployment with Streamlit

## Future Improvements

Possible next improvements:

- compare additional models such as XGBoost or Gradient Boosting
- perform cross-validation and hyperparameter search
- add probability calibration
- show customer-level SHAP explanations inside the app
- deploy publicly with Streamlit Community Cloud
- add screenshots of the app and SHAP plots to this README

## Author

Param Shelar
