# Telecom Churn Prediction using Random Forest

A machine learning project that applies Ensemble Learning techniques—specifically Random Forest—to predict customer attrition in the telecom sector by combining the insights of multiple decision trees.

## Overview

This project explores the power of "Ensemble Learning" to solve the complex problem of customer churn. By leveraging a Random Forest classifier, we aim to improve predictive performance and model stability over a single decision tree. The analysis investigates customer demographics, service usage, and contract details to identify high-risk individuals likely to cancel their subscriptions.

## Dataset

- **Source:** Telecom Customer Churn dataset (`21_customer_churn.csv`).
- **Key columns:**
  - `tenure`: Number of months the customer has stayed with the company.
  - `MonthlyCharges`: The amount charged to the customer monthly.
  - `TotalCharges`: The total amount charged to the customer.
  - `Contract`: The contract term (Month-to-month, One year, Two year).
  - `InternetService`: Type of internet (Fiber optic, DSL, or No service).
  - `Churn`: Target variable indicating if the customer left (Yes/No).

## Objectives

- Understand the concept of **Ensemble Learning** and how Random Forest aggregates multiple models.
- Preprocess categorical features and handle numerical scaling for machine learning.
- Implement a **Random Forest Classifier** to handle complex, non-linear relationships in churn data.
- Evaluate the model's effectiveness using an **Accuracy Score** and **Confusion Matrix**.

## Methods and Analysis

The project follows a systematic machine learning pipeline:

- **Data Loading and Exploration**
  - Initial inspection of the 7,043 customer records.
  - Identification of features ($X$) like `tenure` and `MonthlyCharges` and the target label ($y$).

- **Feature Engineering**
  - Splitting data into Training (80%) and Testing (20%) sets to ensure unbiased evaluation.
  - Preparing data for the Ensemble model by ensuring numerical compatibility.

- **Random Forest Implementation**
  - Training a `RandomForestClassifier` which creates a "forest" of decision trees.
  - Utilizing the **Bootstrap Aggregating (Bagging)** technique to reduce variance and avoid overfitting.



- **Model Evaluation**
  - Generating a **Confusion Matrix** to visualize the balance between True Positives and True Negatives.
  - Calculating the **Accuracy Score** to determine the overall percentage of correct predictions.



## Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas` and `numpy`: Data manipulation and numerical processing.
  - `matplotlib` and `seaborn`: Visualization of churn trends.
  - `scikit-learn`: Random Forest model, data splitting, and performance metrics.
- **Environment:** Jupyter / Google Colab

## How to Run

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/telecom-churn-random-forest.git
   cd telecom-churn-random-forest

2. *Create and activate a virtual environment (optional but recommended):*
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. *Install dependencies:*
   pip install pandas numpy seaborn matplotlib scikit-learn

4.  Ensure the dataset is present: Place 21_customer_churn.csv in the root folder.

5. *Open the notebook:*
   jupyter notebook 21_Random_Forest_Handson.ipynb
