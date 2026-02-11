# Customer Churn Prediction System

An end-to-end machine learning system that predicts customer churn for a telecom company and serves real-time predictions using FastAPI.

This project demonstrates the full machine learning lifecycle, including data preprocessing, feature engineering, model training and evaluation, and deployment as a production-style API.

## Problem Statement

Customer churn occurs when customers discontinue a service, leading to direct revenue loss.
The objective of this project is to predict whether a customer is likely to churn so that proactive retention strategies can be applied.

## Dataset

Source: Telco Customer Churn Dataset (IBM Sample Data)

Target variable: Churn (0 = No, 1 = Yes)

Class distribution: Approximately 27% churners

## Approach
Data Processing

Converted numeric columns such as TotalCharges to appropriate data types

Removed invalid and missing records

Preserved raw categorical features for pipeline-based preprocessing

Feature Engineering and Pipelines

Used scikit-learn Pipelines to prevent data leakage

Applied standard scaling for numerical features

Applied one-hot encoding for categorical features

Ensured identical preprocessing during training and inference

## Model Training and Evaluation

Trained and evaluated multiple models:

Logistic Regression

Random Forest

Gradient Boosting

Focused on recall for churners due to higher business cost of false negatives

Tuned the decision threshold to improve churn detection

## Model Selection

Selected Gradient Boosting as the final model due to:

Strong recall on churn class

Higher ROC-AUC score

Ability to model non-linear relationships

Model Performance

Recall (churn class): ~77%

ROC-AUC: ~0.83

Accuracy: ~80%

These metrics provide a balanced trade-off between churn detection and false positives.

## FastAPI Deployment

The trained model is deployed using FastAPI for real-time inference.

API Features

Accepts raw customer data as JSON

Uses the trained preprocessing and model pipeline

Returns churn probability and classification

Strict input validation using Pydantic schemas

Interactive Swagger UI for testing

## How to Run the Project
1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Start the FastAPI server
uvicorn app.main:app --reload

4. Access Swagger UI
http://127.0.0.1:8000/docs

Example API Request
{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 850.0,
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "InternetService": "Fiber optic",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "PaperlessBilling": "Yes"
}

Example API Response
{
  "churn_probability": 0.56,
  "churn_prediction": 1,
  "threshold": 0.3
}

Input Validation and Safety

Enforced valid categorical values through strict schemas

Applied numeric constraints on input features

Rejected invalid requests before model execution

This ensures reliable and safe predictions.

## Project Structure
ml-churn-prediction-system/
├── app/ # FastAPI application
├── src/ # Data processing, features, models
├── notebooks/ # EDA, training, evaluation
├── models/ # Saved trained models
├── data/ # Dataset
├── requirements.txt
└── README.md

## Future Improvements

Add monitoring for prediction drift

Retrain model with recent customer data

Deploy using Docker or cloud services

Add batch prediction endpoints

## Summary

This project demonstrates how to build a production-oriented machine learning system, not just a standalone model. It emphasizes reproducibility, evaluation aligned with business goals, and deployability using modern APIs.
