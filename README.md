# Customer Churn Prediction System

An end-to-end machine learning system that predicts customer churn for a telecom company. It serves real-time predictions using FastAPI and provides an interactive web dashboard using Streamlit, featuring real-time AI-generated retention strategies and model explainability.

This project demonstrates the full machine learning lifecycle, including data preprocessing, feature engineering, model training and evaluation, and deployment as a production-style application.

## Problem Statement

Customer churn occurs when customers discontinue a service, leading to direct revenue loss. The objective of this project is to predict whether a customer is likely to churn so that proactive, targeted retention strategies can be applied before they leave.

## Dataset

* Source: Telco Customer Churn Dataset (IBM Sample Data)
* Target variable: Churn (0 = No, 1 = Yes)
* Class distribution: Approximately 27% churners

## Technical Architecture

The application is split into a robust backend architecture and an interactive frontend:
* **Backend (FastAPI)**: Serves the primary machine learning inference via a strict REST API.
* **Frontend (Streamlit)**: A comprehensive business dashboard allowing dynamic inputs, visualizations, and automated AI strategy generation.
* **Explainability (SHAP)**: Provides local, prediction-level explanations to build trust in the model outputs (waterfall charts).
* **Generative AI (Groq/OpenAI)**: Automatically drafts personalized retention emails for high-risk customers based on their profile.

## Approach & Model Selection

### Data Processing & Feature Engineering
* Used scikit-learn Pipelines to prevent data leakage.
* Preserved raw categorical features for pipeline-based preprocessing.
* Applied standard scaling for numerical features and one-hot encoding for categorical features.
* Ensured identical preprocessing during training and inference.

### Model Selection
Selected **Gradient Boosting** as the final model due to:
* Strong recall on the churn class.
* Higher ROC-AUC score compared to Logistic Regression and Random Forests.
* Ability to model non-linear relationships.

### Model Performance
* Recall (churn class): ~77%
* ROC-AUC: ~0.83
* Accuracy: ~80%

These metrics provide a balanced trade-off between churn detection and false positives. The priority was capturing potential churners (Recall) due to the higher business cost of missed interventions.

## Project Structure

```
ml-churn-prediction-system/
├── app/              # FastAPI application
├── src/              # Data processing, features, models
├── notebooks/        # EDA, training, evaluation
├── models/           # Saved trained models
├── data/             # Dataset
├── streamlit_app.py  # Streamlit Frontend application
├── requirements.txt  # Core dependencies
└── README.md
```


## Future Improvements

* Add monitoring for prediction drift.
* Retrain the model dynamically with more recent customer data.
* Add batch prediction endpoints (CSV upload).
* Store historical predictions in a database for accuracy tracking over time.

## Summary

This project demonstrates how to build a production-oriented machine learning system, not just a standalone model. It emphasizes reproducibility, evaluation aligned with business goals, and deployability using modern APIs, explainable AI, and generative AI features.
