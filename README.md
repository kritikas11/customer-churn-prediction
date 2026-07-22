# Customer Churn Prediction

An Artificial Neural Network (ANN) that predicts whether a bank customer is likely to churn, with SHAP-based explainability so predictions aren't just a black-box number.

🔗 **Live demo:** _add your Streamlit Cloud link here once deployed_

## Overview

Built using the [Bank Customer Churn dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling), this project trains a feedforward neural network to classify whether a customer will leave the bank, based on features like credit score, geography, age, balance, tenure, and number of products held.

## Why SHAP?

Most churn prediction tutorials stop at a probability score. This project goes a step further — for every prediction, it shows *which features* pushed the model toward "will churn" vs. "will stay," using SHAP (SHapley Additive exPlanations). This makes the model's reasoning inspectable rather than opaque, which matters in real-world use cases where a bank would want to know *why* a customer is flagged as high-risk before acting on it.

## Tech Stack

- **Model:** TensorFlow / Keras (Sequential ANN, Dense layers, ReLU/sigmoid activations)
- **Explainability:** SHAP (GradientExplainer)
- **Preprocessing:** scikit-learn (LabelEncoder, OneHotEncoder, StandardScaler)
- **App/Deployment:** Streamlit
- **Experiment tracking:** TensorBoard

## Features Used

CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

## How It Works

1. User inputs customer details via the Streamlit interface
2. Inputs are encoded (Label/One-Hot) and scaled using the same transformers fit during training
3. The trained ANN outputs a churn probability
4. SHAP computes each feature's contribution to that specific prediction, visualized as a bar chart

## Running Locally

```bash
git clone https://github.com/kritikas11/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── app.py                      # Streamlit app
├── experiments.ipynb           # Model training, tuning, TensorBoard logging
├── prediction.ipynb            # Standalone prediction testing
├── model.h5                    # Trained ANN
├── scaler.pkl                  # Fitted StandardScaler
├── label_encoder_gender.pkl    # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl      # Fitted OneHotEncoder for Geography
├── X_train_sample.pkl          # Background sample used by SHAP
└── requirements.txt
```

## What I'd Improve Next

- Add input validation/bounds in the UI to keep values within realistic ranges
- Cache-optimize the SHAP explainer for faster repeated inference
- Explore feature engineering around NumOfProducts, given its counterintuitive relationship with churn in this dataset