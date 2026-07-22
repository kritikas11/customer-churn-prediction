import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('X_train_sample.pkl', 'rb') as file:
    X_train_sample = pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
balance = st.number_input('Balance', min_value=0.0, max_value=250000.0, value=50000.0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=200000.0, value=60000.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# SHAP explainability
explainer = shap.GradientExplainer(model, X_train_sample)
shap_values = explainer.shap_values(input_data_scaled)

# shap_values comes back as an array; grab the array for our single output
shap_vals_for_input = shap_values[0, :, 0]  # first output, first (only) row

# Build a dataframe pairing each feature with its SHAP value
shap_df = pd.DataFrame({
    'Feature': input_data.columns,
    'SHAP Value': shap_vals_for_input
}).sort_values(by='SHAP Value', key=abs, ascending=True)

st.subheader('Why did the model predict this?')
fig, ax = plt.subplots()
colors = ['crimson' if v > 0 else 'steelblue' for v in shap_df['SHAP Value']]
ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
ax.set_xlabel('Impact on churn prediction')
st.pyplot(fig)

st.caption('🔴 Red = pushes toward churn | 🔵 Blue = pushes toward staying')


st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
