import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Set page configuration
st.set_page_config(page_title="Customer Salary Prediction App", layout="centered")

# Title and description
st.title("Customer Salary Prediction")
st.write("Enter customer details to predict their estimated salary.")

# Load the saved model, encoders, and scaler
@st.cache_resource
def load_model_and_encoders():
    model = tf.keras.models.load_model('regression_model.h5')
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

# Create input form
with st.form("input_form"):
    st.subheader("Customer Information")
    
    # Numeric inputs
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Balance ($)", min_value=0.0, value=0.0)
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox("Has Credit Card?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Is Active Member?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    exited = st.selectbox("Has Exited?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Categorical inputs
    gender = st.selectbox("Gender", options=["Male", "Female"])
    geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Salary")

# Process input and make prediction
if submitted:
    try:
        # Create a dictionary with input data
        input_data = {
            'CreditScore': credit_score,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'Exited': exited,
            'Geography': geography
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
        geo_encoded = onehot_encoder_geo.transform(input_df[['Geography']]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        
        # Combine encoded features and drop original Geography column
        input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)
        
        # Ensure the input DataFrame has the same column order as during training
        expected_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                            'HasCrCard', 'IsActiveMember', 'Exited', 'Geography_France',
                            'Geography_Germany', 'Geography_Spain']
        input_df = input_df[expected_columns]
        
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0][0]
        
        # Since the model uses sigmoid activation, scale the output if necessary
        # Assuming EstimatedSalary was scaled during training, we interpret the output
        st.success(f"Predicted Estimated Salary: ${prediction * 200000:.2f}")
        # Note: The scaling factor (200000) is an assumption; adjust based on your dataset's salary range
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please ensure all inputs are valid and the model files are correctly loaded.")

# Footer
st.markdown("---")
st.write("Developed with Streamlit. Model trained on customer churn data.")