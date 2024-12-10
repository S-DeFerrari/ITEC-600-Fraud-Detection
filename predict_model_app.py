"""
# My first app
Here's our first attempt at using data to create a table:
"""
import streamlit as st
import joblib
import numpy as np
# from sklearn.ensemble import GradientBoostingClassifier
import sklearn

# Load the pickled model
model = joblib.load('gbm_model.sav')

# Title of the app
st.title("Transaction Classifier")

# User input for the variables
st.header("Input the transaction details:")

distance_from_home = st.number_input(
    "Distance from home (km):", min_value=0.0, max_value=1000.0, step=1.0, value=10.0
)
distance_from_last_transaction = st.number_input(
    "Distance from last transaction (km):", min_value=0.0, max_value=1000.0, step=1.0, value=5.0
)
ratio_to_median_purchase_price = st.number_input(
    "Ratio to median purchase price:", min_value=0.0, max_value=100.0, step=0.1, value=1.0
)
repeat_retailer = st.selectbox("Repeat retailer?", options=["No", "Yes"])
used_chip = st.selectbox("Used chip?", options=["No", "Yes"])
used_pin_number = st.selectbox("Used PIN number?", options=["No", "Yes"])
online_order = st.selectbox("Online order?", options=["No", "Yes"])

# Convert categorical inputs to binary
repeat_retailer = 1 if repeat_retailer == "Yes" else 0
used_chip = 1 if used_chip == "Yes" else 0
used_pin_number = 1 if used_pin_number == "Yes" else 0
online_order = 1 if online_order == "Yes" else 0

# Create a numpy array for prediction
input_features = np.array([
    distance_from_home,
    distance_from_last_transaction,
    ratio_to_median_purchase_price,
    repeat_retailer,
    used_chip,
    used_pin_number,
    online_order
]).reshape(1, -1)

# Predict button
if st.button("Predict"):
    # Make a prediction
    prediction = model.predict(input_features)
    result = "True" if prediction[0] else "False"

    # Display the result
    st.subheader(f"Prediction Result: {result}")



