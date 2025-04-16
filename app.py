#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
import streamlit as st
import joblib
import numpy as np

# Load the saved pipeline
model = joblib.load("rf_smote_pipeline.pkl")

st.title("🫁 Lung Cancer Prediction App")
st.markdown("Enter patient information below to predict lung cancer risk.")

# Define categorical options based on your dataset
gender_options = ['F', 'M']
yes_no_options = ['YES', 'NO']
age_range = list(range(20, 90))  # Adjust based on your data

# Input form
age = st.selectbox("Age", age_range)
gender = st.selectbox("Gender", gender_options)
smoking = st.selectbox("Smoking", yes_no_options)
yellow_fingers = st.selectbox("Yellow Fingers", yes_no_options)
anxiety = st.selectbox("Anxiety", yes_no_options)
peer_pressure = st.selectbox("Peer Pressure", yes_no_options)
chronic_disease = st.selectbox("Chronic Disease", yes_no_options)
fatigue = st.selectbox("Fatigue", yes_no_options)
allergy = st.selectbox("Allergy", yes_no_options)
wheezing = st.selectbox("Wheezing", yes_no_options)
alcohol_consuming = st.selectbox("Alcohol Consuming", yes_no_options)
coughing = st.selectbox("Coughing", yes_no_options)
shortness_of_breath = st.selectbox("Shortness of Breath", yes_no_options)
swallowing_difficulty = st.selectbox("Swallowing Difficulty", yes_no_options)
chest_pain = st.selectbox("Chest Pain", yes_no_options)

# Map categorical inputs to 0/1
def binary(val):
    return 1 if val == 'YES' else 0

def gender_to_binary(val):
    return 1 if val == 'M' else 0

# Create input array in order
input_data = np.array([[
    age,
    gender_to_binary(gender),
    binary(smoking),
    binary(yellow_fingers),
    binary(anxiety),
    binary(peer_pressure),
    binary(chronic_disease),
    binary(fatigue),
    binary(allergy),
    binary(wheezing),
    binary(alcohol_consuming),
    binary(coughing),
    binary(shortness_of_breath),
    binary(swallowing_difficulty),
    binary(chest_pain)
]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.error(f"⚠️ High Risk of Lung Cancer (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Lung Cancer (Probability: {prob:.2f})")

