import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("customer_churn_ann (1).keras")
preprocessor = joblib.load("preprocessor (1).pkl")

st.title("Customer churn Prediction using ANN")
credit_score = st.number_input("Credit Score",min_value=300,max_value=900, value=650)

country = st.selectbox("Country",["France","Germany","Spain"])
gender = st.selectbox("Gender",["Male","Female"])
age = st.number_input("Age",min_value=18,max_value=100,value=35)
tenure = st.number_input("Tenure",min_value=0,max_value=10,value=5)
balance = st.number_input("Balance",value=50000.0)
products_number = st.number_input("Number of products",min_value=1,max_value=4,value=1)
credit_card = st.selectbox("Has Credit Card",[0,1])
active_member = st.selectbox("Is Active Member",[0,1])
estimated_salary = st.number_input("Estimated Salary",value=50000.0)
if st.button("Predict"):
    input_df = pd.DataFrame({
        "credit_score": [credit_score],
        "country": [country],
        "gender": [gender],
        "age" : [age],
        "tenure": [tenure],
        "balance": [balance],
        "products_number": [products_number],
        "credit_card": [credit_card],
        "active_member": [active_member],
        "estimated_salary": [estimated_salary]
    })

    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)
    probability = prediction[0][0]

    if probability >= 0.5:
        st.error(f"Customer is likely to churn\n\nProbability: {probability:.2%}")
    else:
        st.success(f"Customer is likely to stay\n\nProbability: {(1-probability):.2%}")