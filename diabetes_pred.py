import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi

# # Connect to the MongoDB cluster to store the inputs and the prediction
# uri = "mongodb+srv://*******@cluster0.ugo9l.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# client = MongoClient(uri, server_api=ServerApi('1'))
# db = client['diabetes']  # Create a new database
# collection = db['diabetes_pred'] # Create a new collection/table in the database

def load_ridge_model():
    with open('diabetes_ridge_final_model.pkl', 'rb') as f:
        ridge_model, scalar = pickle.load(f)
    
    return ridge_model, scalar

def load_lasso_model():
    with open('diabetes_lasso_final_model.pkl', 'rb') as f:
        lasso_model, scalar = pickle.load(f)
    
    return lasso_model, scalar

def preprocesssing_input_data(data, scalar):
    df = pd.DataFrame([data])
    df_transformed = scalar.transform(df)

    return df_transformed

def predict_data(data):
    ridge_model, scalar = load_ridge_model()
    lasso_model, scalar = load_lasso_model()

    processed_data = preprocesssing_input_data(data, scalar)

    ridge_prediction = ridge_model.predict(processed_data)
    lasso_prediction = lasso_model.predict(processed_data)

    return ridge_prediction, lasso_prediction

def main():
    st.title("Diabetes Prediction App")
    st.write("Enter the data to get a prediction for diabetes")
    
    age = st.number_input("Age", min_value = 0, max_value = 100, value = 18)
    sex = st.number_input("Sex/Gender")
    bmi = st.number_input("BMI (Body Mass Index)")
    blood_pressure = st.number_input("ABP (Average Blood Pressure)")
    s1 = st.number_input("S1 (Total Serum Cholesterol)")
    s2 = st.number_input("S2 (Low-Density Lipoproteins)")
    s3 = st.number_input("S3 (High-Density Lipoproteins)")
    s4 = st.number_input("S4 (Total Cholesterol/HDL)")
    s5 = st.number_input("S5 (Possibly Log of Serum Triglycerides Level)")
    s6 = st.number_input("S6 (Blood Sugar Level)")

    if st.button("Predict the diabetes"):
        user_data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "bp": blood_pressure,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
            "s5": s5,
            "s6": s6
        }

        prediction_ridge, prediction_lasso = predict_data(user_data)
        st.success(f"Your prediction result using Ridge Regression is: {round(float(prediction_ridge[0]), 3)}")
        st.success(f"Your prediction result using Lasso Regression is: {round(float(prediction_lasso[0]), 3)}")

        # user_data["prediction_ridge"] = round(float(prediction_ridge[0]), 3)    # Add the ridge prediction to the user_data dictionary
        # user_data["prediction_lasso"] = round(float(prediction_lasso[0]), 3)    # Add the lasso prediction to the user_data dictionary
        # user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, float) else value for key, value in user_data.items()}    # Convert the values to int or float if they are of type np.integer or np.float
        # collection.insert_one(user_data)    # Insert the user_data dictionary as a record to the MongoDB collection

if __name__ == "__main__":
    main()