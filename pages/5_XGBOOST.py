import pandas as pd
import numpy as np
import streamlit as st
from pycaret.classification import *
from xgboost import *

model = load_model('C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Code\\Tuned_XGB')

def predict(model, input_df):
    prediction_df = predict_model(estimator=model, data=input_df)
    return prediction_df

def main():
        
    Age = st.number_input("Age", 0.01, 100.01)
    Gender = st.radio("Gender", ["Male", "Female", "Other"])
    Married = st.radio("Married", ["Yes", "No"])
    Employment_Type = st.selectbox("Employement Type", ["Private", "Children", "Government Job", "Self Employed", "Never Worked"])
    Urban_Rural = st.radio("Urban Rural", ["Urban", "Rural"])
    Smoker = st.selectbox("Smoker", ["Unknown", "Never Smoked", "Formerly Smoked", "Smokes"])
    BodyMassIndex = st.number_input("BMI", 0.01, 50.01, help="Body Mass Index Formula = Weight (in Kg) / Height (Meter)^2")
    Hypertension = st.radio("Hypertension", ["Yes", "No"])
    Had_Heart_Disease = st.radio("Had Heart Disease", ["Yes", "No"])
    Mean_Glucose_Level = st.number_input("Mean Glucose Level", 40.01, 300.01, value=106.02, help="if you don't know your Mean Glucose Level, 106 mg/ dL is an average value. ")
    result =""

    if Hypertension=="Yes":
        Hypertension=1
    else:
        Hypertension=0
    
    if Had_Heart_Disease=="Yes":
        Had_Heart_Disease=1
    else:
        Had_Heart_Disease=0

    input_dict = {'Age': Age, 'Gender' : Gender, 'Married': Married, 'Employment_Type': Employment_Type, 'Uban_Rural': Urban_Rural, 'Smoker': Smoker, 'BodyMassIndex': BodyMassIndex, 'Hypertension': Hypertension, 'Had_Heart_Disease': Had_Heart_Disease, 'Mean_Glucose_Level': Mean_Glucose_Level  }
    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        result = predict(model=model, input_df=input_df)
        st.success('Stroke Probability: {}'.format(result))

if __name__=='__main__':
    main()