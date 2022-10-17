import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

pickle_in = open('C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Code\\tuned_xgb.pkl', 'rb')
classifier = pickle.load(pickle_in)
  
def welcome():
    return 'welcome all'
  
# defining the function which will make the prediction using 
# the data which the user inputs
def prediction(Age, Gender, Married, Employment_Type, Urban_Rural, Smoker, BodyMassIndex, Hypertension, Had_Heart_Disease, Mean_Glucose_Level):  
   
    prediction = classifier.predict(
        [[Age, Gender, Married, Employment_Type, Urban_Rural, Smoker, BodyMassIndex, Hypertension, Had_Heart_Disease, Mean_Glucose_Level]])
    print(prediction)
    return prediction
      
  
# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    st.title("Stroke Probability Prediction")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Stroke Probability Prediction XGBoost Model </h1>
    </div>
    """
    
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction

    Age = st.number_input("Age", 0, 100)
    Gender = st.radio("Gender", ["Male", "Female", "Other"])
    Married = st.radio("Married", ["Yes", "No"])
    Employment_Type = st.selectbox("Employement Type", ["Private", "children", "Govt_job", "Self-employed", "Never_woked"])
    Urban_Rural = st.radio("Urban Rural", ["Urban", "Rural"])
    Smoker = st.selectbox("Smoker", ["never smoked", "formerly smoked", "smokes"])
    BodyMassIndex = st.number_input("BMI", 0, 50)
    Hypertension = st.number_input("Hypertension", 0, 1)
    Had_Heart_Disease = st.number_input("Had Heart Disease", 0, 1)
    Mean_Glucose_Level = st.number_input("Mean Glucose Level", 40, 300)
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(Age, Gender, Married, Employment_Type, Urban_Rural, Smoker, BodyMassIndex, Hypertension, Had_Heart_Disease, Mean_Glucose_Level)
    st.success('The output is {}'.format(result))
     
if __name__=='__main__':
    main()