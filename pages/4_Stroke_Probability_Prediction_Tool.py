import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

st.title("Stroke Probability Prediction Tool")

st.markdown("""
    
You can select a predictive mode, input your data here below to compute a stroke probability prediction. 

**The Different Models:**

* Random Forest Classifier:
parameters: 'n_estimators': 1200, 'min_samples_split': 70, 'min_samples_leaf': 18,
'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': True. 

Documentation : **Scikit-Learn**: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

You can also modify the inputs to see how the probability would evolve. 

* Gradient Boosting Classifier:
parameters: 'n_estimators': 300, 'min_samples_split': 20, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 3, 'learning_rate': 0.01

Documentation : **Scikit-Learn**: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

**Disclaimer** : This Project and the Stroke Probability Prediction Tool are for educational purpose only.

""")

st.subheader("Choose Model Here-Under")
# Choose Predictive Model Here-Under
Model = st.selectbox("Models", ["Random Forest Classifier", "Gradient Boosting Classifier"])
pickle_in = open('C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Code\\RFC.pkl', 'rb')
pickle_in2 = open('C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Code\\GBC.pkl', 'rb')

Selected_Model=pickle_in

if Model=="Random Forest Classifier":
    Selected_Model=pickle_in
elif Model=="Gradient Boosting Classifier":
    Selected_Model=pickle_in2

classifier = pickle.load(Selected_Model)

# this is the main function in which we define our webpage 
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
   
    Gender_Male = Gender_Female = Gender_Other = Married_No = Married_Yes = Employment_Type_Govt_job = Employment_Type_Never_worked = 0
    Employment_Type_Private = Employment_Type_Self_employed = Employment_Type_children = Urban_Rural_Rural = 0
    Urban_Rural_Urban = Smoker_Unknown = Smoker_formerly_smoked = Smoker_Never_smoked = Smoker_smokes = 0

    if Gender =="Male":
        Gender_Male=1
    elif Gender =="Female":
        Gender_Female=1
    elif Gender =="Other":
        Gender_Other=1
    
    if Hypertension =="Yes":
        Hypertension=1
    elif Hypertension =="No":
        Hypertension=0
        
    if Had_Heart_Disease =="Yes":
        Had_Heart_Disease=1
    elif Had_Heart_Disease =="No":
        Had_Heart_Disease=0

    if Married =="Yes":
        Married_Yes=1
    elif Married =="No":
        Married_No=1

    if Employment_Type =="Private":
        Employment_Type_Private=1
    elif Employment_Type =="Self_employed":
        Employment_Type_Self_employed=1
    elif Employment_Type =="Children":
        Employment_Type_children=1
    elif Employment_Type =="Never Worked":
        Employment_Type_Never_worked=1
    elif Employment_Type =="Government Job":
        Employment_Type_Govt_job=1
    
    if Urban_Rural =="Urban":
        Urban_Rural_Urban=1
    elif Urban_Rural =="Rural":
        Urban_Rural_Rural=1

    if Smoker =="Unknown":
        Smoker_Unknown=1
    elif Smoker =="Formerly Smoked":
        Smoker_formerly_smoked=1
    elif Smoker =="Never Smoked":
        Smoker_Never_smoked=1
    elif Smoker == "Smokes":
        Smoker_smokes=1

    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    X = pd.DataFrame({'Age':[Age],
                      'BodyMassIndex':[BodyMassIndex], 
                      'Hypertension':[Hypertension], 
                      'Had_Heart_Disease':[Had_Heart_Disease], 
                      'Mean_Glucose_Level':[Mean_Glucose_Level],
                      'Gender_Female':[Gender_Female],
                      'Gender_Male':[Gender_Male],
                      'Gender_Other':[Gender_Other],
                      'Married_No':[Married_No], 
                      'Married_Yes':[Married_Yes], 
                      'Employment_Type_Govt_job':[Employment_Type_Govt_job],
                      'Employment_Type_Never_worked':[Employment_Type_Never_worked], 
                      'Employment_Type_Private':[Employment_Type_Private],
                      'Employment_Type_Self_employed':[Employment_Type_Self_employed], 
                      'Employment_Type_children':[Employment_Type_children], 
                      'Urban_Rural_Rural':[Urban_Rural_Rural], 
                      'Urban_Rural_Urban':[Urban_Rural_Urban],
                      'Smoker_Unknown':[Smoker_Unknown],
                      'Smoker_formerly_smoked':[Smoker_formerly_smoked], 
                      'Smoker_Never_smoked':[Smoker_Never_smoked], 
                      'Smoker_smokes':[Smoker_smokes]
                     })   
    
    if st.button("Predict"):
        result = classifier.predict_proba(X)[:,1]
    st.success('Stroke Probability: {}'.format(result))
     
if __name__=='__main__':
    main()