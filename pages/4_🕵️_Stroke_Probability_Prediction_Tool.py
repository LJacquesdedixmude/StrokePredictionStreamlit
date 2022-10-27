import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import shap
shap.initjs()

st.title("Stroke Probability Prediction Tool")

st.markdown("""
    
Select a model in the dropdown, input your data here below & the model will predict your stroke probability. 

You can also modify the inputs to see how the stroke probability evolves. 

**Disclaimer** : This Project and the Stroke Probability Prediction Tool are for educational purpose only.

""")

st.subheader("Choose Model Here-Under")
# Choose Predictive Model Here-Under
Model = st.selectbox("Models", ["Random Forest Classifier", "Gradient Boosting Classifier", "Logistic Regression"])
pickle_in = open('Code\\RFC.pkl', 'rb')
pickle_in2 = open('Code\\GBC.pkl', 'rb')
pickle_in3 = open('Code\\LR.pkl', 'rb')

Selected_Model=pickle_in

if Model=="Random Forest Classifier":
    Selected_Model=pickle_in
    st.markdown("""
Parameters: 'n_estimators': 1200, 'min_samples_split': 70, 'min_samples_leaf': 18, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': True. 

Documentation : **Scikit-Learn**: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
""")
elif Model=="Gradient Boosting Classifier":
    Selected_Model=pickle_in2
    st.markdown("""
Parameters: 'n_estimators': 300, 'min_samples_split': 20, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 3, 'learning_rate': 0.01

Documentation : **Scikit-Learn**: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    """)
elif Model=="Logistic Regression":
    Selected_Model=pickle_in3
    st.markdown("""
Parameters:   C=0.602, class_weight={}, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=1000, multi_class='auto', n_jobs=None, penalty='l2', solver='lbfgs', tol=0.0001, verbose=0,warm_start=False

Documentation : **Scikit-Learn**: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
""")

classifier = pickle.load(Selected_Model)

# this is the main function in which we define our webpage 
def main():
        
    Age = st.number_input("Age", 0.01, 100.01)
    Gender = st.radio("Gender", ["Male", "Female", "Other"])
    Married = st.radio("Married", ["Yes", "No"])
    Employment_Type = st.selectbox("Employement Type", ["Private", "Children", "Government Job", "Self Employed", "Never Worked"])
    Urban_Rural = st.radio("Urban Rural", ["Urban", "Rural"])
    Smoker = st.selectbox("Smoker", ["Unknown", "Never Smoked", "Formerly Smoked", "Smokes"])
    BodyMassIndex = st.number_input("BMI", 0.01, 50.01, value=20.00,help="Body Mass Index Formula = Weight (in Kg) / Height (Meter)^2")
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
    result=0    
    if st.button("Predict"):
        result = classifier.predict_proba(X)[:,1]
        st.markdown("#### Stroke Probability: %.2f%%" % (result[0]*100))
        # if Model=="Random Forest Classifier":
            # explainer = shap.TreeExplainer(classifier, X)
            # choosen_instance = X
            # shap_values = explainer.shap_values(X)
            # st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))
            # st.write(shap_values)
            # st.write(shap.force_plot(explainer.expected_value, shap_values[:,:], matplotlib=True), unsafe_allow_html=True)

    if result ==0:
        st.write("Input your data")
    elif result > 0 and result <0.05:
        st.success("""
        
        ### Good News! Low Stroke Probability

        * Looks like you're in good shape 😀 """)
    elif result >= 0.05 and result < 0.15:
        st.warning("""
        
        ### Carefull. Some Stroke Risks 
         
        * Think about counsulting your doctor or a specialist 🤨 """)
    elif result >= 0.15:
        st.error("""
        
        ### Important Stroke Risk! 
        
        * Think about consulting your doctor or a specialist. """)
     
if __name__=='__main__':
    main()