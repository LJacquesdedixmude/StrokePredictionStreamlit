import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 

st.set_page_config(
    page_title="StrokePred",
)

st.sidebar.success("Select a page above.")

st.title("Stroke Prediction Project")

st.markdown("Stroke, (un accident vasculaire cérébral (AVC) in French and cerebrovasculair accident (CVA) in Dutch), is a sudden neurological deficit of vascular origin caused by an infarction or hemorrhage in the brain. In French and Dutch the term ""accident"" emphasizes the sudden or abrupt nature of the symptoms, but in most cases the causes are internal (e.g., age, diet, lifestyle). The World Health Organization (WHO) has found that stroke is the second leading cause of death globally. Actually, stroke is responsible for approximately 11% of total deaths.")

st.markdown("In this project we will train different Machine Learning Models to predict probabilities of having a stroke based on health related attributes (e.g. Age, Gender, BMI,...). We will here present the DataSet, the models, and an interface where you can enter your own health related data and have the (best) model predict your risk of having a stroke")

st.subheader("The Data Set")

st.markdown("""

There are 4088 samples and 12 features. Short descriptions of each column are as follows:

- **id**: ID of each patient
- **Age**: Age of the patient in years
- **Gender**: Gender of the patient
- **Married**: Marital status (Yes=ever married, No= have never been married)
- **Employment_Type**: What type of employment the patient currently has (Private, Self-employed, children, Govt_job, Never_worked
- **Urban_Rural**: Where does the patient live (Urban area or Rural area)
- **Smoker**: Did the patient ever smoked or is still smoking? (smokes, formerly smoked, never smoked, Unknown)
- **BodyMassIndex**: The Body Mass Index (BMI) of the patient
- **Hypertension**: Does the patient have hypertension (high blood pressure) (1: yes, 0: no)
- **Had_Heart_Disease**: Does the patient has a heart disease (1: yes, 0: no never had)
- **Mean_Glucose_Level**: Mean glucose level in the blood of the patient
- **Stroke**: Did the patient had a stroke (1=yes, 0=no) Target Column

""")

st.subheader("The raw training dataset before any data manipulation")
st.dataframe(pd.read_csv('Data\\training-stroke.csv'))

st.write("""

###  Comments:

* In the Training DataSet they are 3 different values for Gender: Male / Female / Other.
* Once we found the best model on the training Data Set. (With the Train Test Split). We can re-train the best performing model on the entire dataset before makes predicting Stroke Probabilities on the Test Data.
* With (only) 4,088 values, the training dataset is rather small. Models performance may be improved with a larger dataset.

### Potential complications:

* They are some missing values for the Body Mass Index. Thankfully they aren't too many (only 160 values for 4088 indiviudals)  
They are different ways too handle these missing values. We could data for these indivudals, use the mean or the median value for the NaN values? Or even use other more complex methods.
* Need to handle categorical and numerical values
* The Dataset is strongly unbalanced as they are only few indivuals in the dataset that had a Stroke. 
Consider this, a model that always predict that an indivual will not have a stroke would here have over 95% accuracy. We have to be carefull with our models. 
* They are different techniques to handle unbalanced DataSets. Such as under-samlppling, over-sampling, or SMOTE (Synthetic Minority Over-Sampling Technique) that consists in synthetizing new instances by using K-nn in the minority class (Here individuals that had a Stroke). 
* In the training Data Set they are 3 different Genders: Male / Female / Others, while in the Test Data Set they are only 2 Gender: Male / Female. If we create dummy variables it could create problems.

""")


