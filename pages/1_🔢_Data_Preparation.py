import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Data Preparation")

st.write("In this part we will try to prepare the dataset for machine learning. Overall they datset is already pretty clean and they are few operations to be made.")

st.subheader("Description of the dataset for the attributes with numerical values") 
st.write("We can see that they are some missing values for the BMI")
df = pd.read_csv('Data\\training-stroke.csv')
st.dataframe(df.describe())

st.write("""

* Handle Missing Values
We need to handle the missing values in the BodyMassIndex column. 
They are several options here we will fill the missing values with the mean value (28.89). 
As they are few missing values (160 missing values for 4088 values total) this shouldn't
mislead the models.

* Create Dummy variables 
For some models we will have to create dummy variables for the 
categorical attributes ("Gender", "Married", "Employement_Type", "Urban_Rual", "Smoker"). When using
PyCaret Library we don't need to as it handles categorical variables.

""")




    