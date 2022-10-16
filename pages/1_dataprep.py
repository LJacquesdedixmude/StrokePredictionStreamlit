import streamlit as st
import matplotlib
import pandas as pd

def app(): 
    st.subheader(" Data Preparation")

    st.write("""

    We need to handle the missing values in the BodyMassIndex column. They 
    are several options here we will fill the missing values with the mean value (28.89). 
    As they are few missing values (160 missing values for 4088 values total) this shouldn't
    mislead the models. For some models we will have to create dummy variables for the 
    categorical attributes ("Gender", "Married", "Employement_Type", "Urban_Rual", "Smoker"). When using
    PyCaret Library we don't need to as it handles categorical variables.

    """)