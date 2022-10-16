from winreg import HKEY_LOCAL_MACHINE
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiapp import MultiApp
from apps import home, dataprep


def app(): 
    
    st.header("Data Information")

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

    st.subheader("Full Dataset")
    st.write("Here is the raw training dataset before any data manipulation")
    st.dataframe(pd.read_csv('C:/Users/ljacquesdedixmude/streamlit_SP/Data/training-stroke.csv'))

    st.write("Description of the dataset for the attributes with numerical values. We can see that they are some missing values for the BMI")
    df = pd.read_csv('C:/Users/ljacquesdedixmude/streamlit_SP/Data/training-stroke.csv')
    st.dataframe(df.describe())

    def main():
        st.write("## Attributes Distributions")
        st.write("Select an attribute to plot its distribution")
        page = st.selectbox(
            "Attributes",
            [
                "Age Plot",
                "BMI Plot",
                "Employment",
                "Glucose Level", 
                "Hypertension",
                "Smoker",
                "Stroke"
            ]
        )

        fig = plt.figure(figsize=(10, 4))

        if page == "Age Plot":
            sns.histplot(x = "Age", data = df, hue='Gender', multiple='stack')
        
        if page == "Smoker":
            sns.countplot(x = "Smoker", data = df, hue='Gender')
        
        if page == "Employment":
            sns.countplot(x = "Employment_Type", data = df, hue='Gender')

        if page == "Stroke":
            sns.countplot(x = "Stroke", data = df, hue='Gender')
    
        if page == "Hypertension":
            sns.countplot(x = "Hypertension", data = df, hue='Gender')

        if page == "Glucose Level":
            sns.histplot(x = "Mean_Glucose_Level", data = df, hue='Gender', multiple='stack')

        elif page == "BMI Plot":
            sns.histplot(x = "BodyMassIndex", data = df, hue='Gender', multiple='stack')

        st.pyplot(fig)

    if __name__ == "__main__":
        main()

        st.subheader("Correlation Matrix")
        d2 = pd.get_dummies(df, columns = ['Gender', 'Married', 'Employment_Type', 'Urban_Rural', 'Smoker'])
        d2 = d2.fillna(28.9)

        fig, ax = plt.subplots()
        cmap = sns.color_palette("crest", as_cmap=True)
        sns.heatmap(df.corr(),cmap = cmap, ax=ax, annot=True)
        st.write(fig)

