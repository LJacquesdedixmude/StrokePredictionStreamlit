from winreg import HKEY_LOCAL_MACHINE
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiapp import MultiApp
from apps import home, dataprep

app = MultiApp()

st.title("Stroke Prediction Project")

st.markdown("Stroke, (un accident vasculaire cérébral (AVC) in French and cerebrovasculair accident (CVA) in Dutch), is a sudden neurological deficit of vascular origin caused by an infarction or hemorrhage in the brain. In French and Dutch the term ""accident"" emphasizes the sudden or abrupt nature of the symptoms, but in most cases the causes are internal (e.g., age, diet, lifestyle). The World Health Organization (WHO) has found that stroke is the second leading cause of death globally. Actually, stroke is responsible for approximately 11% of total deaths.")

st.markdown("In this project we will train different Machine Learning Models to predict probabilities of having a stroke based on health related attributes (e.g. Age, Gender, BMI,...). We will here present the DataSet, the models, and an interface where you can enter your own health related data and have the (best) model predict your risk of having a stroke")
app.add_app("Home", home.app)
app.add_app("Data prep", dataprep.app)
app.run()

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

st.write("""

###  Comments:

* In the Train DS, Gender ( Male / Female / Other )
* Once the best model found retrain model on entire training DS
* Rather small dataset (+/- 4000 Rows)

### Potential complications:

* Need to handle NaN values in BMI ==> Delete? Use Mean? Use Median? Other Methods...
* Need to handle categorical and numerical values
* Unbalanced dataset can mislead models (very few strokes)
* They are no "Other" gender in the test dataset)

""")

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

st.subheader(" Data Preparation")
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

