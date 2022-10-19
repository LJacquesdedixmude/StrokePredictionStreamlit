import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Data Exploration & First Insights")

st.write("In this part we will discover the dataset and try to find first insights on the datset and on our Stroke Preidction problem. We will plot and look at the distributions of the different attributes to understand the dataset better. We will also look at the correlation matrix ")

df = pd.read_csv('Data\\training-stroke.csv')

st.write("A desciption of the DataSet for the attributes with numerical values.")
st.dataframe(df.describe())

def main():
    st.write("## Attributes Distributions")
    st.write("Select an attribute to plot its distribution. For each plot you can found the count of individuals, Females are in blue, Males in orange and Other in Green.")
    page = st.selectbox(
        "Attributes",
        [
            "Age",
            "Body Mass Index",
            "Employment",
            "Glucose Level", 
            "Hypertension",
            "Smoker",
            "Stroke"
        ]
    )

    fig = plt.figure(figsize=(10, 4))

    if page == "Age":
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

    elif page == "Body Mass Index":
        sns.histplot(x = "BodyMassIndex", data = df, hue='Gender', multiple='stack')

    st.pyplot(fig)

if __name__ == "__main__":
    main()

st.subheader("Correlation Matrix")
st.write("You can find the value for thecorrelation between the different attributes in each box.")
d2 = pd.get_dummies(df, columns = ['Gender', 'Married', 'Employment_Type', 'Urban_Rural', 'Smoker'])
d2 = d2.fillna(28.9)

fig, ax = plt.subplots()
cmap = sns.color_palette("crest", as_cmap=True)
sns.heatmap(df.corr(),cmap = cmap, ax=ax, annot=True)
st.write(fig)

