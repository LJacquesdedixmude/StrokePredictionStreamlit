import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image

st.title("The Models")

st.write("""

In this part the different models trained and their results will be presented.

One of the first step after data preparation in Machine Learning is to split the dataSet into a training and a testing dataset. 
Here we will use 30% of our dataset for testing and keep 70% for training the models.

""")



st.subheader("Random Forest Classifier")

st.write("""

réf. Wikipedia. 

Random forests or random decision forests is an ensemble learning method for classification, 
regression and other tasks that operates by constructing a multitude of decision trees at training time.
For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, 
the mean or average prediction of the individual trees is returned. Random decision forests correct for decision trees' 
habit of overfitting to their training set. Random forests generally outperform decision tree. However, data characteristics 
can affect their performance.

We will start by fitting a model with default parameters. We will look at the result then try to 
tune the hyperparamers with gridsearch to improve model performance.

""")


img = Image.open("AUCRFC.png")
st.image(img, caption='Enter any caption here')