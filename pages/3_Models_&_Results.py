import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image

st.title("Models & Results")

st.write("""

In this part the different models trained and their results will be presented.

One of the first step after data preparation in Machine Learning is to split the dataSet into a training and a testing dataset. 
Here we will use 30% of our dataset for testing and keep 70% for training the models.

""")



st.markdown("## **Random Forest Classifier**")

st.write("""

réf. Wikipedia. 

Random forests or random decision forests is an ensemble learning method for classification, 
regression and other tasks that operates by constructing a multitude of decision trees at training time.
For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, 
the mean or average prediction of the individual trees is returned. Random decision forests correct for decision trees' 
habit of overfitting to their training set. Random forests generally outperform decision tree. However, data characteristics 
can affect their performance.

We will start by fitting a model with parameters: bootstrap=True, max_features='auto',
min_samples_split = 30, min_samples_leaf = 4, n_estimators=10. We will look at the result then try to 
tune the hyperparamers with gridsearch to improve model performance.

""")

st.markdown('##### **ROC AUC Plot:**')

img =Image.open("C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Image\\RFCROC.png")
st.image(img, caption='ROC AUC SCORE: 0.809')

st.markdown('#### **Confusion Matrix:**')

img =Image.open("C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Image\\RFCCM.png")
st.image(img, caption='Confusion Matrix:')

st.markdown("""
* The RFC Model classified all individuals as No Stroke, even if it's true for 1170 individuals we still have 57 missclassified indiviuals. This result is a bit predictable with unbalanced datasets.

### Hyperameter Tuning with Grid Search 

* We will run a random GridSearch with K-Fold Cross-Validation (4) to try to find better parameters. We have to be carefull
because it can be computively heavy and time consuming (reminder: The GridSearch needs to fit ant test each time with new parameters).

**Best parameters found:**
 'n_estimators': 1200, 'min_samples_split': 70, 'min_samples_leaf': 18, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': True

**ROC AUC Score:** 
0.8623931623931623

We managed to improve the ROC AUC Score of our Random Forest Classifier

We can now predict stroke probability for the individuals of the testing-stroke dataset, then format and submit results on kaggle.
""")