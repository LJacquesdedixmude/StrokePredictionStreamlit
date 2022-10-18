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
Here we will use 30% of our dataset for testing and keep 70% for training the models. (always use random state 1170 if you want same results)

Here-Under you can choose a model. For each model you can find information such as Parameters, ROC AUC Score, Confusion Matrix, Hyperparameter
Tuning or performance. Other Models were trained and fitted on the data set but here only the better performing models are presented.  

""")

st.subheader("Select Model")
Model = st.selectbox("Models", ["Random Forest Classifier", "Gradient Boosting Classifier", "Logistic Regression"])

if Model=="Random Forest Classifier":
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

    * For Documentation : **Scikit-Learn**: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    """)

    st.markdown('##### **ROC AUC Plot:**')

    img =Image.open("C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Image\\RFCROC.png")
    st.image(img, caption='ROC AUC SCORE: 0.796')

    st.markdown('#### **Confusion Matrix:**')

    img =Image.open("C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Image\\RFCCM.png")
    st.image(img, caption="""Confusion Matrix:[[1170  :   0]
    [  57  :   0]]
    """)

    st.markdown("""
    * The RFC Model classified all individuals as No Stroke, even if it's true for 1170 individuals we still have 57 missclassified indiviuals. This result is a bit predictable with unbalanced datasets.

    ### Hyperameter Tuning with Grid Search 

    * We will run a random GridSearch with K-Fold Cross-Validation (4) & optimize ROC AUC (scoring=ROC AUC) to try to find better parameters. We have to be carefull
    because it can be computively heavy and time consuming (reminder: The GridSearch needs to fit ant test each time with new parameters).

    **Best parameters found:**
    'n_estimators': 800, 'min_samples_split': 80, 'min_samples_leaf': 16, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': True

    **ROC AUC Score:** 
    0.817

    We managed to improve the ROC AUC Score of our Random Forest Classifier

    We can now predict stroke probability for the individuals of the testing-stroke dataset, then format and submit results on kaggle.
    """)

elif Model=="Gradient Boosting Classifier":
    st.markdown("## **Gradient Boosting Classifier**")

    st.write("""

    réf. Wikipedia. 

    Gradient boosting is a machine learning technique used in regression and classification tasks, among others. 
    It gives a prediction model in the form of an ensemble of weak prediction models, which are typically decision trees.
    When a decision tree is the weak learner, the resulting algorithm is called gradient-boosted trees; it usually outperforms random forest.
    A gradient-boosted trees model is built in a stage-wise fashion as in other boosting methods, but it generalizes the other methods
    by allowing optimization of an arbitrary differentiable loss function.

    We will start by fitting a model with parameters: n_estimators=100, learning_rate=1.0 and max_depth=1 (and Randomstate=0)

    * For Documentation: **Scikit-Learn** https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    """)

    st.markdown('##### **ROC AUC Plot:**')

    img =Image.open("C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Image\\GBCROC.png")
    st.image(img, caption='ROC AUC SCORE: 0.811')

    st.markdown('#### **Confusion Matrix:**')

    img =Image.open("C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Image\\GBCCM.png")
    st.image(img, caption='''Confusion Matrix:[1166  :  4]

    [  55  :   2]''')

    st.markdown("""

    * If we refer to the ROC AUC Score, the model performs bettre than the Random Forest Classifier before Hyperparameter tuning but not as well as the RFC after hyperparameter tuning.

    * When we look at the confusion matrix we can see that the model attempted to predict Strokes. It predicted 6 Strokes and got 2 stroke predictions accurate.

    ### Hyperparameter tuning with Grid Search

    * We will run a random GridSearch with K-Fold Cross-Validation (4) and optimize ROC AUC (scoring = ROC AUC). 

    **Best Parameters Found:** 
    'n_estimators': 300, 'min_samples_split': 20, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 3, 'learning_rate': 0.01}

    **ROC AUC SCORE:** 
    0.837
    """)

elif Model=="Logistic Regression":
    st.markdown("## **Logistic Regression**")

    st.write("""

    Logistic Regression is model that origniates from statistics but also have applications in Machine Learning. (See Support Vector Machines)
    If we refer to PyCaret, before tuning Logistic Regression is the model that has the best ROC AUC Score on the data set.

    réf. Wikipedia. 

    In statistics, the logistic model (or logit model) is a statistical model that models the probability of an event taking place by having 
    the log-odds for the event be a linear combination of one or more independent variables. In regression analysis, logistic regression
    (or logit regression) is estimating the parameters of a logistic model (the coefficients in the linear combination).
    Formally, in binary logistic regression there is a single binary dependent variable, coded by an indicator variable, where the two values
    are labeled "0" and "1", while the independent variables can each be a binary variable (two classes, coded by an indicator variable) or a 
    continuous variable (any real value). The corresponding probability of the value labeled "1" can vary between 0 (certainly the value "0") 
    and 1 (certainly the value "1"), hence the labeling;[2] the function that converts log-odds to probability is the logistic function,
    hence the name. 

    We will start by fitting a model with parameters: bootstrap=True, max_features='auto',
    min_samples_split = 30, min_samples_leaf = 4, n_estimators=10. We will look at the result then try to 
    tune the hyperparamers with gridsearch to improve model performance.

    * For Documentation : **Scikit-Learn**: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    """)

    st.markdown('##### **ROC AUC Plot:**')

    img =Image.open("C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Image\\LRROC.png")
    st.image(img, caption='ROC AUC SCORE: 0.819')

    st.markdown('#### **Confusion Matrix:**')

    img =Image.open("C:\\Users\\ljacquesdedixmude\\Git\\solvay-digital-society-stroke-prediction-hackathon-2022\\Image\\LRCM.png")
    st.image(img, caption="""Confusion Matrix:[1164 :   6]
    [  56  :  1]]
    """)

    st.markdown("""

    * If we refer to ROC AUC, the Logistic Regression Model performs slightly better than the Gradient Boosting Classifier Model, but not as well as the tuned GBC Model. 

    * If we look at the confusion matrix we can see that on the model predicted 7 Strokes and got 6 accurate

    """)
