**Overview**

This project builds a Machine Learning model to predict the survival of passengers on the Titanic.
It includes data cleaning, feature engineering, encoding, model training, and evaluation using multiple ML algorithms.

This project is designed for Data Science / Machine Learning / Data Engineering portfolios.

**Project Structure**

Titanic-ML-Project/
│
├── README.md
├── requirements.txt
├── titanic_ml.py
│
└── data/
    └── titanic.csv

**Data Cleaning & Preprocessing**

The dataset is cleaned using:
Filling missing values (most frequent / most specified)
Dropping unnecessary columns
Label encoding categorical variables
Handling duplicates
Converting strings → numerical values
Scaling/normalizing numerical values (optional)

**Machine Learning Models Used**

The project trains and compares multiple ML models:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier

**Each model is evaluated on:**

Accuracy
Precision, Recall, F1 Score
Confusion Matrix
Classification Report

**ML Workflow**

Load dataset
Clean missing values
Encode categorical features
Split into Train/Test sets
Train ML models
Evaluate performance
Plot Confusion Matrix using Matplotlib
Display results

**How to Run the Project**
1. Install dependencies
   pip install -r requirements.txt
2.python titanic_ml.py
   python Titanic Dataset Project.ipynb
3. Output
    Accuracy score of each model
    Classification Report
    Confusion Matrix Plot
    Predicted vs Actual values

**Files Included**

Titanic Dataset Project.ipynb

Contains the full project code:

Data cleaning
Encoding
ML model training
Evaluation metrics
Visualization

**requirements.txt**

**Packages required:**

pandas
numpy
scikit-learn
matplotlib
seaborn

**Titanic-Dataset.xlsx**

Input dataset used for training.

**Skills Demonstrated**

Data Cleaning
Data Preprocessing
Feature Engineering
Exploratory Data Analysis
Machine Learning (Classification)
Confusion Matrix Plotting
Scikit-Learn
Matplotlib
GitHub Project Structure
Python Scripting

**Final Results**

ML model performance comparison
Visual confusion matrix
Predictions on unseen test data
