# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data: Import the dataset and inspect column names.

2.Prepare Data: Separate features (X) and target (y).

3.Split Data: Divide into training (80%) and testing (20%) sets.

4.Scale Features: Standardize the data using StandardScaler.

5.Train Model: Fit a Logistic Regression model on the training data.

6.Make Predictions: Predict on the test set.

7.Evaluate Model: Calculate accuracy, precision, recall, and classification report.

8.Confusion Matrix: Compute and visualize confusion matrix. 

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: NIXAN DASS A
RegisterNumber:  212222040109
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('food_items (1).csv')

# Inspect the dataset
print('Name:NIXAN DASS A')
print('Reg. No: 212222040109')
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Separate features and target variable
X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]

# Scaling the raw input features
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Encode the target variable
y = label_encoder.fit_transform(y_raw.values.ravel())
# Note that ravel() function flattens the vector.

# Split the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

# L2 penalty to shrink coefficients without removing any features from the model
penalty = 'l2'
# Our classification problem is multinomial
multi_class = 'multinomial'
# Use lbfgs for L2 penalty and multinomial classes
solver = 'lbfgs'
# Max iteration = 1000
max_iter = 1000

# Define a logistic regression model with above arguments
l2_model = LogisticRegression(
    random_state=123, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter
)

# Train the model
l2_model.fit(X_train, y_train)

# Predictions
y_pred = l2_model.predict(X_test)

# Evaluate the model
print('Name: NIXAN DASS A')
print('Reg. No: 212222040109')
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print('Name: NIXAN DASS A')
print('Reg. No: 212222040109')
```

## Output:
<img width="1410" height="528" alt="image" src="https://github.com/user-attachments/assets/8baf53c7-6cf5-4e80-9051-45f333482944" />
<img width="1478" height="503" alt="image" src="https://github.com/user-attachments/assets/9b28bd26-b0c4-4a43-bf80-9608fe266abe" />
<img width="1270" height="394" alt="image" src="https://github.com/user-attachments/assets/83d9edd0-27db-45fc-bc05-8433f7ca03ef" />


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
