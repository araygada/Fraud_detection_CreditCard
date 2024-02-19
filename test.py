import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

creditCard_df = pd.read_csv("creditcard.csv")

# separate legitimate and fraudulent transactions
legit = creditCard_df[creditCard_df.Class ==0]
fraud = creditCard_df[creditCard_df.Class ==1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns='Class', axis=1)
Y = data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)
ypred = model.predict(X_test)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(ypred, Y_test)

# Create a streamlit app
st.title("Credit Card Detection Model")
st.write("Enter the string of the transaction features values to check if it is legitamate or fraudulent:")

# Create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')

# Create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    
    # Make prediction
    prediction = model.predict(features.reshape(1,-1))
    
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")
        
            