# Import libraries

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')

# Import data

df = pd.read_csv('antenna_model.csv')
print(df.head())


# Check NULL values if any


print(df.isnull().sum())


# We will predict the values of W given the rest 4 parameters.

X = df.drop(columns = ['w'])
y = df['w']


print(X.head())

print(y.head())


# Import Neural Network model from sci-kit learn


from sklearn import svm 

from sklearn.model_selection import train_test_split


# Training data will be 70% of the the total dataset


X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 10)


# Scaling has far-reaching affects MLP NN 


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler() 
# Don't cheat - fit only on training data
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
# apply same transformation to test data
X_test = scaler.transform(X_test)


print(100*len(y_train)/len(y))  #we will use 70% of data to train our model

#set the regressor

regr = svm.SVR(kernel = 'rbf')


# fit your model


regr.fit(X_train, y_train)


# Store your predictions in y_pred


y_pred = regr.predict(X_test)


# get the accuracy of model


print(regr.score(X_test,y_test))

