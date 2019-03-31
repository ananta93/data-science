# Simple linear regression
"""
Created on Tue Mar  5 17:00:42 2019

@author: anadas1
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#predicting the test set results
y_pred = regressor.predict(x_test)


#Visulising the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('salary vs experience(Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show

#visualizing the test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs experience(Test set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show

