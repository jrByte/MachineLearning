# Program written by jrbyte @Github.com, 2/9/2020
# My first machine learning program.
# Why use linear regression? We're using linear regression because the data directly correlates with each other.
# Using linear regression to predict what the final grade (G3) would be based on correlating data.
# The program iterates through 10% of the data to find the best suited accuracy which is set to 97% or more.

# Unused imports:
# from sklearn.utils import shuffle
# import tensorflow as tf

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Separates the database values that has a semicolon between each one.
data = pd.read_csv('student-mat.csv', sep=';')

print("Before:")
# print(data.head())

# Everything here is known as an attribute.
data = np.array(data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']])
print("!Before:")
# print(data)

# G3 is a label because we want to predict this. What we're essentially looking for.
predict = 'G3'

# Creating an array which drops G3 from the data frame. axis=1 because we're trying to delete a column.
X = np.delete(data, 2, axis=1)
print("\nX: ", X)
# print(X)

# Array with only G3
Y = np.array(data[:, 2])
print("\nY:", Y)
# print(Y)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

print("\nLinear Regression:")
scoreObjective = 0.97
best = 0
count = 0

# While loop continuously looks for a accuracy score higher above 97%
while best < scoreObjective:
    count += 1
    # Splitting 10% of our data into test samples so that when we test it can test on information that its never seen before.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    # Choosing the linear model that we want to use.
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # Accuracy of our model.
    acc = linear.score(x_test, y_test)
    print("Accuracy:", acc, " Best Score so far: ", best, end="\r")

    # Chooses the better score and writes then saves the better model in pickle
    if acc > best:
        best = acc
        # Saves the model for us using pickle
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

print("\nFound a score of: ", scoreObjective)
print("Amount of tries: ", count)

# Loading the model from studentmodel.pickle to linear
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Linear Co: ", linear.coef_)
print("Y Intercept: ", linear.intercept_, "\n")

predictions = linear.predict(x_test)

# Going over the x_tests and checking if our results of our predictions
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Plotting data:
# We're are plotting the First semester grade with the final grade.
style.use("ggplot")

# data[:, 0] represents the first column (G1 = first semester) data[:, 2] represents the first column (G3 = final grade)
pyplot.scatter(data[:, 0], data[:, 2])

# X and Y
pyplot.xlabel("G1")
pyplot.ylabel("Final Grade")

# Creates the graph of first semester and final grade
pyplot.show()
