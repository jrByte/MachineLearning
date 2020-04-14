# Developed by jrbyte on Github.com. This was used for practice for machine learning using K nearest Algorithm.
# Some of the code written here was aided with a youtube tutorial.

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as pyplot
from matplotlib import style


# K Nearest Algorithm (Basic summary): Classification algorithm, tries to classify the data points with classes that it already knows.
# We need to look at the groups of data points which are classified already part of that group. Whatever the data point
# we don't know the classification of looks for the closest group to be classified as.
#
# K: is the number of data points closest to the unknown data point (magnitude). Whatever has a higher amount of a certain class is
# what the unknown class data point becomes. K needs to always be an odd number because if it is an even number then it
# has the possibility of it becoming a tie.
# More information: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
# https://scikit-learn.org/stable/getting_started.html
class firstTest:
    def main(self):
        data = pd.read_csv("car.data")
        print("\nChecking that data is being parsed correctly.")
        print(data.head())

        # Converting the the identifiers of the Category (car color: red, blue, green ,and etc) to values (car color: 1, 2, 3, 4)
        le = preprocessing.LabelEncoder()
        buying = le.fit_transform(list(data["buying"]))
        maint = le.fit_transform(list(data["maint"]))
        door = le.fit_transform(list(data["door"]))
        persons = le.fit_transform(list(data["persons"]))
        lug_boot = le.fit_transform(list(data["lug_boot"]))
        safety = le.fit_transform(list(data["safety"]))
        listClass = le.fit_transform(list(data["class"]))

        predict = "class"

        # Splitting the data to X and Y and putting chosen
        X = list(zip(buying, maint, door, persons, lug_boot, safety))
        Y = list(listClass)

        acc = 0.00
        count = 0
        neighbors = 1
        maxAttempts = 20
        maxNeighbors = 15
        bestAcc = 0.00
        bestNeighbors = 0
        bestModel = None
        bestX_test = None
        bestY_test = None

        # This while loop will find the best neighbor for the data set with the best accuracy. It can be specified above.
        while neighbors <= maxNeighbors:
            # Training the algorithm with k nearest neighbor
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
            # Every data set is different, so it may require you to tweak for the best amount of neighbors.
            model = KNeighborsClassifier(n_neighbors=neighbors)
            model.fit(x_train, y_train)
            acc = model.score(x_test, y_test)
            print("Attempts: ", count, ", Current Neighbors: ", neighbors, ", and Best Neighbor: ", bestNeighbors)
            print("Current Accuracy: ", acc, " vs BestAccuracy: ", bestAcc, "\n")

            if acc > bestAcc:
                bestAcc = acc
                bestNeighbors = neighbors
                bestModel = model
                bestX_test = x_test
                bestY_test = y_test

            if count == maxAttempts:
                count = 0
                # All neighbors must be odd.
                neighbors += 2
            count += 1

        predicted = bestModel.predict(x_test)
        names = ["unacc", "acc", "good", "vgood"]

        for x in range(len(x_test)):
            print("Predicted: ", names[predicted[x]], " Data: ", bestX_test[x], " Actual: ", names[bestY_test[x]])
            # Distance between each data point in neighbors.
            # n = model.kneighbors([x_test[x]], 9, True)
            # print("N: ", n)

        print("\nBest accuracy: ", bestAcc, "Best Neighbor: ", bestNeighbors)


if __name__ == '__main__':
    firstTest().main()
