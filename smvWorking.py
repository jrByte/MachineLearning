import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics


# SVM (support vector machines) can be used for regression and classification.
# Linear way to divide your data. Typically working with data points that are straight.
class example1:
    def main(self):
        cancer = datasets.load_breast_cancer()

        # Label names for each cancer feature/array of data.
        # print(cancer.feature_names)
        # print(cancer.target_names)

        x = cancer.data
        y = cancer.target

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

        # print(x_train.shape, y_train.shape)
        classes = ['malignant', 'benign']

        clf = svm.SVC(kernel="poly", C=2)
        clf.fit(x_train, y_train)

        y_prediction = clf.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_prediction)
        print("Accuracy: " + str(acc * 100) + "%")


if __name__ == '__main__':
    example1().main()
