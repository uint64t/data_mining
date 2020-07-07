import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn import svm
import SVM
import naive_bayes as nb

iris_path = "./dataset/iris"
labor_path = "./dataset/labor/C4.5"

iris_data = "iris.data"
labor_data = "labor-neg.data"

iris_name_list = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]


# for implement, you can see:
# https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea
def Decision_Tree_ID3(X, y, visulize_tree=False, print_info_matrix=False, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)

    acc = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        classifier = DecisionTreeClassifier(criterion="entropy", max_depth=10)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc.append(sum([y_test[i] == y_pred[i] for i in range(len(y_test))]) / float(len(y_test)) * 100.0)
        if print_info_matrix:
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
    if visulize_tree:
        plt.figure(dpi=300)
        plot_tree(classifier, filled=True, class_names=y.unique(), feature_names=list(X))
        plt.show()
    print(f"average acc: {sum(acc) / len(acc)}")


def Naive_Bayes(X, y, print_info_matrix=False, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)

    acc = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        acc.append(accuracy_score(y_test, y_pred) * 100)
        if print_info_matrix:
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
    print(f"average acc: {sum(acc) / len(acc)}")
    # nb.main(dataset)


def SVM_sklearn(X, y, n_splits=5, p_info=False):
    kf = KFold(n_splits=n_splits)

    acc = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        clf = svm.SVC(gamma="auto", C=10000, kernel="linear")
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        acc.append(accuracy_score(y_test, y_pred) * 100)
        if p_info:
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
    print(f"average acc: {sum(acc) / len(acc)}")
    # nb.main(dataset)


def read_iris():
    dataset = os.path.join(iris_path, iris_data)
    data = pd.read_csv(dataset, names=iris_name_list)
    X = data.drop(["class", "sepal_len", "sepal_wid"], axis=1)
    y = data["class"]
    return X, y


def read_breast_cancer_wisconsin():
    print("reading dataset...")
    dataset = './dataset/classification/svm/Breast_Cancer_Wisconsin.csv'
    data = pd.read_csv(dataset)

    # drop last column (extra column added by pd)
    # and unnecessary first column (id)
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

    # convert categorical labels to numbers
    diag_map = {'M': 1.0, 'B': -1.0}
    data['diagnosis'] = data['diagnosis'].map(diag_map)
    # put features & outputs in different data frames
    Y = data.loc[:, 'diagnosis']
    X = data.iloc[:, 1:]
    return X, Y


def main(dataset):
    if dataset == "iris":
        X, y = read_iris()
    elif dataset == "breast_cancer":
        X, y = read_breast_cancer_wisconsin()
    else:
        raise ValueError("Please provide one of [iris, breast_cancer]")

    # there's a notable circumstance, when applying feature selection,
    # the performance of DT and NB decreased, while the SVM increased
    # how it could be?
    # SVM.filter_feature(X, y)
    Decision_Tree_ID3(X, y)
    Naive_Bayes(X, y)
    # SVM.main(X, y)
    SVM_sklearn(X, y, p_info=False)


if __name__ == '__main__':
    main("breast_cancer")


