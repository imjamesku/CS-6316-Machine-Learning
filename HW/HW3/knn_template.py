# Starting code for UVA CS 4501 Machine Learning- KNN

__author__ = '**'
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
np.random.seed(37)
# for plot
# more imports
# the only purpose of the above import is in case that you want to compare your knn with sklearn knn


# Load file into np arrays
# x is the features
# y is the labels
def read_file(file):
    data = np.loadtxt(file, skiprows=1)
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x, y

# 2. Generate the i-th fold of k fold validation
# Input:
# x is an np array for training data
# y is an np array for labels
# i is an int indicating current fold
# nfolds is the total number of cross validation folds


def fold(x, y, i, nfolds):
    # your code
    num_examples = x.shape[0]
    num_per_fold = num_examples//nfolds
    test_start = i*num_per_fold
    x_train = np.concatenate(
        (x[0:test_start], x[test_start+num_per_fold:]), axis=0)
    y_train = np.concatenate(
        (y[0:test_start], y[test_start+num_per_fold:]), axis=0)
    x_test = x[test_start: test_start+num_per_fold]
    y_test = y[test_start: test_start+num_per_fold]
    return x_train, y_train, x_test, y_test

# 3. Classify each testing points based on the training points
# Input
# x_train: a numpy array of training data
# x_test: a numpy array
# k: the number of neighbors to take into account when predicting the label
# Output
# y_predict: a numpy array


def classify(x_train, y_train, x_test, k):
    def distanceFrom(x_to_predict):
        return lambda x: np.linalg.norm(x_to_predict - x[0])
    # your code
    # Euclidean distance as the measurement of distance in KNN
    y_predict = []
    for xi in x_test:
        # sorted_x = [xi for xi, yi in sorted(zip(x_train, y_train), key=distanceFrom(xi))]
        sorted_y = [yi for xi, yi in sorted(
            zip(x_train, y_train), key=distanceFrom(xi))]
        k_nearest = sorted_y[:k]
        # print(k_nearest)
        counts = Counter(k_nearest)
        y_predict.append(counts.most_common(1)[0][0])
    return y_predict

# 4. Calculate accuracy by comaring with true labels
# Input
# y_predict is a numpy array of 1s and 0s for the class prediction
# y is a numpy array of 1s and 0s for the true class label


def calc_accuracy(y_predict, y):
    # your code
    count = 0
    for y_predict_i, yi in zip(y_predict, y):
        if y_predict_i == yi:
            count += 1
    acc = count/y.shape[0]
    return acc

# 5. Draw the bar plot of k vs. accuracy
# klist: a list of values of ks
# accuracy_list: a list of accuracies


def barplot(klist, accuracy_list):
    # your code
    print(klist)
    print(accuracy_list)
    plt.bar(klist, accuracy_list)
    axes = plt.gca()
    axes.set_ylim([0.6, 0.7])
    plt.show()
    # use matplot lib to generate bar plot with K on x axis and cross validation accuracy on y-axis
    return

# 1. Find the best K


def findBestK(x, y, klist, nfolds):
    kbest = 0
    best_acc = 0
    accuracy_list = []
    for k in klist:
        # your code here
        n_fold_accuracy = []
        for i in range(nfolds):
            x_train, y_train, x_test, y_test = fold(x, y, i, nfolds)
            y_predict = classify(x_train, y_train, x_test, k)
            n_fold_accuracy.append(calc_accuracy(y_predict, y_test))
        # to get nfolds cross validation accuracy for k neighbors
        # implement fold(x, y, i, nfolds),classify(x_train, y_train, x_test, k) and calc_accuracy(y_predict, y)

        # CROSS VALIDATION accuracy for k neighbors
        accuracy = sum(n_fold_accuracy)/len(n_fold_accuracy)
        if accuracy > best_acc:
            kbest = k
            best_acc = accuracy
        accuracy_list.append(accuracy)
        print(k, accuracy)
    # plot cross validation error for each k : implement function barplot(klist, accuracy_list)
    barplot(klist, accuracy_list)
    return kbest


if __name__ == "__main__":
    filename = "Movie_Review_Data.txt"
    # read data
    x, y = read_file(filename)
    nfolds = 4
    klist = [3, 5, 7, 9, 11, 13]
    # Implementation covers two tasks, both part of findBestK function
    # Task 1 : implement kNN classifier for a given x,y,k
    # Task 2 : implement 4 fold cross validation to select best k from klist

    bestk = findBestK(x, y, klist, nfolds)
    # report best k, and accuracy, discuss why some k work better than others
