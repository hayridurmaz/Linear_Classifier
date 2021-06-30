import argparse
import random
from contextlib import redirect_stdout

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.cluster import KMeans


def read_data(filename):
    '''
    Read and pre-process data
    :param filename:
    :return:
    '''
    with open(filename) as file:
        lines = file.readlines()
    dataset = []
    for s in lines:
        s = s.strip('\n').strip()
        s_1 = s[s.rfind(' '):]
        s_2 = s[:s.rfind(' ')]
        tuple_data = (float(s_1.strip()), float(s_2.strip()))
        dataset.append(tuple_data)
    random.shuffle(dataset)
    arr = np.array(dataset)
    return arr[10:], arr[:10]


def plotData(x, y, title=None, x_label=None, y_label=None):
    # Getting unique labels
    u_labels = np.unique(y)
    # plotting the results:
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i in u_labels:
        plt.scatter(x[y == i, 0], x[y == i, 1], label=i)
    plt.legend()
    plt.show()


def purity_score(y_true, y_pred):
    """
    K-Means purity calculator
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def testKMeans(train_data, train_labels, test_data, test_labels):
    """
    K-Means test method
    """
    km = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    km.fit(train_data)
    y_pred = km.predict(test_data)
    plotData(test_data, y_pred, title="KMeans Result")
    print("---------KMEANS---------")
    # print("K-Means Accuracy:", metrics.accuracy_score(test_labels, y_pred))
    print("K-Means Purity: ", purity_score(y_true=test_labels, y_pred=y_pred))
    # print("K-Means Rand-index score: ", rand_index_score(test_labels, y_pred))
    print("---------KMEANS---------")


'''
Perceptron algo
'''


class Perceptron(object):
    def __init__(self, rate=0.01, niter=10):
        self.rate = rate
        self.niter = niter

    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        """

        # weights
        self.weight = np.zeros(1 + X.shape[1])

        # Number of misclassifications
        self.errors = []  # Number of misclassifications

        for i in range(self.niter):
            err = 0
            for xi, target in zip(X, y):
                delta_w = self.rate * (target - self.predict(xi))
                self.weight[1:] += delta_w * xi
                self.weight[0] += delta_w
                err += int(delta_w != 0.0)
            self.errors.append(err)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    plt.legend(loc='upper left')
    plt.show()


def tests():
    file_dir = ''
    if args.cmd:
        file_dir = '../'
    C_1_Training, C_1_Test = read_data('{}C1__pts.txt'.format(file_dir))
    C_2_Training, C_2_Test = read_data('{}C2__pts.txt'.format(file_dir))

    dataset = np.concatenate((C_1_Training, C_2_Training))
    labels = np.zeros(180)
    labels[90:] = 2
    labels[:90] = 1

    labels_test = np.zeros(20)
    labels_test[10:] = 2
    labels_test[:10] = 1
    plotData(dataset, labels, title="Training data")

    data = {
        "both_train_data": dataset,
        "both_train_labels": labels,
        "both_test_data": np.concatenate((C_1_Test, C_2_Test)),
        "both_test_labels": labels_test
    }

    # K-MEANS TEST
    testKMeans(data["both_train_data"], data["both_train_labels"], data["both_test_data"], data["both_test_labels"])

    pn = Perceptron(0.01, 100)
    pn.fit(dataset, labels)
    plot_decision_regions(data['both_test_data'], data['both_test_labels'], classifier=pn)


def main():
    # for i in range(10):
    tests()


parser = argparse.ArgumentParser(description='Test Some Linear Classifier for BLG527E (ML) Class of ITU \'21 ')
parser.add_argument("-f", "--file-output", action="store_true",
                    help="Write output to a file instead of terminal.")
parser.add_argument("-b", "--best-estimator", action="store_true",
                    help="While testing svm, using best estimator instead of pre-defined ones.(Takes more time)")
parser.add_argument("-c", "--cmd", action="store_true",
                    help="Run from cmd (without an ide)")
args = parser.parse_args()

if __name__ == '__main__':
    if args.file_output:
        with open('output', 'a+') as f:
            with redirect_stdout(f):
                main()
                print('\n')
    else:
        main()
