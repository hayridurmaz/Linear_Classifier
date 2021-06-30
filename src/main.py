import argparse
import random
from contextlib import redirect_stdout
from time import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV


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


def getSVMBestEstimator(train_data, train_labels):
    t0 = time()
    # Create a dictionary of possible parameters
    params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'gamma': [0.0001, 0.001, 0.01, 0.1],
                   'kernel': ['linear', 'rbf']}

    # Create the GridSearchCV object
    grid_clf = GridSearchCV(svm.SVC(class_weight='balanced'), params_grid)

    # Fit the data with the best possible parameters
    grid_clf = grid_clf.fit(train_data, train_labels)

    # Print the best estimator with it's parameters
    print(grid_clf.best_estimator_)
    print("Best estimator done in %0.3fs" % (time() - t0))
    return grid_clf


def testSVM(train_data, train_labels, test_data, test_labels):
    if args.best_estimator:
        # Get svm with best estimators (Comment out if not necessary)
        clf = getSVMBestEstimator(train_data, train_labels)
    else:
        # Create a svm Classifier
        clf = svm.SVC(C=0.1, class_weight='balanced', gamma=0.0001, kernel='linear')  # Linear Kernel
        # Train SVM using the training sets
        clf.fit(train_data, train_labels)

    # Predict the response for test dataset
    y_pred = clf.predict(test_data)
    # Model Accuracy calculated
    plotData(test_data, y_pred, title="SVM Result")
    print("---------SVM---------")
    print("SVM Accuracy:", metrics.accuracy_score(test_labels, y_pred))
    precision, recall, fscore, support = score(test_labels, y_pred)
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))
    print("---------SVM---------")


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

    # svm_test_2(data["both_train_data"], data["both_train_labels"], data["both_test_data"], data["both_test_labels"])
    # K-MEANS TEST
    testKMeans(data["both_train_data"], data["both_train_labels"], data["both_test_data"], data["both_test_labels"])


def main():
    # for i in range(10):
    tests_2()


parser = argparse.ArgumentParser(description='Test Some Linear Classifier for BLG527E (ML) Class of ITU \'21 ')
parser.add_argument("-f", "--file-output", action="store_true",
                    help="Write output to a file instead of terminal.")
parser.add_argument("-b", "--best-estimator", action="store_true",
                    help="While testing svm, using best estimator instead of pre-defined ones.(Takes more time)")
parser.add_argument("-c", "--cmd", action="store_true",
                    help="Run from cmd")
args = parser.parse_args()

if __name__ == '__main__':
    if args.file_output:
        with open('output', 'a+') as f:
            with redirect_stdout(f):
                main()
                print('\n')
    else:
        main()
