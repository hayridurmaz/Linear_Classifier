import argparse
import random
from contextlib import redirect_stdout

import numpy as np
from matplotlib import pyplot as plt


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


def main():
    C_1_Training, C_1_Test = read_data('./data/C1__pts.txt')
    C_2_Training, C_2_Test = read_data('./data/C2__pts.txt')

    dataset = np.concatenate((C_1_Training, C_2_Training))
    labels = np.zeros(180)
    labels[90:] = 2
    labels[:90] = 1
    plotData(dataset, labels, title="DATASET")


parser = argparse.ArgumentParser(description='Test Some Linear Classifier for BLG527E (ML) Class of ITU \'21 ')
parser.add_argument("-f", "--file-output", action="store_true",
                    help="Write output to a file instead of terminal.")
args = parser.parse_args()

if __name__ == '__main__':
    if args.file_output:
        with open('output', 'a+') as f:
            with redirect_stdout(f):
                main()
                print('\n')
    else:
        main()
