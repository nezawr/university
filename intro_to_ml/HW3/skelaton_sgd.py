#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
import math

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784',as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
   
    # Here w_t is w_1
    w_t = np.zeros(data.shape[1])

    for t in range(1,T+1):
        eta_t = eta_0/t
        i = np.random.randint(data.shape[0])
        x_i = data[i]
        y_i = labels[i]
        p = y_i * np.dot(w_t, x_i)
        if p < 1:
            w_t = ((1 - eta_t) * w_t) + (eta_t * C * y_i * x_i)
        else:
            w_t = (1 - eta_t) * w_t

    return w_t
    

def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    # TODO: Implement me
    pass

#################################

# Place for additional code

#################################
def calculate_loss(data, labels, w):
    n = data.shape[0]
    accuracy = 0
    for i in range(n):
        x_i = data[i]
        y_i = labels[i]
        margin = y_i * np.dot(w, x_i)
        #Corrent classification
        if (margin >= 1):
            accuracy += 1
        #insufficient margin - wrong classification
        else:
            continue
    return accuracy/n 


def find_best_eta0(T, C):
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    loss_avg = []
    etas = np.arange(0.00001, 10, 0.01)

    for eta in etas:
        avg = 0
        for t in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta, T)
            avg += calculate_loss(validation_data, validation_labels, w)
        loss_avg.append(avg/10)

    plt.plot(etas, loss_avg, label="Eta Accuracy")
    plt.legend()
    plt.xlabel("Eta")
    plt.ylabel("Accuracy")
    plt.show()
    
    return etas[np.argmin(loss_avg)]

def find_best_C(T, op_eta, t_d, t_l, v_d, v_l):
    loss = []
    C = np.arange(0.00001,10, 0.01)
    for c in C:
        l = 0
        for i in range(10):
            w = SGD_hinge(t_d, t_l, c, op_eta, T)
            l += calculate_loss(v_d, v_l, w)
        loss.append(l/10)
    
    plt.plot(C, loss, label="Accuracy by C")
    plt.legend()
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.show()

    return C[np.argmin(loss)]
    
        
if __name__ == '__main__':
    #1.A
    find_best_eta0(1000, 1)
    #train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
