#################################
# Your name: Nazar Aburas
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
from numpy.core.fromnumeric import size
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
    #Label classes are 0,1,...,9
    W = np.zeros((10, data.shape[1]))
    for t in range(1,T+1):
        i = np.random.randint(data.shape[0])
        x_i = data[i]
        y_i = labels[i]
        gradients = ce_gradient(W, x_i, y_i)
        W = W - eta_0 * gradients
    return W

#################################

# Place for additional code

#################################
def hinge_accuracy(data, labels, w):
    n = data.shape[0]
    loss = 0
    for x_i, y_i in zip(data, labels):
        margin = y_i * np.dot(w, x_i)
        #Correct classification
        if (margin >= 1):
            continue
        #insufficient margin - wrong classification
        else:
           loss += 1
    return 1 - loss/n 

def hinge_optimal_eta(T, C, train_data, train_labels, validation_data, validation_labels):
    accuracy_avg = []
    # Started search with log scale values
    # etas = [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10, 10**2 ,10**3, 10**4]
    # Narrowed down search
    etas = np.arange(0.00001, 10, 0.1)
    for eta in etas:
        avg = 0
        for t in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta, T)
            avg += hinge_accuracy(validation_data, validation_labels, w)
        accuracy_avg.append(avg/10)

    plt.plot(etas, accuracy_avg, label="Accuracy by Eta")
    plt.legend()
    plt.xlabel("Eta")
    plt.ylabel("Accuracy")
    plt.show()
    
    return etas[np.argmax(accuracy_avg)]

def hinge_optimal_c(T, op_eta, t_d, t_l, v_d, v_l):
    accuracy = []
    C = [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10, 10**2 ,10**3, 10**4]
    #C = np.arange(0.00001, 20, 0.1)
    for c in C:
        a = 0
        for i in range(10):
            w = SGD_hinge(t_d, t_l, c, op_eta, T)
            a += hinge_accuracy(v_d, v_l, w)
        accuracy.append(a/10)
    
    plt.plot(C, accuracy, label="Accuracy by C")
    plt.legend()
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.show()

    return C[np.argmax(accuracy)]

def ce_accuracy(data, labels, W):
    loss = 0
    for x_i, y_i in zip(data, labels):
        prediction = np.argmax(np.array([np.dot(W[i], x_i) for i in range(W.shape[0])]))
        if prediction != int(y_i):
            loss += 1
    return 1 - loss/data.shape[0]
    
def ce_gradient(W, x, y):
    gradient = np.zeros((W.shape[0], W.shape[1]))
    for k in range(W.shape[0]):
        gradient[k] = ce_derivative(W, x, y, k)
    return gradient

def ce_derivative(W, x, y, k):
    #Log-sum-exp trick
    A = np.array([np.dot(W[i], x) for i in range(W.shape[0])])
    a_max = np.max(A)
    lse = a_max + np.log(np.sum(np.exp(A - a_max)))
    p = np.exp(A[k] - lse)
    if k != int(y):
        return (p)*x
    return (p - 1)*x


def ce_optimal_eta(T, t_d, t_l, v_d, v_l):
    accuracy_avg = []
    #etas = np.array([10**-8 ,10**-7, 10**-6, 10**-5 ,10**-4, 10**-3, 10**-2, 10**-1, 10**1, 10**2 ,10**3, 10**4, 10**5, 10**6, 10**7, 10**8])
    etas = np.arange(0.000001,1, 0.01)
    for eta in etas:
        avg = 0
        for t in range(10):
            w = SGD_ce(t_d, t_l, eta, T)
            avg += ce_accuracy(v_d, v_l, w)
        accuracy_avg.append(avg/10)
    
    plt.plot(etas, accuracy_avg, label="Cross Entropy Eta Accuracy")
    plt.legend()
    plt.xlabel("Eta")
    plt.ylabel("Accuracy")
    plt.show()
    
    return etas[np.argmax(accuracy_avg)]
  
        
if __name__ == '__main__':
    #SECTION 1
    train_data_hinge, train_labels_hinge, validation_data_hinge, validation_labels_hinge, test_data_hinge, test_labels_hinge = helper_hinge()
    T0 = 1000
    C = 1
    #1.a
    opt_eta = hinge_optimal_eta(T0, C, train_data_hinge, train_labels_hinge, validation_data_hinge, validation_labels_hinge)
    print(f"Optimal Hinge eta is: {opt_eta}")
    #1.b
    opt_eta = 0.70001
    opt_c = hinge_optimal_c(T0, opt_eta, train_data_hinge, train_labels_hinge, validation_data_hinge, validation_labels_hinge)
    print(f"Optimal Hinge C is: {opt_c}")
    #1.c
    T1 = 2000
    w_hinge = SGD_hinge(train_data_hinge, train_labels_hinge, opt_c, opt_eta, T1)
    hw = w_hinge.reshape((28,28))
    plt.imshow(hw, interpolation='nearest')
    plt.show()
    #1.d
    hinge_accuracy = hinge_accuracy(test_data_hinge, test_labels_hinge, w_hinge)
    print(f"Hinge classifier test accuracy: {hinge_accuracy}")

    #SECTION 2
    T2 = 1000
    train_data_ce, train_labels_ce, validation_data_ce, validation_labels_ce, test_data_ce, test_labels_ce = helper_ce()
    ce_eta_opt = ce_optimal_eta(T2, train_data_ce, train_labels_ce, validation_data_ce, validation_labels_ce)
    print(f"Optimal Cross Entropy eta is: {ce_eta_opt}")
    T3 = 2000
    w = SGD_ce(train_data_ce, train_labels_ce, ce_eta_opt, T3)
    for i in range(10):
        x = w[i].reshape((28,28))
        plt.imshow(x, interpolation='nearest')
        plt.show()
    
    cross_entropy_accuracy = ce_accuracy(test_data_ce, test_labels_ce, w)
    print(f"Cross Entropy classifier test accuracy: {cross_entropy_accuracy}") 
    
    
    