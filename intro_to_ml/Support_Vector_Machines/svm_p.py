#################################
# Your name: Tomer Porian
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    clf_lin = svm.SVC(C=1000, kernel='linear')
    clf_lin.fit(X_train, y_train)
    clf_quad = svm.SVC(C=1000, kernel='poly', degree=2, coef0=1)
    clf_quad.fit(X_train, y_train)
    clf_rbf = svm.SVC(C=1000, kernel='rbf')
    clf_rbf.fit(X_train, y_train)
    return np.stack((clf_lin.n_support_, clf_quad.n_support_, clf_rbf.n_support_))


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C_arr = np.logspace(-5.0, 5.0, num=11)
    clfs_acc = np.array([svm.SVC(C=c, kernel='linear').fit(X_train,y_train).score(X_val, y_val) for c in C_arr])
    clfs_acc_train = np.array([svm.SVC(C=c, kernel='linear').fit(X_train,y_train).score(X_train, y_train) for c in C_arr])
    plt.plot(np.linspace(-5.0, 5.0, num=11), clfs_acc, label="Validation Accuracy")
    plt.plot(np.linspace(-5.0, 5.0, num=11), clfs_acc_train, label="Test Accuracy")
    plt.xlabel(r'$C = 10^{i}$')
    plt.legend(loc="lower left")
    plt.ylabel('Accuracy')
    plt.axis([-5, 5, 0, 1.1])
    plt.show()
    return clfs_acc


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    gamma_arr = np.logspace(-5.0, 5.0, num=11)
    clfs_acc = np.array([svm.SVC(C=10, kernel='rbf', gamma=g).fit(X_train, y_train).score(X_val, y_val) for g in gamma_arr])
    clfs_acc_train = np.array([svm.SVC(C=10, kernel='rbf', gamma=g).fit(X_train, y_train).score(X_train, y_train) for g in gamma_arr])
    plt.plot(np.linspace(-5.0, 5.0, num=11), clfs_acc, label="Validation Accuracy")
    plt.plot(np.linspace(-5.0, 5.0, num=11), clfs_acc_train, label="Test Accuracy")
    plt.xlabel(r'$\gamma = 10^{i}$')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower left")
    plt.axis([-5, 5, 0, 1.05])
    plt.show()
    return clfs_acc


def sect_a_plots(X_train, y_train):
    clf_lin = svm.SVC(C=1000, kernel='linear')
    clf_lin.fit(X_train, y_train)
    clf_quad = svm.SVC(C=1000, kernel='poly', degree=2, coef0=1)
    clf_quad.fit(X_train, y_train)
    clf_rbf = svm.SVC(C=1000, kernel='rbf')
    clf_rbf.fit(X_train, y_train)
    create_plot(X_train, y_train, clf_lin)
    plt.show()
    create_plot(X_train, y_train, clf_quad)
    plt.show()
    create_plot(X_train, y_train, clf_rbf)
    plt.show()


def sect_b_plots(X_train, y_train, X_val, y_val):
    Y = linear_accuracy_per_C(X_train, y_train, X_val, y_val)

    plt.show()
    clf_lin = svm.SVC(C=1, kernel='linear')
    clf_lin.fit(X_train, y_train)
    create_plot(X_val, y_val, clf_lin)
    plt.title('C = 1')
    plt.show()
    clf_lin = svm.SVC(C=100, kernel='linear')
    clf_lin.fit(X_train, y_train)
    create_plot(X_val, y_val, clf_lin)
    plt.title('C = 100')
    plt.show()
    clf_lin = svm.SVC(C=100000, kernel='linear')
    clf_lin.fit(X_train, y_train)
    create_plot(X_val, y_val, clf_lin)
    plt.title('C = 100000')
    plt.show()


def sect_c_plots(X_train, y_train, X_val, y_val):
    rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val)
    clf_rbf = svm.SVC(C=10, kernel='rbf', gamma=1).fit(X_train,y_train)
    clf_rbf.fit(X_train, y_train)
    create_plot(X_val, y_val, clf_rbf)
    plt.title(r'$\gamma = 1$')
    plt.show()
    clf_rbf = svm.SVC(C=10, kernel='rbf', gamma=100)
    clf_rbf.fit(X_train, y_train)
    create_plot(X_val, y_val, clf_rbf)
    plt.title(r'$\gamma = 100$')
    plt.show()


training_data, training_labels, validation_data, validation_labels = get_points()
validation_labels = 2*validation_labels -1
training_labels = 2*training_labels - 1
# sect_a_plots(training_data, training_labels)
# sect_b_plots(training_data, training_labels, validation_data, validation_labels)
# sect_c_plots(training_data, training_labels, validation_data, validation_labels)
