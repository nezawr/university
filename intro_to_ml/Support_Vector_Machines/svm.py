#################################
# Your name:
# Nazar Aburas
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
    clf_linear = svm.SVC(C=1000, kernel='linear')
    clf_quadratic = svm.SVC(C=1000, kernel='poly', degree=2, coef0=1)
    clf_rbf = svm.SVC(C=1000, kernel='rbf')

    clf_linear.fit(X_train, y_train)
    clf_quadratic.fit(X_train, y_train)
    clf_rbf.fit(X_train, y_train)

    return np.stack((clf_linear.n_support_, clf_quadratic.n_support_, clf_rbf.n_support_))


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C_arr = np.logspace(-5.0, 5.0, num=11)
    clf_acc_train = np.array([svm.SVC(C=c, kernel='linear').fit(X_train, y_train).score(X_train, y_train) for c in C_arr])
    clf_acc_val = np.array([svm.SVC(C=c, kernel='linear').fit(X_train, y_train).score(X_val, y_val) for c in C_arr])

    plt.plot(np.linspace(-5.0, 5.0, num=11), clf_acc_train, label="Train Accuracy")
    plt.plot(np.linspace(-5.0, 5.0, num=11), clf_acc_val, label="Validation Accuracy")
    plt.xlabel(r'$C = 10^{i}$')
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.axis()
    plt.show()

    return clf_acc_val

    


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    
    # C = 10
    gammas = np.logspace(-5.0, 5.0, num=11)

    clf_acc_val = np.array([svm.SVC(C=10, kernel="rbf", gamma=g).fit(X_train, y_train).score(X_val, y_val) for g in gammas])

    clf_acc_train = np.array([svm.SVC(C=10, kernel="rbf", gamma=g).fit(X_train, y_train).score(X_train, y_train) for g in gammas])

    plt.plot(np.linspace(-5, 5, num=11), clf_acc_val, label="Validation Accuracy")
    plt.plot(np.linspace(-5, 5, num=11), clf_acc_train, label="Train Accuracy")

    plt.xlabel(r'$\gamma = 10^{i}$')
    plt.ylabel("Accuracy")
    plt.axis([-5, 5, 0, 1.1])
    plt.legend(loc="lower right")
    plt.show()

    return clf_acc_val


def section_a_plots(X_train, y_train):

    clf_linear = svm.SVC(C=1000, kernel='linear')
    clf_quadratic = svm.SVC(C=1000, kernel='poly', degree=2, coef0=1)
    clf_rbf = svm.SVC(C=1000, kernel='rbf')

    clf_linear.fit(X_train, y_train)
    clf_quadratic.fit(X_train, y_train)
    clf_rbf.fit(X_train, y_train)

    create_plot(X_train, y_train, clf_linear)
    plt.show()
    create_plot(X_train, y_train, clf_quadratic)
    plt.show()
    create_plot(X_train, y_train, clf_rbf)
    plt.show()


def section_b_plots(X_train, y_train, X_val, y_val):
    Y = linear_accuracy_per_C(X_train, y_train, X_val, y_val)

    clf_linear = svm.SVC(C=1, kernel='linear')
    clf_linear.fit(X_train, y_train)
    create_plot(X_val, y_val, clf_linear)
    plt.title('C = 1')
    plt.show()

    clf_linear = svm.SVC(C=100, kernel='linear')
    clf_linear.fit(X_train, y_train)
    create_plot(X_val, y_val, clf_linear)
    plt.title('C = 100')
    plt.show()
    
    clf_linear = svm.SVC(C=10000, kernel='linear')
    clf_linear.fit(X_train, y_train)
    create_plot(X_val, y_val, clf_linear)
    plt.title('C = 10000')
    plt.show()


def section_c_plots(X_train, y_train, X_val, y_val):
    rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val)

    clf_rbf = svm.SVC(C=10, kernel='rbf', gamma=0.01).fit(X_train, y_train)
    create_plot(X_val, y_val, clf_rbf)
    plt.title(r'$\gamma = 0.01$')
    plt.show()

    clf_rbf = svm.SVC(C=10, kernel='rbf', gamma=1).fit(X_train,y_train)
    create_plot(X_val, y_val, clf_rbf)
    plt.title(r'$\gamma = 1$')
    plt.show()

    clf_rbf = svm.SVC(C=10, kernel='rbf', gamma=100).fit(X_train,y_train)
    create_plot(X_val, y_val, clf_rbf)
    plt.title(r'$\gamma = 100$')
    plt.show()




if __name__ == '__main__':
    t_d, t_l, v_d, v_l = get_points()

    #section_a_plots(t_d, t_l)
    #section_b_plots(t_d, t_l, v_d, v_l)
    section_c_plots(t_d, t_l, v_d, v_l)



    



    