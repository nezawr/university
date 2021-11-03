from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]



#section (a)
def kNN(train, labels, query, k):
    
    #calculate indices of k nearest neighbours (closest images).
    distances = calc_distances(train, query)
    nearest_neighbor_ids = np.argsort(distances)[:k]

    #get the labels matching the indices using numpy.take function
    nearest_neighbor_labels = np.take(labels,nearest_neighbor_ids).astype(int)

    #count the labels with bincount  and choose the most frequence one using argmax
    return np.bincount(nearest_neighbor_labels).argmax()
    

def calc_distances(train_set, query):
    distances = np.zeros(len(train_set))
    for i in range(len(train_set)):
        distances[i] = np.linalg.norm(train[i] - query)
    return distances
 
#section (b + c)
test_predictions = np.zeros(len(test))
k = np.arange(1, 101)
k_accuracy = np.zeros(100)

for i in range(100):
    for j in range(len(test)):
        test_predictions[j] = kNN(train[:1000], train_labels[:1000], test[j], k[i])
    k_accuracy[i] = np.linalg.norm(test_predictions == test_labels.astype(int), 1)/len(test)
    #section b
    if (k[i] == 10):
        print(k_accuracy[i])

plt.plot(k, k_accuracy)
plt.show()


#section (d)
n = np.arange(100, 5001, 100)
n_accuracy = np.zeros(len(n))
for i in range(len(n)):
    for j in range(len(test)):
        test_predictions[j] = kNN(train[:n[i]], train_labels[:n[i]], test[j], 1)
    n_accuracy[i] = np.linalg.norm(test_predictions == test_labels.astype(int), 1)/len(test)

plt.plot(n, n_accuracy)
plt.show()


    





