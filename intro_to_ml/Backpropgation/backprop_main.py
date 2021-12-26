import backprop_data

import backprop_network

import numpy as np
import matplotlib.pyplot as plt


def part1():
    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)


def part2():
    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
    rates = [0.001, 0.01, 0.1, 1, 10, 100]
    train_loss = [None] * 6
    test_accuracy = [None] * 6
    train_accuracy = [None] * 6

    for i in range(6):
        net = backprop_network.Network([784, 40, 10])
        train_accuracy[i], train_loss[i], test_accuracy[i] = net.SGD(training_data, epochs=30, mini_batch_size=10,
                                                                     learning_rate=rates[i], test_data=test_data)
        

    for i in range(6):
        plt.plot(np.arange(30), train_accuracy[i], label="rate = {}".format(rates[i]))

    plt.xlabel('epochs')
    plt.ylabel('Train Accuracy')
    plt.legend()
    plt.show()

    for i in range(6):
        plt.plot(np.arange(30), train_loss[i], label="rate = {}".format(rates[i]))

    plt.xlabel('epochs')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.show()

    for i in range(6):
        plt.plot(np.arange(30), test_accuracy[i], label="rate = {}".format(rates[i]))

    plt.xlabel('epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.show()


def part3():
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

def part4():
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = backprop_network.Network([784, 2048,  10])
    net.SGD(training_data, epochs=30 ,mini_batch_size=20, learning_rate=0.1, test_data=test_data)


# part1
#()
#part2()
# part3()
part4()