import numpy as np
import matplotlib.pyplot as plt

N = 200000
n = 20

mat = np.random.randint(2, size=(N, n))
err = np.abs(np.mean(mat, axis=1) - 0.5)
eps = np.linspace(0,1, 50)
empirical = np.zeros(50)
for i in range(50):
    empirical[i] = sum(err > eps[i])/N


plt.plot(eps, empirical, label='empirical')
plt.plot(eps, 2*np.exp(-2*n*eps**2), label='hoeffding')
plt.legend(loc='best')
plt.show()