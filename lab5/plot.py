import matplotlib.pyplot as plt
import numpy as np


def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            vals = [np.float64(x) for x in l.strip().split(',')]
            x.append(vals[0])
            y.append(vals[1])
    return np.array(x), np.array(y)


def f(x):
    return (2/np.pi) * np.sin(x)**2


x, y = read_data("rejection_method.csv")
plt.title("rejection method")
plt.plot(x, y, 'ko', ls='none', markersize=1)
plt.plot(x, f(x), 'r--')
plt.show()

x, y = read_data("importance_sampling.csv")
plt.title("importance sampling")
plt.plot(x, y, 'ko', ls='none', markersize=1)
plt.plot(x, f(x), 'r--')
plt.show()
