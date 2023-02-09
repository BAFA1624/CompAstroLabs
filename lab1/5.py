import numpy as np
import matplotlib.pyplot as plt

def simpsons_rule(func, a, b):
    return ( ( b - a ) / 6 ) * ( func(a) + 4 * func((a+b)/2) + func(b) )

def definite_integral(func, l_bound, u_bound, N):
    area = 0
    step_size = (u_bound - l_bound) / ( N - 1 )
    bounds = np.arange(l_bound, u_bound, step_size)
    for i in range(len(bounds)-1):
        area += simpsons_rule(func, bounds[i], bounds[i+1])
    return area

def J(x, m):
    def f(y):
        return np.cos(m * y - x * np.sin(y))

    return (1 / np.pi) * definite_integral(f, 0, np.pi, 10000)

xvals = np.arange(0, 21, 0.1)
J_0 = J(xvals, 0)
J_1 = J(xvals, 1)
J_2 = J(xvals, 2)

plt.plot(xvals, J_0, label="J_0")
plt.plot(xvals, J_1, label="J_1")
plt.plot(xvals, J_2, label="J_2")
plt.axhline(0, c='k', linewidth=0.5)
plt.axhline(0.5, c='k', linewidth=0.5, ls='--')
plt.axvline(0, c='k', linewidth=0.5)
plt.legend()
plt.show()

def I(r, wavelength, I0, focal_length):
    x = (np.pi * r) / (wavelength * focal_length)
    return I0 * ( 2 * J(x, 1) / x)**2

rvals = np.linspace(-25e-6, 25e-6, 1000)

intensities = I(rvals, 50e-8, 1, 10)

plt.plot(rvals, intensities)
plt.show()
