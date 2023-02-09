import numpy as np
import matplotlib.pyplot as plt

def simpsons_rule(func, a, b):
    return ( ( b - a ) / 6 ) * ( func(a) + 4 * func((a+b)/2) + func(b) )

def definite_integral(func, l_bound, u_bound, step_size):
    area = 0
    bounds = np.arange(l_bound, u_bound, step_size)
    for i in range(len(bounds)-1):
        area += simpsons_rule(func, bounds[i], bounds[i+1])
    return area

def f(t):
    return np.exp(-(t**2))

def quadratic(x):
    return -(x - 1)**2 + 1

upper_bounds = np.linspace(0, 4, 10)
lower_bounds = np.zeros_like(upper_bounds)

areas = []
for l, u in zip(lower_bounds, upper_bounds):
    areas.append(definite_integral(f, l, u, 0.1))

plt.plot(upper_bounds, areas, marker='o', markersize=1.2, ls='none', c='k')
plt.show()
