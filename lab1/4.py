import numpy as np

h = 6.62607015e-34
k = 1.380649e-23
c = 2.99792458e8

def simpsons_rule(func, a, b):
    return ( ( b - a ) / 6 ) * ( func(a) + 4 * func((a+b)/2) + func(b) )

def definite_integral(func, l_bound, u_bound, step_size):
    area = 0
    bounds = np.arange(l_bound, u_bound, step_size)
    for i in range(len(bounds)-1):
        area += simpsons_rule(func, bounds[i], bounds[i+1])
    return area

def f(x):
    print(x)
    if x == 0:
        return 0
    return x**3 / (np.exp(x)-1)

def SB(u_bound, step_size):
    return ((2 * np.pi * (k**4)) / (h**3 * c**2)) * definite_integral(f, 0, u_bound, step_size)

print(SB(1000, 0.1))
