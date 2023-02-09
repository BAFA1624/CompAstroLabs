import numpy as np

memoise = { 0: 1, 1: 1 }

def factorial(n):
    assert(n >= 0)

    total = 1
    current_n = n
    while memoise.get(current_n) is None:
        total *= current_n
        current_n -= 1
    total *= memoise.get(current_n)

    memoise[n] = total

    return total

def trunc_e(x, n_terms):
    assert(n_terms >= 1)

    estimate = 1
    for i in range(1, n_terms):
        estimate += (x**i) / factorial(i)

    return estimate

#print(factorial(4))
print(trunc_e(3, 20))

