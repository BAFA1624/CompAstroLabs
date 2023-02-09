

def NR_step(func, x_1, x_2):
    return x_2 - func(x_2) * ( ( x_2 - x_1 ) / ( func(x_2) - func(x_1) ) )

def NR(func, x_1, x_2, n_steps):

    for i in range(n_steps):
        tmp = x_1
        x_1 = NR_step(func, x_1, x_2)
        x_2 = tmp

    return x_2 

def P(x):
    return 924 * x**6 - 2772 * x**5 + 3150 * x**4 - 1680 * x**3 + 420 * x**2 - 42 * x + 1

print(NR(P, 0, 0.1, 10))

