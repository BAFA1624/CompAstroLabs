import numpy as np
import matplotlib.pyplot as plt

def RK4(f, y0, x1, xN, N, coeffs):
    step = (xN - x1) / (N - 1)
    t, y = x1, y0
    result = [(t, y)]
    while t < xN:
        k1 = step * f(y, t, *coeffs)
        k2 = step * f(y + k1/2, t + step/2, *coeffs)
        k3 = step * f(y + k2/2, t + step/2, *coeffs)
        k4 = step * f(y + k3/2, t + step, *coeffs)
        y += (0.166666666) * (k1 + 2*k2 + 2*k3 + k4)
        t += step
        result.append((t, y))
    return np.array(result)

def vRK4(f, y0, x1, xN, N, coeffs):
    assert( np.shape(f) == np.shape(y0) )
    assert( np.shape(f)[0] == np.shape(coeffs)[0] )

    # Calls each function in f_arr
    #with it's respective coeffs in c_arr.
    def caller(f_arr, t, p_arr, c_arr):
        return np.array([f(t, [*p, *c]) for f, p, c in zip(f_arr, p_arr, c_arr)])

    step = (xN - x1) / (N - 1)

    times = np.linspace(x1, xN, N)
    result = np.zeros((N, len(y0) + 1))

    # Set initial conditions
    result[0, 1:] = y0 
    result[:, 0] = times

    for i in range(1, len(times)):
        t = times[i - 1]

        # [ [t, x, y, rabbit_coeffs],
        #   [t, x, y, fox_coeffs] ]
        # Construct initial coefficients
        params = np.array([ result[i-1, 1:] for j in range(len(f)) ])

        k1 = step * caller(f, t, params, coeffs)
        k2 = step * caller(f, t + step/2, params + k1/2, coeffs)
        k3 = step * caller(f, t + step/2, params + k2/2, coeffs)
        k4 = step * caller(f, t + step, params + k3/3, coeffs)

        y = params + (0.166666666) * (k1 + 2*k2 + 2*k3 + k4)
        result[i, 1:] = np.diag(y)

    return result

def rabbit_pop(t, r):
    x, y, a, b = r
    return [a * x - b * x * y, y]
def fox_pop(t, r):
    x, y, gamma, delta = r
    return [x, gamma * x * y - delta * y]

f = [rabbit_pop, fox_pop]
y0 = [2, 2]
coeffs = [[1, 0.5],
          [0.5, 2]]

model = vRK4(f, y0, 0, 50, 50000, coeffs)

plt.plot(model[:, 0], model[:, 1], c='r', label='Rabbits')
plt.plot(model[:, 0], model[:, 2], c='c', label='Foxes')
plt.axhline(7)
plt.legend()
plt.show()
