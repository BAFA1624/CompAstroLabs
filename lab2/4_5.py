import numpy as np
import matplotlib.pyplot as plt

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

        # Construct initial coefficients
        params = np.array([ result[i-1, 1:] for j in range(len(f)) ])

        k1 = step * caller(f, t, params, coeffs)
        k2 = step * caller(f, t + step/2, params + k1/2, coeffs)
        k3 = step * caller(f, t + step/2, params + k2/2, coeffs)
        k4 = step * caller(f, t + step, params + k3/2, coeffs)

        y = params + (0.166666666) * (k1 + 2*k2 + 2*k3 + k4)

        result[i, 1:] = np.diag(y)

    return result

def s_dx(t, r):
    x, y, omega = r
    return [y, y]
def s_dy(t, r):
    x, y, omega = r
    return [x, -(omega**2)*x]

SHM = vRK4([s_dx, s_dy], [1, 0], 0, 50, 10000, [[1], [1]])

plt.title("Simple Harmonic Motion")
plt.plot(SHM[:, 0], SHM[:, 1], "r--", label="Position")
plt.plot(SHM[:, 0], SHM[:, 2], "c--", label="Velocity")
plt.legend()
plt.show()

def a_dx(t, r):
    x, y, omega = r
    return [y, y]
def a_dy(t, r):
    x, y, omega = r
    return [x, -(omega**2) * (x**3)]

AHM = vRK4([a_dx, a_dy], [2, 0], 0, 50, 100000, [[1], [1]])

plt.title("Anharmonic Motion")
plt.plot(AHM[:, 0], AHM[:, 1], "r--", label="Position")
plt.plot(AHM[:, 0], AHM[:, 2], "c--", label="Velocity")
plt.legend()
plt.show()

plt.title("Phase Diagram")
plt.plot(AHM[:, 1], AHM[:, 2], "--")
plt.show()
