import numpy as np
import matplotlib.pyplot as plt
import time

def vRK4(f, y0, x1, xN, N, coeffs=None, t_func=None):
    assert( np.shape(f) == np.shape(y0) )

    # Calls each function in f_arr
    #with it's respective coeffs in c_arr.
    def caller(f_arr, t, p_arr, c_arr):
        return np.array([f(t, [*p, *c]) for f, p, c in zip(f_arr, p_arr, c_arr)])
    def constant_step(t, x1, xN, N):
        return t + ( ( xN - x1 ) / ( N - 1 ) )

    step_func = None
    if t_func is None:
        step_func = constant_step
    else:
        step_func = t_func

    if coeffs is None:
        coeffs = [[] for i in range(np.shape(y0)[0])]

    assert( np.shape(f)[0] == np.shape(coeffs)[0] )

    result = np.zeros((N, len(y0) + 1))

    # Set initial conditions
    result[0, 1:] = y0 
    result[0, 0] = x1

    k1 = np.zeros_like(y0)
    k2 = np.zeros_like(y0)
    k3 = np.zeros_like(y0)
    k4 = np.zeros_like(y0)
    y = np.zeros_like(y0)

    for i in range(1, np.shape(result)[0]):
        t = result[i - 1, 0]
        result[i, 0] = step_func(t, x1, xN, N)
        step = abs(result[i, 0] - result[i-1, 0])

        # Construct initial coefficients
        params = np.array([ result[i-1, 1:] for j in range(len(f)) ])

        k1 = step * caller(f, t, params, coeffs)
        k2 = step * caller(f, t + step/2, params + k1/2, coeffs)
        k3 = step * caller(f, t + step/2, params + k2/2, coeffs)
        k4 = step * caller(f, t + step, params + k3, coeffs)

        y = params + (0.166666666) * (k1 + 2*k2 + 2*k3 + k4)

        result[i, 1:] = np.diag(y)

    return result

s_year = 3.1536e7

G = 6.6743e-11 # m^3 kg^-1 s^-2
G_year = G * s_year**2
M = 1.989e30   # kg
GM = G * M
GM_year = G_year * M

def dX(t, r):
    x, vx, y, vy = r
    return [vx, vx,
            y, vy]
def dV_X(t, r):
    x, vx, y, vy = r
    return [x, -(GM_year * x) / pow(x**2 + y**2, 1.5),
            y, vy]
def dY(t, r):
    x, vx, y, vy = r
    return [x, vx,
            vy, vy]
def dV_Y(t, r):
    x, vx, y, vy = r
    return [x, vx,
            y, -(GM_year * y) / pow(x**2 + y**2, 1.5)]

f = [dX, dV_X, dY, dV_Y]
# 2.775e10 m/year
y0 = [5.2e12, 0, 0, 2.775e10]

start = time.time()
model = vRK4(f, y0, 0, 300, 1000000)
print(f"Model took: {time.time() - start}s.")

t = model[:, 0]
x = model[:, 1] / 5.2e12
vx = model[:, 2]
y = model[:, 3] / 5.2e12
vy = model[:, 4]


# Plotting

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(421)
ax2 = fig.add_subplot(422)
ax3 = fig.add_subplot(423)
ax4 = fig.add_subplot(424)
ax5 = fig.add_subplot(425)
ax6 = fig.add_subplot(426)

ax1.plot(t, x)
ax1.set_xlabel("t")
ax1.set_ylabel("x")

ax2.plot(t, vx)
ax2.set_xlabel("t")
ax2.set_ylabel("vx")

ax3.plot(t, y)
ax3.set_xlabel("t")
ax3.set_ylabel("y")

ax4.plot(t, vy)
ax4.set_xlabel("t")
ax4.set_ylabel("vy")

ax5.plot(x, y)
ax5.set_xlabel("x")
ax5.set_ylabel("y")
ax5.axhline(0, c='k', ls='--', linewidth=1.2)
ax5.axvline(0, c='k', ls='--', linewidth=1.2)

ax6.plot(vx, vy)
ax6.set_xlabel("vx")
ax6.set_ylabel("vy")

plt.tight_layout()
fig.savefig(fname="6")

plt.show()
