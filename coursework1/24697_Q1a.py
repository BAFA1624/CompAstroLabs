import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import time

def normalize(arr):
    assert(type(arr) == np.ndarray)
    return arr / max(arr)

# Single step of the 4th order Runge-Kutta method
def RK4_step(f, t, h, r):
    k1 = np.array([h * func(t, *r) for func in f])
    k2 = np.array([h * func(t + h/2, *(r + k1/2)) for func in f])
    k3 = np.array([h * func(t + h/2, *(r + k2/2)) for func in f])
    k4 = np.array([h * func(t + h, *(r + k3)) for func in f])
    return r + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

def cRK4(f, y0, t0, tf, N):
    step = abs(tf - t0) / (N - 1)
    result = [y0]
    times = [t0]

    for i in range(N):
        result.append(RK4_step(f, times[-1], step, result[-1]))
        times.append(times[-1] + step)
        #exit()

    return np.array(times), np.array(result)

# Constants
s_year = 3.1536e7      # Seconds in a year
G = 6.6743e-11         # Gravitational constant
G_year = G * s_year**2 # Gravitational constant using years for time unit
M = 1.989e30           # Mass of Sun in kg
GM_year = G_year * M   # Combine G_year & M into single constant

# Two sets of coupled eqns, 1 set each for x & y coordinates
def dX(t, x, vx, y, vy):
    return vx
def dV_X(t, x, vx, y, vy):
    return -(GM_year * x) / pow(x**2 + y**2, 1.5)

def dY(t, x, vx, y, vy):
    return vy
def dV_Y(t, x, vx, y, vy):
    return -(GM_year * y) / pow(x**2 + y**2, 1.5)

# Simulate for 100 years
# All time units converted from seconds -> years

# Initial conditions
f = np.array([ dX, dV_X, dY, dV_Y ])
r = np.array([ 5.2e12, 0, 0, 2.775e10 ])

# Apply Runge-Kutta for x values
# vals_Nk returns:
#  - Array of time arrays.
#  - Array of arrays of [x, vx, y, vy] for each time point.
print("Calculating N = 100...")
start = time.time()
t_100, vals_100 = cRK4(f, r, 0, 100, 100)
print(f"Done. ({(time.time() - start):.3f}s)")
print("Calculating N = 1k...")
start = time.time()
t_1k, vals_1k = cRK4(f, r, 0, 100, 1000)
print(f"Done. ({(time.time() - start):.3f}s)")
print("Calculating N = 10k...")
start = time.time()
t_10k, vals_10k = cRK4(f, r, 0, 100, 10000)
print(f"Done. ({(time.time() - start):.3f}s)")

# Plot result

fig = plt.figure(figsize=(16, 8))
grid = gs.GridSpec(3,2)

ax1 = fig.add_subplot(grid[0, :])
ax2 = fig.add_subplot(grid[1, :])
ax3 = fig.add_subplot(grid[2, :])

ax1.set_title("N = 100")
ax1.plot(normalize(vals_100[:, 0]), normalize(vals_100[:, 2]),
         marker='o', ls='none', markersize=1.4)
ax1.set_xlabel(f"x / {max(vals_100[:, 0]):.2e}m")
ax1.set_ylabel(f"y / {max(vals_100[:, 2]):.2e}m")
ax1.axhline(0, c='k')
ax1.axvline(0, c='k')

ax2.set_title("N = 1k")
ax2.plot(normalize(vals_1k[:, 0]), normalize(vals_1k[:, 2]),
         marker='o', ls='none', markersize=1.4)
ax2.set_xlabel(f"x / {max(vals_1k[:, 0]):.2e}m")
ax2.set_ylabel(f"y / {max(vals_1k[:, 2]):.2e}m")
ax2.axhline(0, c='k')
ax2.axvline(0, c='k')

ax3.set_title("N = 10k")
ax3.plot(normalize(vals_10k[:, 0]), normalize(vals_10k[:, 2]),
         marker='o', ls='none', markersize=1.4)
ax3.set_xlabel(f"x / {max(vals_10k[:, 0]):.2e}m")
ax3.set_ylabel(f"y / {max(vals_10k[:, 2]):.2e}m")
ax3.axhline(0, c='k')
ax3.axvline(0, c='k')

plt.tight_layout()
plt.show()

