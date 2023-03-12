import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import time

def normalize(arr):
    assert(type(arr) == np.ndarray)
    return arr / max(arr)

# Variational time-step, vectorised, 4th-order Runge-Kutta Method
# Arguments:
# - f: Array of functions for each equation, e.g. if you have set of coupled eqns.
#      Functions are of the form f(t, r, [additional coefficients])
# - y0: Initial set of values.
# - t0: Start of time range to simulate.
# - tmax: End of time range to simulate.
# - init_step: Initial step size before adjustment due to errors.
# - err_scale: Max. power of error relative to current calculated values.
#              e.g. if err_scale = -6 & the most recent calcualted value is 1.3x10^10,
#                   the max error would be 1.3x10^4.
def vRK4(f, y0, t0, tmax, init_step=0.001, err_scale=-6):
    if err_scale > 0:
        raise RuntimeError(f"err_scale ({err_scale}) must < 0")
    
    # Convert to numpy arrays
    f = np.array(f)
    y0 = np.array(y0)

    # Check input shapes are compatible
    assert( np.shape(f) == np.shape(y0) )

    # Process one RK4 step
    def RK4_step(t, h, y_c):
        k1 = np.array([ h * func(t, y_c) for func in f ])
        k2 = np.array([ h * func(t + (0.5 * h), y_c + (0.5 * k1)) for func in f ])
        k3 = np.array([ h * func(t + (0.5 * h), y_c + (0.5 * k2)) for func in f ])
        k4 = np.array([ h * func(t + h, y_c + k3) for func in f ])
        return np.array(y_c + (1/6) * (k1 + 2*k2 + 2*k3 + k4))
    # Get array of closest powers of 10 for items in the input.
    def pow_10(x):
        result = np.zeros(x.shape)
        not_zero = np.not_equal(np.abs(x), 0)
        result[not_zero] = np.floor(np.log10(np.abs(x[not_zero])))
        return result

    N = int(np.floor(abs(tmax - t0) / init_step))
    if N == 0:
        raise RuntimeError( f"floor(abs({tmax} (tmax) - {t0} (t0) / {init_step} (init_step) = {N} (N), must not equal 0." )

    # Set initial conditions
    # Result has the structure: time, parameters for each respective y0 value.
    times, result = [t0], [y0]

    # Array of max errors for each parameter
    max_err_scale = np.array([10 ** (np.full_like(y0, err_scale) + pow_10(y0)) for func in f])
    epsilon_arr = [y0 * np.full_like(y0, 10 ** err_scale)]

    # The current time step value is results[i - 1, 0]
    # The next t value is set to results[i, 0] when the step size is determined.
    step = init_step
    i, t = 1, t0
    while t <= tmax:
        t = times[-1]
        params = result[-1]

        # Evaluate if error in step is too large, adjust step accordingly.
        err_too_large = True
        while err_too_large:
            # Two steps of size h
            y = RK4_step(t, step, params)
            y1 = RK4_step(t, step, y)
            # One step of size 2h
            y2 = RK4_step(t, 2 * step, params)

            # Check error is within tolerance
            maximum = np.maximum(np.abs(y1), np.abs(y2))
            max_err = 10 ** ( np.full_like(y0, err_scale) + pow_10(maximum) )
            err = (1/30) * np.abs(y2 - y1)
            err_too_small = np.any(np.less(err, 0.5 * max_err))
            err_too_large = np.any(np.greater(err, max_err))

            # Adjust step size
            if err_too_large:
                step /= 2
            elif err_too_small and ~np.any(np.greater(100 * err, max_err)):
                step *= 2
        
        # Write result
        times.append(t + step)
        result.append(y)
        i += 1

    return np.array(times), np.array(result)

# Constants
s_year = 3.1536e7      # Seconds in a year
G = 6.6743e-11         # Gravitational constant
G_year = G * s_year**2 # Gravitational constant using years for time unit
M = 1.989e30           # Mass of Sun in kg
GM_year = G_year * M   # Combine G_year & M into single constant

# dX and dV_X, and dY and dV_Y are coupled sets of differential eqns
# for their respective cartesian coordinates
def dX(t, r):
    x, vx, y, vy = r
    return vx
def dV_X(t, r):
    x, vx, y, vy = r
    return -(GM_year * x) / pow(x**2 + y**2, 1.5)
def dY(t, r):
    x, vx, y, vy = r
    return vy
def dV_Y(t, r):
    x, vx, y, vy = r
    return -(GM_year * y) / pow(x**2 + y**2, 1.5)

f = [dX, dV_X, dY, dV_Y]
# 2.775e10 m/year
r = [5.2e12, 0, 0, 2.775e10]

print("Calculating err_scale=0...")
start=time.time()
t_0, vals_0 = vRK4(f, r, 0, 80, err_scale=0)
print(f"Done. ({time.time()-start:.3f}s)")
print("Calculating err_scale=-3...")
start=time.time()
t_milli, vals_milli = vRK4(f, r, 0, 80, err_scale=-3)
print(f"Done. ({time.time()-start:.3f}s)")
print("Calculating err_scale=-6...")
t_micro, vals_micro = vRK4(f, r, 0, 80, err_scale=-6)
print(f"Done. ({time.time()-start:.3f}s)")

print("\n")

print(f"err_scale = 0 -> steps taken = {len(t_0)})")
print(f"err_scale = -3 -> steps taken = {len(t_milli)}")
print(f"err_scale = -6 -> steps taken = {len(t_micro)}")

# Plot result

fig = plt.figure(figsize=(16, 16))
grid = gs.GridSpec(3,2)

ax1 = fig.add_subplot(grid[0, :])
ax2 = fig.add_subplot(grid[1, :])
ax3 = fig.add_subplot(grid[2, :])

ax1.set_title("err_scale = 0")
ax1.plot(normalize(vals_0[:, 0]), normalize(vals_0[:, 2]),
         marker='x', ls='--', markersize=6)
ax1.set_xlabel(f"x / {max(vals_0[:, 0]):.2e}m")
ax1.set_ylabel(f"y / {max(vals_0[:, 2]):.2e}m")
ax1.axhline(0, c='k')
ax1.axvline(0, c='k')

ax2.set_title("err_scale = -3")
ax2.plot(normalize(vals_milli[:, 0]), normalize(vals_milli[:, 2]),
         marker='x', ls='--', markersize=6)
ax2.set_xlabel(f"x / {max(vals_milli[:, 0]):.2e}m")
ax2.set_ylabel(f"y / {max(vals_milli[:, 2]):.2e}m")
ax2.axhline(0, c='k')
ax2.axvline(0, c='k')

ax3.set_title("err_scale = -6")
ax3.plot(normalize(vals_micro[:, 0]), normalize(vals_micro[:, 2]),
         marker='x', ls='--', markersize=6)
ax3.set_xlabel(f"x / {max(vals_micro[:, 0]):.2e}m")
ax3.set_ylabel(f"y / {max(vals_micro[:, 2]):.2e}m")
ax3.axhline(0, c='k')
ax3.axvline(0, c='k')

plt.tight_layout()
plt.show()
