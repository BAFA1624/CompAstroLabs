import numpy as np
import matplotlib.pyplot as plt
import time

def vRK4(f, y0, t0, tmax, init_step=1, err_scale=-6, coeffs=None):
    if err_scale >= 0:
        raise RuntimeError(f"err_scale ({err_scale}) must < 0")
    f = np.array(f)
    y0 = np.array(y0)

    # Check input shapes are compatible
    assert( np.shape(f) == np.shape(y0) )
    if coeffs is None:
        coeffs = [[] for i in range(np.shape(y0)[0])]
    assert( np.shape(f)[0] == np.shape(coeffs)[0] )

    # Calls each function in f_arr with correct arguments for current step.
    def caller(t, p_arr):
        return np.array([func(t, [*p, *c]) for func, p, c in zip(f, p_arr, coeffs)])
    # Process one RK4 step
    def vRK4_step(t, h, y_c):
        k1 = h * caller(t + h, y_c)
        k2 = h * caller(t + h/2, y_c + k1/2)
        k3 = h * caller(t + h/2, y_c + k2/2)
        k4 = h * caller(t, y_c + k3)
        return y_c + (0.166666666) * (k1 + 2*k2 + 3*k3 + k4)
    def expand_array(arr, n_rows):
        return np.pad(arr, ((0, n_rows), (0, )), 'constant', 0)
    def pow_10(x):
        result = np.zeros(x.shape)
        not_zero = np.not_equal(np.abs(x), 0)
        result[not_zero] = np.floor(np.log10(np.abs(x[not_zero])))
        return result

    # Initialise results array
    N = int(np.floor(abs(tmax - t0) / init_step))
    if N == 0:
        raise RuntimeError( f"floor(abs({tmax} (tmax) - {t0} (t0) / {init_step} (init_step) = {N} (N), must not equal 0." )
    result_shape = (N, len(y0) + 1)
    result = np.zeros(result_shape)

    # Set initial conditions
    result[0, 1:] = y0 
    result[0, 0] = t0
    result[1, 0] = t0 + init_step

    # Initialise arrays for RK4 method
    k1 = np.zeros_like(y0)
    k2 = np.zeros_like(y0)
    k3 = np.zeros_like(y0)
    k4 = np.zeros_like(y0)
    y = np.zeros_like(y0)

    # Array of max errors for each parameter
    max_err_scale = np.array([10 ** (np.full_like(y0, err_scale) + pow_10(y0)) for func in f])
    epsilon_arr = [y0 * np.full_like(y0, 10 ** err_scale)]

    i, t = 1, t0
    while t < tmax:
        t = result[i - 1, 0]
        step = abs(result[i, 0] - result[i-1, 0])

        # Construct initial coefficients
        params = np.array([ result[i-1, 1:] for j in range(len(f)) ])

        err_too_large = True
        while err_too_large:
            #print("step: ", step)
            # Two steps of size h
            y = vRK4_step(t, step, params)
            y1 = vRK4_step(t, step, y)

            # One step of size 2h
            y2 = vRK4_step(t, 2 * step, params)

            diff = np.abs(y2 - y1)
            maximum = np.maximum(np.abs(y1), np.abs(y2))
            max_err = 10 ** (np.full_like(maximum, err_scale) + pow_10(maximum))
            #print("max_err", max_err)

            # Check error is within tolerance
            err = (0.03333333333333) * diff
            #print("err", err)
            #print("err", err)
            #print(np.greater(err, epsilon_arr))
            err_too_small = np.any(np.less(err, 0.0001 * max_err))
            err_too_large = np.any(np.greater(err, max_err))
            #print("err_too_small", np.less(err, 0.01 * max_err))
            #print("err_too_large", np.greater(err, max_err))

            # Adjust step size
            if err_too_large:
                step /= 2
                #print(f"Step shrunk, {step * 2} -> {step}")
            elif err_too_small and ~np.any(np.greater(100 * err, max_err)):
                step *= 2
                print(f"Step increased, {step / 2} -> {step}")

        # Check result array size & adjust if needed
        # Current row = i
        required_rows = int(i + np.floor(abs(tmax - t) / step))
        #print(required_rows)
        
        # Total rows in result = result_shape[0]
        # Too few rows:
        #   Expand by 2 * (predicted rows - total rows)
        # Too many rows:
        #   If 2 * predicted rows < total rows:
        #     Shrink to predicted rows
        if required_rows > result_shape[0] or 2 * required_rows < result_shape[0]:
            result = np.resize(result, (required_rows, result_shape[1]))
            result_shape = np.shape(result)

        # Calculate next time value from step
        result[i, 0] = t + step
        
        # Write result
        result[i, 1:] = np.diag(y)
        #print(np.diag(y))
        i += 1

    print(f"{i} steps taken.")

    # if i < total rows, shrink result to required size
    if i < result_shape[0]:
        required_rows = int(i + np.floor(abs(tmax - t) / step))
        result = np.resize(result, (required_rows, result_shape[1]))

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
model = vRK4(f, y0, 0, 300, 0.01, -7)
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
fig.savefig(fname="7")

plt.show()
