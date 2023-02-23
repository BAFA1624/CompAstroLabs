import numpy as np
import matplotlib.pyplot as plt

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
    def caller(t, param_arr):
        return np.array([func(t, [*p, *c]) for func, p, c in zip(f, param_arr, coeffs)])
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
            # Two steps of size h
            y = vRK4_step(t, step, params)
            y1 = vRK4_step(t, step, y)

            # One step of size 2h
            y2 = vRK4_step(t, 2 * step, params)

            diff = np.abs(y2 - y1)
            maximum = np.maximum(np.abs(y1), np.abs(y2))
            max_err = 10 ** (np.full_like(maximum, err_scale) + pow_10(maximum))

            # Check error is within tolerance
            err = (0.03333333333333) * diff
            #print(np.greater(err, epsilon_arr))
            err_too_small = np.any(np.less(err, 0.0001 * max_err))
            err_too_large = np.any(np.greater(err, max_err))

            # Adjust step size
            if err_too_large:
                step /= 2
                print(f"step size changed: {step*2} -> {step}.")
            elif err_too_small and ~np.any(np.greater(100 * err, max_err)):
                step *= 2

        # Check result array size & adjust if needed
        # Current row = i
        required_rows = int(i + np.floor(abs(tmax - t) / step))
        
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
        i += 1


    # if i < total rows, shrink result to required size
    if i < result_shape[0]:
        required_rows = int(i + np.floor(abs(tmax - t) / step))
        result = np.resize(result, (required_rows, result_shape[1]))

    return result
