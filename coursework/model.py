import numpy as np

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
# - coeffs: parameter which allows non-variable coefficients to be passed to functions.
def vec_vstep_RK4(f, y0, t0, tmax, init_step=1, err_scale=-6, coeffs=None):
    if err_scale >= 0:
        raise RuntimeError(f"err_scale ({err_scale}) must < 0")
    
    # Convert to numpy arrays
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
        k1 = np.array([ h * func(t, y_c) for func in f ])
        k2 = np.array([ h * func(t + (0.5 * h), y_c + (0.5 * k1)) for func in f ])
        k3 = np.array([ h * func(t + (0.5 * h), y_c + (0.5 * k2)) for func in f ])
        k4 = np.array([ h * func(t + h, y_c + k3) for func in f ])
        result = np.array(y_c + (1/6) * (k1 + 2*k2 + 2*k3 + k4))
        return result
    # Get array of closest powers of 10 for items in the input.
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
    result = np.zeros(result_shape, dtype=np.float64)

    # Set initial conditions
    # Result has the structure: time, parameters for each respective y0 value.
    result[0, 1:] = y0 
    result[0, 0] = t0

    # Initialise arrays for RK4 method
    y = np.zeros_like(y0)

    # Array of max errors for each parameter
    max_err_scale = np.array([10 ** (np.full_like(y0, err_scale) + pow_10(y0)) for func in f])
    epsilon_arr = [y0 * np.full_like(y0, 10 ** err_scale)]

    # The current time step value is results[i - 1, 0]
    # The next t value is set to results[i, 0] when the step size is determined.
    step = init_step
    i, t = 1, t0
    while t < tmax:
        #if i == 2:
        #    exit()
        # Current time step
        t = result[i - 1, 0]

        # Construct initial coefficients
        params = np.array( result[i-1, 1:] )

        # Evaluate if error in step is too large, adjust step accordingly.
        err_too_large = True
        while err_too_large:
            # Two steps of size h
            y = vRK4_step(t, step, params)
            y1 = vRK4_step(t, step, y)

            # One step of size 2h
            y2 = vRK4_step(t, 2 * step, params)

            # diff -> error between two different step calculations
            # max_err -> max allowable error for specified err_scale
            diff = np.abs(y2 - y1)
            maximum = np.maximum(np.abs(y1), np.abs(y2))
            max_err = 10 ** (np.full_like(maximum, err_scale) + pow_10(maximum))

            # Check error is within tolerance
            err = (0.03333333333333) * diff
            err_too_small = np.any(np.less(err, 0.0001 * max_err))
            err_too_large = np.any(np.greater(err, max_err))

            # Adjust step size
            if err_too_large:
                step /= 2
            elif err_too_small and ~np.any(np.greater(100 * err, max_err)):
                step *= 2
        
        # Total rows in result = result_shape[0]
        # If there's insufficient rows in result, expand
        if i == np.shape(result)[0]:
            result = np.resize(result, (int(result_shape[0] * 1.2), result_shape[1]))
            result_shape = np.shape(result)

        # Calculate next time value from step
        result[i, 0] = t + step
        
        # Write result
        result[i, 1:] = y
        i += 1

    # if i < total rows, shrink result to required size
    if i < result_shape[0]:
        required_rows = int(i + np.floor(abs(tmax - t) / step))
        result = np.resize(result, (required_rows, result_shape[1]))
    
    return result

# Shooting Method
def shooting(f, y0, t0, tmax, N, coeffs=None):
    return

# Leap-Frog Method
def leapfrog(f, y0, t0, tmax, N, coeffs=None):
    return

# def vvRK4(f, y0, t0, tmax, init_step=1, err_scale=-6, coeffs=None):
# Verlet Method
def vec_fstep_verlet(f, y0, t0, tmax, N, coeffs=None):
    assert( np.shape(f) == np.shape(y0) )
    assert( np.shape(f) == np.shape(coeffs)[0] )
    assert( np.shape(y0)[0] == np.shape(coeffs)[1] )

    step = ( tmax - t0 ) / ( N - 1 )

    # Initialise result array.
    result_shape = (N, np.shape(y0)[0] + 1)
    result = np.zeros_like(result_shape, dtype=np.float64)
    result[0, :] = [t, *y0]

    for i in range(1, N):
        continue

    return result

# Yoshida Method

