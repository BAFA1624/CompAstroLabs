import numpy as np
import matplotlib.pyplot as plt
import time

def RK4(f, y0, t0, tmax, init_step=1, err_scale=-6):
    if err_scale >= 0:
        raise RuntimeError(f"err_scale ({err_scale}) must < 0")
    
    # Ensure numpy arrays in use
    f = np.array(f)
    y0 = np.array(y0)

    # Check input shapes are compatible
    assert( np.shape(f) == np.shape(y0) )

    # Takes current t, and arrays of Nx2 functions & parameters
    # Returns same shape array containing results of elementwise calling f_i(x_i)
    def caller(t, x_arr):
        return np.reshape( np.array([ func(t, x_arr) for func in f.flatten() ]), x_arr.shape )
    # Process one RK4 step
    def vRK4_step(t, h, y_c):
        k1 = h * caller(t, y_c)
        k2 = h * caller(t + (0.5 * h), y_c + (0.5 * k1))
        k3 = h * caller(t + (0.5 * h), y_c + (0.5 * k2))
        k4 = h * caller(t + h, y_c + k3)
        #k1 = np.array([ h * func(t, y_c) for func in f ])
        #k2 = np.array([ h * func(t + (0.5 * h), y_c + (0.5 * k1)) for func in f ])
        #k3 = np.array([ h * func(t + (0.5 * h), y_c + (0.5 * k2)) for func in f ])
        #k4 = np.array([ h * func(t + h, y_c + k3) for func in f ])
        result = np.array(y_c + (1/6) * (k1 + 2*k2 + 2*k3 + k4))
        assert(y_c.shape == result.shape)
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
    result = [[t0], [y0]]

    # Set initial conditions
    # Result has the structure: time, parameters for each respective y0 value.
    #result[0, 1:] = y0 
    #result[0, 0] = t0

    # Initialise arrays for RK4 method
    y = np.zeros_like(y0)

    # Array of max errors for each parameter
    max_err_scale = np.array([10 ** (np.full_like(y0, err_scale) + pow_10(y0)) for func in f])
    epsilon_arr = [y0 * np.full_like(y0, 10 ** err_scale)]

    # The current time step value is results[i - 1, 0]
    # The next t value is set to results[i, 0] when the step size is determined.
    step = init_step
    i, t = 1, t0
    while t <= tmax:
        #if i == 2:
        #    exit()
        # Current time step
        t, params = result[0][-1], result[1][-1]

        # Construct initial coefficients
        #t = result[i - 1, 0]
        #params = np.array( result[i-1, 1:] )

        # Evaluate if error in step is too large, adjust step accordingly.
        err_too_large = True
        while err_too_large:
            # Two steps of size h
            y = vRK4_step(t, step, params)
            y1 = vRK4_step(t, step, y)

            # One step of size 2h
            y2 = vRK4_step(t, 2 * step, params)
            #print(i, t, step)
            #print(y1, y2)

            # diff -> error between two different step calculations
            # max_err -> max allowable error for specified err_scale
            
            # Check error is within tolerance
            diff = np.abs(y2 - y1)
            maximum = np.maximum(np.abs(y1), np.abs(y2))
            max_err = 10 ** ( np.full_like(y0, err_scale) + pow_10(maximum) )
            err = (0.03333333333333) * diff
            err_too_small = np.any(np.less(err, 0.0001 * max_err))
            err_too_large = np.any(np.greater(err, max_err))

            # Adjust step size
            if err_too_large:
                step /= 2
            # No need to keep the step size excessively small
            elif err_too_small and ~np.any(np.greater(100 * err, max_err)):
                step *= 2
        
        # Write result
        #result[i, 0] = t + step
        #result[i, 1:] = y
        result[0].append(t + step)
        result[1].append(y)
        i += 1

    #if i < result.shape[0]:
    #    required_rows = int(i + np.floor(abs(tmax - t) / step))
    #    result = np.resize(result, (required_rows, *result.shape[1:]))
    
    return np.array(result[0]), np.array(result[1])

def shooting_method(f, boundary_conditions, t0, tmax, init_vals=None, rtol=1e-6, atol=1e-8, init_step=0.001, max_iter=100):
    f = np.array(f)
    boundary_conditions = np.array(boundary_conditions)

    assert( f.shape[0] == boundary_conditions.shape[0] )
    assert( boundary_conditions.shape[1] > 1 )

    # Internal functions
    # Get array of closest powers of 10 for items in the input.
    def pow_10(x):
        if type(x) == np.ndarray:
            result = np.zeros(x.shape)
            not_zero = np.not_equal(np.abs(x), 0)
            result[not_zero] = np.floor(np.log10(np.abs(x[not_zero])))
            return result
        else:
            if x == 0:
                return 0
            else:
                return np.floor(np.log10(np.abs(x)))
    def secant(x0, x1, endpoints, boundary_condition):
        x0, x1 = np.array(x0), np.array(x1)
        #print(endpoints)
        #print("f(tf, x1)", endpoints[:, :, 1][-1])
        #print("f(tf, x0)", endpoints[:, :, 1][-2])
        #print("zf", boundary_condition[:, 1])
        print("x0", x0)
        print("x1", x1)
        #print(boundary_condition.shape)
        x2 = x1 - ( (endpoints[:, :, 1][-1] - boundary_condition[:, 1]) * (x1 - x0) ) / ( endpoints[:, :, 1][-1] - endpoints[:, :, 1][-2] )
        #x2 = x1 - 
        print("x2", x2)
        i = 1
        while ~np.allclose(x1, x2, rtol, atol) and i < max_iter:
            x0, x1 = x1, x2
            x2 = x1 - ( (endpoints[:, :, 1][-1] - boundary_condition[:, 1]) * (x1 - x0) ) / ( endpoints[:, :, 1][-1] - endpoints[:, :, 1][-2] )
            #x2 = x1 - ( (endpoints[1, :] - boundary_condition) * (x1 - x0) ) / ( endpoints[1, :] - endpoints[0, :] )
            print(x2)
            i += 1

        if i == max_iter:
            print(f"WARNING: max_iter ({max_iter}) reached in secant.")

        return x2

    err_scale = pow_10(rtol)

    if init_vals is None:
        # Need to guess IVs for coupled equations
        # Get I.V guess from rate of change over entire interval
        init_vals = np.stack(
                (boundary_conditions[:, 0], (boundary_conditions[:, 1] - boundary_conditions[:, 0]) / (tmax - t0)),
                axis=1
                )
    else:
        init_vals = np.array(init_vals)
    prev_init_vals = np.zeros_like(init_vals)
    print(init_vals.shape, init_vals[:, 1].shape)

    # Apply RK4 to guess endpoint for I.V
    t_prev, prev_model = RK4(f, prev_init_vals, t0, tmax, init_step=init_step, err_scale=err_scale)
    t, model = RK4(f, init_vals, t0, tmax, init_step=init_step, err_scale=err_scale)

    plt.plot(t_prev, [z for z in prev_model[:, 0, 0]], label='prev')
    plt.plot(t, [z for z in model[:, 0, 0]], label='current')
    plt.legend()
    #plt.show()

    print("prev_model[-1, :]")
    print(prev_model[-1, :], prev_model[-1, :].shape)
    print("model[-1, :]")
    print(model[-1, :], model[-1, :].shape)
    endpoints = np.array([ prev_model[-1, :], model[-1, :] ])

    # Check if boundary conditions are satisfied
    # If any endpoints aren't close enough to the boundary condition
    BC_satisfied = np.allclose(endpoints[1, :], boundary_conditions[:, -1], rtol, atol)

    # Loop while endpoint guess != boundary_condition
    i = 0
    while BC_satisfied is False and i < max_iter:
        # Secant for root guess -> new I.V
        # Using array of zeros as "x0" in secant method, init_vals is "x1" 
        init_vals = secant(prev_init_vals[:, 1], init_vals[:, 1], endpoints, boundary_conditions)

        # Apply RK4 to guess endpoint for I.V
        t, model = RK4(f, init_vals, t0, tmax, init_step=init_step, err_scale=err_scale)
        endpoints = np.array([ prev_model[-1, 1:], model[-1, 1:] ])

        plt.plot(t_prev, [z for z in prev_model[:, 0, 0]], label='prev')
        plt.plot(t, [z for z in model[:, 0, 0]], label='current')
        plt.legend()
        #plt.show()
    
        BC_satisfied = np.allclose(endpoints[1, :], boundary_conditions[:, -1], rtol, atol)

        prev_init_vals = init_vals
        prev_model = model

        i += 1
    
    if i == max_iter:
        print("WARNING: max_iter ({max_iter}) reached.")

    return init_vals, t, model

# Constants
g = 9.81

def dz(t, r):
    z, v = r.flatten()
    return v
def dv(t, r):
    z, v = r.flatten()
    return -g

f = np.array([ [dz, dv] ])

IVs, t, model = shooting_method(f, [[0, 0]], 0, 10, init_vals=[[0, 40]])
