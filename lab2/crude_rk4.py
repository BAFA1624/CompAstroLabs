import numpy as np
import matplotlib.pyplot as plt
import time

def d_x(t, r):
    x, vx, y, vy = r
    return []
def d_vx(t, r):
    x, vx, y, vy = r
    return []
def d_y(t, r):
    return []
def d_vy(t, r):
    return []

def rk4(f, y0, t0, tN, N):

    step = ( tN - t0 ) / ( N - 1 )
    
    result_shape = ( N, len(y0) + 1 )
    result = np.zeros( result_shape, dtype=np.float64 )



    return result
