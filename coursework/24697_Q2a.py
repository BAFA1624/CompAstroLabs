import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import time

class SolarObject:
    def __init__(self, mass, position, velocity):
        self.m = mass
        self.r = position
        self.v = velocity
    
    def euclidean_distance(self, obj):
        return np.sqrt(sum(np.square(np.diff(self.r, obj.r))))

class Orbit:
    def __init__(self, obj):
        self.o = obj
        return

def nbody():
    return
