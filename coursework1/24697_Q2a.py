import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import time

# Constants
s_year = 3.1536e7      # Seconds in a year
G = 6.6743e-11         # Gravitational constant
G_year = G * s_year**2 # Gravitational constant using years for time unit
M = 1.989e30           # Mass of Sun in kg
GM_year = G_year * M   # Combine G_year & M into single constant
AU = 1.495979e11       # Astronomical unit in metres

def normalize(arr):
    assert(type(arr) == np.ndarray)
    norm_arr = np.zeros_like(arr)
    nonzero = np.nonzero(arr)
    norm_arr[nonzero] = arr[nonzero] / max(np.abs(arr[nonzero]))
    return norm_arr

def magnitude(r):
    return np.sqrt(np.sum(np.square(r), axis=None))

class SolarObject:
    def __init__(self, object_name, mass, position, velocity):
        assert(type(position) == list or type(position) == np.ndarray)
        assert(type(velocity) == list or type(velocity) == np.ndarray)
        self.name = object_name     # Name to distinguish between objects in data
        self.m = mass               # Mass of object
        self.r = np.array(position) # [x, y] vector. Relative to orbit CoM
        self.v = np.array(velocity) # [v_x, v_y] vector.
        return None
    def __repr__(self):
        return f"SolarObject: {mass: {self.m}, position: {self.r}, velocity: {self.v}\}"
    def __str__(self):
        return f"SolarObject: (mass: {self.m}, position: {self.r}, velocity: {self.v})"
    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name
    def __ne__(self, other):
        return not self.__eq__(other)

    # Euclidean distance between this SolarObject & a given position
    def distance(self, position):
        return np.sqrt(np.sum(np.square(self.r - position), axis=None))
    
    # Vector between this object & another.
    def rel_position(self, obj):
        return obj.r - self.r
    
    # Full state of object
    def full_state(self):
        return (self.name, self.m, self.r, self.v)
    
    # Position and velocity vectors of this object
    def current_state(self):
        return np.array([self.r, self.v])
    # object_states = [ [[pos], [vel]], ...]
    
    # Calculate the ith dimension of the next state of this SolarObject from the current states
    # of all objects in the system.
    def next_state(self, i, object_masses, object_states):
        r, v = object_states[i]
        # Change in position over a time step ~= velocity
        dr_next = v
        # Change in velocity over a time step ~= sum of accelerations on object
        dv_next = -np.sum(
        [
            (G * mass * (r - pos)) / self.distance(pos)**3
            for j, (mass, (pos, vel)) in enumerate(zip(object_masses, object_states))
            if i != j
        ], axis=0)
        return (dr_next, dv_next)
    
    # Update the current state of the object
    def update_state(self, mass=None, state=None):
        position = None
        velocity = None
        if state is not None:
            position, velocity = state
        if mass is not None:
            self.m = mass
        if position is not None:
            self.r = position
        if velocity is not None:
            self.v = velocity
        return None
    
    # Clear the object
    def clear_state(self):
        self.m = None
        self.r = None
        self.v = None

Planet = SolarObject
   
# Same as SolarObject, however position and velocity are fixed in place.
class FixedObject(SolarObject):
    def __init__(self, object_name, mass, position, velocity):
        super().__init__(object_name, mass, position, velocity)
        return None
    # Fixed version of next_state
    def next_state(self, i, object_masses, object_states):
        return np.array([self.r, self.v])
    # Fixed version of update_state
    def update_state(self, mass=None, state=None):
        return None
    # Force change function to manipulate state outside of calculations
    def force_change(self, mass, position, velocity):
        self.m = mass
        self.r = position
        self.v = velocity
        return None

Star = FixedObject

class Orbit:
    def __init__(self, obj, history = False):
        assert(type(obj) == SolarObject or type(obj) == FixedObject)
        self.o = obj         # Object in this orbit
        # Boolean to check if previous positions & velocities should be stored
        self.store_history = history
        self.times = [0]             # All previous times, if store_history is True, else current only
        self.positions =  [self.o.r] # All previous positions, if store_history is True, else current only
        self.velocities = [self.o.v] # All previous velocities, if store_history is True, else current only
        return None
    
    # Retrieve individual parts of underling object state
    def name(self):
        return self.o.name
    def mass(self):
        return self.o.m
    def position(self):
        return self.o.r
    def velocity(self):
        return self.o.v
    
    # Retrieve entire history of orbit
    def history(self):
        return (np.array(self.times), np.array(self.positions), np.array(self.velocities))
    
    # Retrieve full state of SolarObject owned by this orbit
    def full_state(self):
        return self.o.full_state()
    
    # Retrieve position and velocity data of SolarObject owned by this orbit
    def current_state(self):
        return self.o.current_state()
    
    # Calculate next state of SolarObject owned by this orbit
    def next_state(self, i, object_masses, object_states):
        return self.o.next_state(i, object_masses, object_states)
    
    # Set current state of this orbit, including underlying SolarObject
    def set_state(self, time=None, mass=None, state=None):
        position = None
        velocity = None
        if state is not None:
            position, velocity = state
            self.positions[-1] = position
            self.velocities[-1] = velocity
        if time is not None:
            self.times[-1] = time
        if mass is not None and state is not None:
            self.o.update_state(mass, state)
        elif mass is not None:
            self.o.update_state(mass=mass)
        elif state is not None:
            self.o.update_state(state=state)
        return None
        
    # Specify the next state of this object, inluding underlying SolarObject
    def update_state(self, time=None, mass=None, state=None):
        position = None
        velocity = None
        if state is not None:
            position, velocity = state
        if self.store_history:
            if time is not None:
                self.times.append(time)
            if position is not None:
                self.positions.append(position)
            if velocity is not None:
                self.velocities.append(velocity)
        else:
            if time is not None:
                self.times[0] = time
            if position is not None:
                self.positions[0] = position
            if velocity is not None:
                self.velocities[0] = velocity
        self.o.update_state(mass=mass, state=state)
        return None
    
    # Clear this orbit, and underlying SolarObject
    def clear_state(self):
        self.o.clear_state()
        self.o = None
        self.history = None
        self.times = None
        self.positions = None
        self.velocities = None

class System:
    def __init__(self):
        self.orbits = []       # Orbits in this system
        self.time = 0          # Current time
        self.time_step = 0.001 # Current time step
        return None
    
    # Add a new object into a system
    def add(self, orbit):
        assert(type(orbit) == Orbit)
        if len(self.orbits) == 0:
            self.orbits=[orbit]
            self.time = 0
            self.time_step = 0.001
            return None
        self.orbits.append(orbit)
        return None
    
    # Get / set the current time step
    def step(self, step=None):
        if step is not None:
            self.time_step = step
        return self.time_step
    
    # Perform single RK4 step on all planets
    def RK4_step(self, h, object_masses, current_states):
        k1 = h * np.array([
            orbit.next_state(i, object_masses, current_states)
            for i, orbit in enumerate(self.orbits)
        ])
        k2 = h * np.array([
            orbit.next_state(i, object_masses, current_states + k1 / 2)
            for i, orbit in enumerate(self.orbits)
        ])
        k3 = h * np.array([
            orbit.next_state(i, object_masses, current_states + k2 / 2)
            for i, orbit in enumerate(self.orbits)
        ])
        k4 = h * np.array([
            orbit.next_state(i, object_masses, current_states + k3)
            for i, orbit in enumerate(self.orbits)
        ])
        return np.array([
            state + (1/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)
            for state, k_1, k_2, k_3, k_4 in zip(current_states, k1, k2, k3, k4)
        ])
    
    # Convert provided object_states into the centre-of-mass frame.
    def com_converter(self, object_masses, object_states):
        m_total_reciprocal = 1 / np.sum(object_masses, axis=None)
        
        # Centre of mass (com) formula is same for position and velocity
        # Therefore apply to all parts of object_states
        com = np.sum(
            np.array([
                m_i * r_i * m_total_reciprocal
                for m_i, r_i in zip(object_masses, object_states)
            ]),
            axis=0
        )
        
        return object_states - com
    
    # Project system forwards by 'time' amount.
    def simulate(self, time, rtol=1e-05, atol=1e-08, err_scale=-6, com=True):
        initial_time = self.time
        final_time = self.time + time
        object_masses = np.array([orbit.mass() for orbit in self.orbits])
        
        if com: # Convert initial position into centre-of-mass
            current_states = np.array([orbit.current_state() for orbit in self.orbits])
            com_states = self.com_converter(object_masses, current_states)
            for orbit, state in zip(self.orbits, com_states):
                orbit.set_state(mass=orbit.mass(), state=state)
        
        # Variational step size, 4th order Runge-Kutta method
        while self.time <= final_time:
            current_states = np.array([orbit.current_state() for orbit in self.orbits])
            
            err_too_large = True
            while err_too_large:
                step_h_1 = self.RK4_step(self.time_step, object_masses, current_states)
                step_h_2 = self.RK4_step(self.time_step, object_masses, step_h_1)
                
                step_2h = self.RK4_step(2 * self.time_step, object_masses, current_states)

                max_err = rtol * np.maximum(np.abs(step_h_2), np.abs(step_2h)) + atol
                step_err = (1/30) * np.abs(step_2h - step_h_2)
                
                err_too_large = np.any(np.greater(step_err, max_err))
                err_too_small = np.any(np.less(1000 * step_err, max_err))
                
                if err_too_large:
                    self.time_step /= 2
                elif err_too_small:
                    self.time_step *= 2
            if com:
                step_h_2 = self.com_converter(object_masses, step_h_2)
                
            # Update orbit states
            for orbit, state in zip(self.orbits, step_h_2):
                orbit.update_state(time=self.time, state=state)
            self.time += self.time_step

        return None
    
    # Retrieve current states of all Orbits in the System
    def current_state(self):
        return np.array([orbit.current_state() for orbit in self.orbits])
    
    # Retrieve full states of all Orbits in the System
    def full_state(self):
        return np.array([ orbit.full_state() for orbit in self.orbits])
    
    # Retrieve full history of all Orbits in the System
    def history(self, i=None):
        if i is None:
            return np.array([ orbit.history() for orbit in self.orbits ])
        return self.orbits[i].history()
    
    # Plot Orbits in the system
    def plot_history(self, figsize=(8,8), title="",
                     marker='o', marker_sizes=None,
                     ls='--', linewidth=1.0,
                     scale=None, norm_scale=False,
                     units=None, fname=None):
        if type(marker_sizes) == list:
            assert(len(marker_sizes) == len(self.orbits))
        elif type(marker_sizes) == np.ndarray:
            assert(len(marker_sizes.shape) == 1 and
                   marker_sizes.shape[0] == len(self.orbits))
        elif marker_sizes is not None:
            marker_sizes = [marker_sizes] * len(self.orbits)
        
        fig = plt.figure(figsize=figsize)
        
        scale_x = 1
        scale_y = 1
        units_x = ""
        units_y = ""
            
        if norm_scale:
            nonzero_x = np.nonzero(x)
            nonzero_y = np.nonzero(y)
            x[nonzero_x] = x[nonzero_x] / max(x[nonzero_x])
            y[nonzero_y] = y[nonzero_y] / max(y[nonzero_y])
            scale_x = max(x[nonzero_x])
            scale_y = max(y[nonzero_y])
        
        for i, orbit in enumerate(self.orbits):
            t, p, v = orbit.history()
            x, y = p[:, 0], p[:, 1]
            vx, vy = v[:, 0], v[:, 1]
        
            if units is not None and scale is not None:
                assert(type(units) == np.ndarray or type(units) == list)
                if type(units) == list:
                    assert(len(units) == 2)
                if type(units) == np.ndarray:
                    assert(len(units.shape) == 1 and units.shape[-1] == 2)
                assert(type(units[0]) == str)
                units_x = units[0]
                units_y = units[1]
            
                if type(scale) == np.ndarray:
                    assert(len(scale.shape) == 1 and len(scale.shape[0]) == 2)
                if type(scale) == list or type(scale) == np.ndarray:
                    x = x / scale[0]
                    y = y / scale[1]
                    scale_x = scale[0]
                    scale_y = scale[1]
            
                units_x = f"/ {scale_x:.2e}{units[0]}"
                units_y = f"/ {scale_y:.2e}{units[1]}"
            else:
                units_x = ""
                units_y = ""
            
            p = plt.plot(x[-1], y[-1], marker=marker, markersize=float(marker_sizes[i]), ls='none', label=orbit.name())
            plt.plot(x, y, c=p[0].get_color(), ls=ls, linewidth=linewidth)

            plt.xlabel(f"x {units_x}")
            plt.ylabel(f"y {units_y}")
        
        if fname is not None:
            plt.savefig(fname)
        
        plt.title(title)
        plt.axis("equal")
        plt.legend()
        plt.show()
        
        return None
    
    def clear_state(self):
        for o in self.orbits:
            o.clear_state()
        self.orbits = []
        self.time = None
        self.time_step = None
        return None

# Constants for 2.a
m1 = 1 * (10 ** -3) * M
m2 = 4 * (10 ** -2) * M

# Initial radius
a1 = 2.52 * AU
a2 = 5.24 * AU

# Assuming they begin at [a1, 0], [a2, 0] at t=0
r1 = np.array([a1, 0])
r2 = np.array([a2, 0])
v1_init = np.sqrt(G * M / a1)
v2_init = np.sqrt(G * M / a2)

v1 = np.array([0, -v1_init])
v2 = np.array([0, -v2_init])

# Setting up system for 2.a
Sun = Planet("Sun", M, [0, 0], [0, 0])
p1 = Planet("P1", m1, r1, v1)
p2 = Planet("P2", m2, r2, v2)

# Setting up the solar system
solar_system1 = System()
solar_system1.add(Orbit(Sun, True))
solar_system1.add(Orbit(p1, True))
solar_system1.add(Orbit(p2, True))

years = 50
time_seconds = years * s_year

start = time.time()
print("Simulating p1, p2 with non-fixed Sun position...")
solar_system1.simulate(time_seconds, rtol=1e-8, atol=1e-10, com=False)
print(f"Done. ({time.time() - start:.3f}s)")

solar_system1.plot_history(title="P1, P2, non-fixed Sun", marker_sizes=[12, 6, 6],
                           units=['m','m'], scale=[AU, AU])

solar_system1.clear_state()

Sun_fixed = FixedObject("Sun", M, [0, 0], [0, 0])
p1 = Planet("P1", m1, r1, v1)
p2 = Planet("P2", m2, r2, v2)

solar_system1.add(Orbit(Sun_fixed, True))
solar_system1.add(Orbit(p1, True))
solar_system1.add(Orbit(p2, True))

print("Simulating p1, p2 with fixed Sun position...")
start = time.time()
solar_system1.simulate(time_seconds, rtol=1e-8, atol=1e-10, com=False)
print(f"Done. ({time.time() - start:.3f}s)")

solar_system1.plot_history(title="P1, P2, fixed Sun", marker_sizes=[12, 6, 6],
                           units=['m','m'], scale=[AU, AU])

solar_system1.clear_state()

def r_perihelion(a, e):
    return (1 - e) * a
def r_aphelion(a, e):
    return (1 + e) * np.sqrt(a * (1 - e))

j_eccentricity = 0.049
s_eccentricity = 0.057

j_m = 1.898e27
s_m = 5.683e26

j_r = np.array([r_perihelion(5.204 * AU, j_eccentricity), 0])
s_r = np.array([r_perihelion(9.583 * AU, s_eccentricity), 0])

j_v = np.array([0, -np.sqrt(G * M / r_perihelion(5.204 * AU, j_eccentricity))])
s_v = np.array([0, -np.sqrt(G * M / r_perihelion(9.583 * AU, s_eccentricity))])

Sun = SolarObject("Sun", M, [0, 0], [0, 0])
jupiter = Planet("Jupiter", j_m, [5.204 * AU, 0], j_v)
saturn = Planet("Saturn", s_m, [9.583 * AU, 0], s_v)

solar_system2 = System()
solar_system2.add(Orbit(Sun, True))
solar_system2.add(Orbit(jupiter, True))
solar_system2.add(Orbit(saturn, True))

print("Simulating Sun, Jupiter, and Saturn system...")
start = time.time()
solar_system2.simulate(time_seconds, rtol=1e-10, atol=1e-10, com=False)
print(f"Done. {time.time() - start:.3f}")

solar_system2.plot_history(title="Jupiter, Saturn, and non-fixed Sun", marker_sizes=[12, 6, 6],
                           units=['m','m'], scale=[AU, AU])

solar_system2.clear_state()
