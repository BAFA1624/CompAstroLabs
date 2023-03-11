import time
import numpy as np
import matplotlib.pyplot as plt
from model import vec_vstep_RK4

# Constants
s_year = 3.1536e7
G = 6.6743e-11 # m^3 kg^-1 s^-2
G_year = G * s_year**2
M = 1.989e30   # kg
GM = G * M
GM_year = G_year * M

# dX/dt
def dX(t, r):
    x, vx, y, vy = r
    return vx
# dV_x/dt
def dV_X(t, r):
    x, vx, y, vy = r
    return -(GM_year * x) / pow(x**2 + y**2, 1.5)
# dY/dt
def dY(t, r):
    x, vx, y, vy = r
    return vy
# dV_y/dt
def dV_Y(t, r):
    x, vx, y, vy = r
    return -(GM_year * y) / pow(x**2 + y**2, 1.5)

f = [dX, dV_X, dY, dV_Y]
# 2.775e10 m/year
y0 = [5.2e12, 0, 0, 2.775e10]

start = time.time()
model = vec_vstep_RK4(f, y0, 0, 100, 0.01, -3)
print(f"Model took: {time.time() - start}s.")
print(f"Model shape: {np.shape(model)}")

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

ax5.plot(x, y, ls='none', marker='x')
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
