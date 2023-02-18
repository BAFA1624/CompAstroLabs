import numpy as np
import matplotlib.pyplot as plt

def euler(f, y0, x1, xN, N):
    step = (xN - x1) / (N - 1)
    t, y = x1, y0
    result = [(t, y)]
    while t < xN:
        y += step * f(y, t)
        t += step
        result.append((t, y))
    return np.array(result)

def RK2(f, y0, x1, xN, N):
    step = (xN - x1) / (N - 1)
    t, y = x1, y0
    result = [(t, y)]
    while t < xN:
        k1 = step * f(y, t)
        k2 = step * f(y + k1/2, t + step/2)
        y += k2
        t += step
        result.append((t, y))
    return np.array(result)

def RK4(f, y0, x1, xN, N):
    step = (xN - x1) / (N - 1)
    t, y = x1, y0
    result = [(t, y)]
    while t < xN:
        k1 = step * f(y, t)
        k2 = step * f(y + k1/2, t + step/2)
        k3 = step * f(y + k2/2, t + step/2)
        k4 = step * f(y + k3/2, t + step)
        y += (0.166666666) * (k1 + 2*k2 + 2*k3 + k4)
        t += step
        result.append((t, y))
    return np.array(result)


def f(x, t):
    return np.sin(t) - x**3

def is_sq(n):
    if int(np.sqrt(n))**2 == n:
        return True
    else:
        return False

def nearest_sq(n):
    while is_sq(n) is False:
        n += 1
    return n

euler_approx = {}
RK2_approx = {}
RK4_approx = {}

N = [10, 30, 100, 150, 250]
for n in N:
    euler_approx.update({n: euler(f, 0, 0, 10, n)})
    RK2_approx.update({n: RK2(f, 0, 0, 10, n)})
    RK4_approx.update({n: RK4(f, 0, 0, 10, n)})

side_length = int(np.sqrt(nearest_sq(len(N))))
fig = plt.figure(figsize=(5*side_length, 5*(len(N)/side_length)))
axes = [fig.add_subplot(int(np.ceil(len(N)/side_length)), side_length, i) for i in range(1, len(N)+1)]

for ax, step in zip(axes, N):
    euler_a = euler_approx[step]
    RK2_a = RK2_approx[step]
    RK4_a = RK4_approx[step]

    ax.plot(euler_a[:, 0], euler_a[:, 1], "r-", label="Euler")
    ax.plot(RK2_a[:, 0], RK2_a[:, 1], "g--", label="RK2")
    ax.plot(RK4_a[:, 0], RK4_a[:, 1], "c.-.", markersize=0.8, label="RK4")
    ax.set_title(f"RK2: {step}")
    ax.legend()

plt.tight_layout()
plt.show()
