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

approximations = {}
for i in [10, 30, 100, 150, 250]:
    approximations.update({i: euler(f, 0, 0, 10, i)})

side_length = int(np.sqrt(nearest_sq(len(approximations))))
fig = plt.figure(figsize=(5*side_length, 5*(len(approximations)/side_length)))
axes = [fig.add_subplot(int(np.ceil(len(approximations)/side_length)), side_length, i) for i in range(1, len(approximations)+1)]

for ax, (step, approx) in zip(axes, list(approximations.items())):
    ax.plot(approx[:, 0], approx[:, 1])
    ax.set_title(step)

plt.tight_layout()
plt.show()
