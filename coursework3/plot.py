import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re


def d(df):
    return df['q1'].to_numpy()


def v(df):
    return df['q2'].to_numpy() / df['q1'].to_numpy()


def epsilon(df):
    q1 = df['q1'].to_numpy()
    q2 = df['q2'].to_numpy()
    q3 = df['q3'].to_numpy()
    return (q3 / q1) - ((q2**2) / (2 * q1**2))


def p(df):
    return 0.4 * df['q1'].to_numpy() * epsilon(df)


a = pd.read_csv("../shock_a.txt", delimiter='\s+')
b = pd.read_csv("../shock_b.txt", delimiter='\s+')

# current grid time: 2.500000e-01
# X resolution: 10000
spherical = pd.read_csv("../shock_sphere.txt", delimiter='\t', header=3)

# Shocktube A:

fig = plt.figure()
grid = gs.GridSpec(2, 2)
ax1 = fig.add_subplot(grid[0, 0])
ax2 = fig.add_subplot(grid[0, 1])
ax3 = fig.add_subplot(grid[1, 0])
ax4 = fig.add_subplot(grid[1, 1])

ax1.plot(a['x'], a['d'])
ax2.plot(a['x'], a['v'])
ax3.plot(a['x'], a['p'])
ax4.plot(a['x'], a['e'])

pattern = re.compile(r"A_[\d]*\.[\d]*s_([\w]+)_state\.csv")
for filename in os.listdir():
    if pattern.match(filename):
        sim_type = re.findall(pattern, filename)[0]
        df = pd.read_csv(filename)
        ax1.plot(df['x'], d(df), 'r--', label=sim_type)
        ax2.plot(df['x'], v(df), 'r--', label=sim_type)
        ax3.plot(df['x'], p(df), 'r--', label=sim_type)
        ax4.plot(df['x'], epsilon(df), 'r--', label=sim_type)

plt.show()

# Shocktube B:
# Shocktube spherical:
