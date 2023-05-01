import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re


a = pd.read_csv("../shock_a.txt", delimiter='\s+')
b = pd.read_csv("../shock_b.txt", delimiter='\s+')

# current grid time: 2.500000e-01
# X resolution: 10000
spherical = pd.read_csv("../shock_sphere.txt", delimiter='\t', header=3)

gamma = 1.4

# Shocktube A:

figA = plt.figure()
grid = gs.GridSpec(2, 2)
ax1 = figA.add_subplot(grid[0, 0])
ax2 = figA.add_subplot(grid[0, 1])
ax3 = figA.add_subplot(grid[1, 0])
ax4 = figA.add_subplot(grid[1, 1])

ax1.plot(a['x'], a['d'])
ax2.plot(a['x'], a['v'])
ax3.plot(a['x'], a['p'])
ax4.plot(a['x'], a['e'])

pattern = re.compile(r"A_[\d]*\.[\d]*s_([\w]+)_state\.csv")
for filename in os.listdir():
    if pattern.match(filename):
        sim_type = re.findall(pattern, filename)[0]
        df = pd.read_csv(filename)
        pl = ax1.plot(df['x'], df['d'], ls='-', label=sim_type)[0]
        color = pl.get_color()
        ax2.plot(df['x'], df['v'], c=color, ls='-', label=sim_type)
        ax3.plot(df['x'], df['p'], c=color, ls='-', label=sim_type)
        ax4.plot(df['x'], df['e'], c=color, ls='-', label=sim_type)
        os.remove(filename)
plt.legend()
plt.show()

# Shocktube B:
# Shocktube spherical:

figB = plt.figure()
grid = gs.GridSpec(2, 2)
ax1 = figB.add_subplot(grid[0, 0])
ax2 = figB.add_subplot(grid[0, 1])
ax3 = figB.add_subplot(grid[1, 0])
ax4 = figB.add_subplot(grid[1, 1])

ax1.plot(b['x'], b['d'])
ax2.plot(b['x'], b['v'])
ax3.plot(b['x'], b['p'])
ax4.plot(b['x'], b['e'])

pattern = re.compile(r"B_[\d]*\.[\d]*s_([\w]+)_state\.csv")
for filename in os.listdir():
    if pattern.match(filename):
        sim_type = re.findall(pattern, filename)[0]
        df = pd.read_csv(filename)
        pl = ax1.plot(df['x'], df['d'], ls='-', label=sim_type)[0]
        color = pl.get_color()
        ax2.plot(df['x'], df['v'], c=color, ls='-', label=sim_type)
        ax3.plot(df['x'], df['p'], c=color, ls='-', label=sim_type)
        ax4.plot(df['x'], df['e'], c=color, ls='-', label=sim_type)
        os.remove(filename)
plt.legend()
plt.show()
