import matplotlib as mpl
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re


mpl.rcParams['mathtext.fontset'] = "cm"
mpl.rcParams['font.sans-serif'] = "Times New Roman"
mpl.rcParams['font.family'] = "sans-serif"


delete_files = True


a = pd.read_csv("../shock_a.txt", delimiter='\s+')
b = pd.read_csv("../shock_b.txt", delimiter='\s+')

# current grid time: 2.500000e-01
# X resolution: 10000
s = pd.read_csv("../shock_sphere.txt", delimiter=', ')

gamma = 1.4

# Shocktube A:

figA = plt.figure(figsize=(8, 8))
grid = gs.GridSpec(2, 2)
figA.suptitle("Shocktube A")
ax1 = figA.add_subplot(grid[0, 0])
ax2 = figA.add_subplot(grid[0, 1])
ax3 = figA.add_subplot(grid[1, 0])
ax4 = figA.add_subplot(grid[1, 1])

ax1.set_title("Density")
ax2.set_title("Velocity")
ax3.set_title("Pressure")
ax4.set_title("Specific Internal Energy")
ax1.plot(a['x'], a['d'], 'k-')
ax2.plot(a['x'], a['v'], 'k-')
ax3.plot(a['x'], a['p'], 'k-')
ax4.plot(a['x'], a['e'], 'k-')

pattern = re.compile(r"A_[\d]*\.[\d]*s_([\w]+)_state\.csv")
for filename in os.listdir():
    if pattern.match(f"{filename}"):
        sim_type = re.findall(pattern, filename)[0]
        print(f"Plotting: {filename} - {sim_type}")
        df = pd.read_csv(filename)
        pl = ax1.plot(df['x'], df['d'], ls='-',
                      linewidth=1, label=sim_type)[0]
        color = pl.get_color()
        ax2.plot(df['x'], df['v'], c=color, ls='-',
                 linewidth=1, label=sim_type)
        ax3.plot(df['x'], df['p'], c=color, ls='-',
                 linewidth=1, label=sim_type)
        ax4.plot(df['x'], df['e'], c=color, ls='-',
                 linewidth=1, label=sim_type)
        if delete_files:
            os.remove(filename)
ax3.legend(loc='lower left')
plt.tight_layout()
plt.savefig("shocktube_A.png")

# Shocktube B:
# Shocktube spherical:

figB = plt.figure(figsize=(8, 8))
figB.suptitle('Shocktube B')
ax1 = figB.add_subplot(grid[0, 0])
ax2 = figB.add_subplot(grid[0, 1])
ax3 = figB.add_subplot(grid[1, 0])
ax4 = figB.add_subplot(grid[1, 1])

ax1.set_title("Density")
ax2.set_title("Velocity")
ax3.set_title("Pressure")
ax4.set_title("Specific Internal Energy")
ax1.plot(b['x'], b['d'], 'k-')
ax2.plot(b['x'], b['v'], 'k-')
ax3.plot(b['x'], b['p'], 'k-')
ax4.plot(b['x'], b['e'], 'k-')

pattern = re.compile(r"B_[\d]*\.[\d]*s_([\w]+)_state\.csv")
for filename in os.listdir():
    if pattern.match(filename):
        sim_type = re.findall(pattern, filename)[0]
        print(f"Plotting: {filename} - {sim_type}")
        df = pd.read_csv(filename)
        pl = ax1.plot(df['x'], df['d'], ls='-',
                      linewidth=1, label=sim_type)[0]
        color = pl.get_color()
        ax2.plot(df['x'], df['v'], c=color, ls='-',
                 linewidth=1, label=sim_type)
        ax3.plot(df['x'], df['p'], c=color, ls='-',
                 linewidth=1, label=sim_type)
        ax4.plot(df['x'], df['e'], c=color, ls='-',
                 linewidth=1, label=sim_type)
        if delete_files:
            os.remove(filename)
ax1.legend(loc='upper left')
plt.tight_layout()
plt.savefig("shocktube_B.png")

# Spherical coords shocktube


def e(rho, p, gamma):
    return p / (rho * (gamma - 1))


figS = plt.figure(figsize=(8, 8))
figS.suptitle('Spherical Shocktube')
ax1 = figS.add_subplot(grid[0, 0])
ax2 = figS.add_subplot(grid[0, 1])
ax3 = figS.add_subplot(grid[1, 0])
ax4 = figS.add_subplot(grid[1, 1])

ax1.set_title("Density")
ax2.set_title("Velocity")
ax3.set_title("Pressure")
ax4.set_title("Specific Internal Energy")
ax1.plot(s['x'], s['rho'], 'k-')
ax2.plot(s['x'], s['v_x'], 'k-')
ax3.plot(s['x'], s['p'], 'k-')
ax4.plot(s['x'], e(s['rho'], s['p'], gamma), 'k-')

pattern = re.compile(r"S_[\d]*\.[\d]*s_([\w]+)_state\.csv")
for filename in os.listdir():
    if pattern.match(filename):
        sim_type = re.findall(pattern, filename)[0]
        print(f"Plotting: {filename} - {sim_type}")
        df = pd.read_csv(filename)
        pl = ax1.plot(df['x'], df['d'], ls='-',
                      linewidth=1, label=sim_type)[0]
        color = pl.get_color()
        ax2.plot(df['x'], df['v'], c=color, ls='-',
                 linewidth=1, label=sim_type)
        ax3.plot(df['x'], df['p'], c=color, ls='-',
                 linewidth=1, label=sim_type)
        ax4.plot(df['x'], df['e'], c=color, ls='-',
                 linewidth=1, label=sim_type)
        if delete_files:
            os.remove(filename)
ax3.legend(loc='best')
plt.tight_layout()
plt.savefig("shocktube_S.png")

plt.show()
