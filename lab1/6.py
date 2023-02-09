import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_csv(path, delim=','):
    data = {}
    with open(path, "r") as file:
        lines = list(file)
        col_titles = lines[0].split(delim)
        for title in col_titles:
            data[title] = []
        for l in lines[1:]:
            vals = l.split(delim)
            for i, v in enumerate(vals):
                data[col_titles[i]].append(np.float64(v))

        for title in col_titles:
            data[title] = np.array(data[title])
    return data

data = read_csv("/Users/ben/Documents/gitrepos/CompAstroLabs/lab1/spectrum.csv")
keys = list(data.keys())

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

mask = np.logical_or(np.less(data[keys[0]], 1427.5), np.greater(data[keys[0]], 1428.5))

q_popt, q_pcov = curve_fit(quadratic, data[keys[0]][mask], data[keys[1]][mask])
c_popt, c_pcov = curve_fit(cubic, data[keys[0]][mask], data[keys[1]][mask])

q_baseline = data[keys[1]] - quadratic(data[keys[0]], *q_popt)
c_baseline = data[keys[1]] - cubic(data[keys[0]], *c_popt)

fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(data['RF  (MHz)'], data['P_IF  (mW)\n'],
         marker='x', c='k', ls='none', markersize=4)
ax1.plot(data[keys[0]], quadratic(data[keys[0]], *q_popt), ls='--', label='quadratic')
#plt.plot(data[keys[0]], cubic(data[keys[1]], *c_popt), ls='--', label='cubic')
ax1.legend()

ax2.plot(data[keys[0]], q_baseline, marker='x', c='k', ls='none', markersize=4)

plt.show()
