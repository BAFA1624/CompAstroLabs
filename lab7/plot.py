import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

data = {}
pattern = re.compile(r"[\d]*\.[\d]*s_diffusive_state\.csv")
for filename in os.listdir():
    if re.match(pattern, filename):
        data[filename] = pd.read_csv(filename)

plt.xlim(0, 400)
for filename, df in data.items():
    plt.plot(df['x'], df['q'])
plt.show()

data = {}
pattern = re.compile(r"[\d]*\.[\d]*s_csdd_state\.csv")
for filename in os.listdir():
    if re.match(pattern, filename):
        data[filename] = pd.read_csv(filename)

plt.xlim(0, 400)
for filename, df in data.items():
    plt.plot(df['x'], df['q'])
plt.show()

data = {}
pattern = re.compile(r"[\d]*\.[\d]*s_lax_wendroff_state\.csv")
for filename in os.listdir():
    if re.match(pattern, filename):
        data[filename] = pd.read_csv(filename)

plt.xlim(0, 400)
for filename, df in data.items():
    plt.plot(df['x'], df['q'])
plt.show()
