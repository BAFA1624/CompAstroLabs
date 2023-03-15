import numpy as np

a = [[1,2,3],[4,5,6]]

b = [ x[0] for x in a]

c = np.array(a)[:, 0]

print(b)
print(c)
