import numpy as np


a = np.arange(12).reshape(2, -1).tolist()
b = np.arange(12, 24).reshape(2, -1).tolist()

print(b)

c = a + b

a.append(b)

print(c)
print(a)