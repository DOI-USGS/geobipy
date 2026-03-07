from copy import deepcopy
from geobipy import DataArray
import numpy as np

x = np.arange(10)
y = DataArray(x)
z = deepcopy(y)

y[5] = 99
z[6] = 100
print(f"{x=}")
print(f"{y=}")

print(f"{z=}")