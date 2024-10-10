"""
DataArray Class
----------------

Extends the numpy ndarray class to add extra attributes such as names, and
units, and allows us to attach statistical descriptors of the array.
The direct extension to numpy maintains speed and functionality of numpy arrays.

"""
import  numpy as np
from geobipy import DataArray, StatArray

# Integer
test = DataArray(1, name='1')
assert isinstance(test, DataArray) and test.size ==  1 and test.item() == 0.0, TypeError("da 0")
print(test.summary)
test = DataArray(10, name='10')
assert isinstance(test, DataArray) and test.size ==  10 and np.all(test == 0.0), TypeError("da 1")
print(test.summary)
# tuple/Shape
test = DataArray((2, 10), name='(2, 10)')
assert isinstance(test, DataArray) and np.all(test.shape ==  (2, 10)) and np.all(test == 0.0), TypeError("da 2")
print(test.summary)

test = DataArray([2, 10], name='(2, 10)')
assert isinstance(test, DataArray) and np.all(test ==  [2, 10]), TypeError("da 2")
print(test.summary)

# float
test = DataArray(45.454, name='45.454')
assert isinstance(test, DataArray) and test.size ==  1 and test.item() == 45.454, TypeError("da 3")
print(test.summary)
test = DataArray(np.float64(45.454), name='45.454')
assert isinstance(test, DataArray) and test.size ==  1 and test.item() == 45.454, TypeError("da 4")
print(test.summary)

# array
test = DataArray(np.random.randn(1), name="test", units="$\frac{g}{cc}$")
assert isinstance(test, DataArray) and test.size ==  1, TypeError("da 5")
print(test.summary)

test = DataArray(np.arange(10.0), name="test", units="$\frac{g}{cc}$")
assert isinstance(test, DataArray) and test.size ==  10, TypeError("da 6")
print(test.summary)

test = DataArray(test)
assert isinstance(test, DataArray) and test.size ==  10, TypeError("da 6")
print(test.summary)