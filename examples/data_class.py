"""
Data Class
----------

The Data class is an extenstion to the [3D Point Cloud](pointCloud3D.ipynb) Class
"""

# Import EM data class, Set up python
from geobipy import StatArray
from geobipy import Data
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
# Set the example file name
filename = join('supplementary','Data','Resolve1.txt')

################################################################################
# We can read data from an ascii file.  The number of headers is given, and the
# columns that you want to read in.
#
# The first three indices are the x, y, z columns.
D = Data()
iCols = [2,3,4,6,7,8,9,10,11]
D.read(filename, columnIndex=iCols, nHeaders=1)

################################################################################
# We can grab one of the channels as an StatArray
ch = D.getDataChannel(channel=0)
ch.summary()

################################################################################
# We can write a summary of the data.
D.summary()

################################################################################
# Plot one or more channels in the data
plt.figure()
x, y = D.plot(channels=[0,3,4], log=10, linewidth=0.5)

################################################################################
# Or we can make maps out of them!
plt.figure()
D.mapData(channel=0)
plt.axis('equal')

################################################################################

# D.toVTK(fileName='Data')
