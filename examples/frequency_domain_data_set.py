"""
Frequency Domain Data Set
-------------------------

"""
import matplotlib.pyplot as plt
from geobipy import FdemData
import numpy as np
from os.path import join

################################################################################
# Let's read in a frequency domain data set

FD1 = FdemData()
dataFname=join('supplementary','Data','Resolve2.txt')
systemFname=join('supplementary','Data','FdemSystem2.stm')
FD1.read(dataFname, systemFname)

################################################################################

FD1.channelNames

################################################################################

FD1.getDataPoint(0)

################################################################################
# Print out a small summary of the data

FD1.summary()

################################################################################
plt.figure()
FD1.scatter2D()

################################################################################
# Plot all the data along the specified line

plt.figure()
ax = FD1.plotLine(30010.0, log=10)

################################################################################
# Or, plot specific channels in the data

plt.figure()
FD1.plot(channels=[0,11,8], log=10, linewidth=0.5);

################################################################################
# Read in a second data set


FD2 = FdemData()
FD2.read(dataFilename=join('supplementary','Data','Resolve1.txt'), systemFilename=join('supplementary','Data','FdemSystem1.stm'))

################################################################################
# We can create maps of the elevations in two separate figures

plt.figure()
FD1.mapPlot(dx=50.0, dy=50.0, mask = 200.0, method='ct');plt.axis('equal')

################################################################################

plt.figure()
FD2.mapPlot(dx=50.0, dy=50.0, mask = 200.0, method = 'ct');plt.axis('equal');

################################################################################
# Or, we can plot both data sets in one figure to see their positions relative
# to each other.
#
# In this case, I use a 2D scatter plot of the data point co-ordinates, and pass
# one of the channels as the colour.

plt.figure()
FD1.scatter2D(s=1.0, c=FD1.getDataChannel(0))
FD2.scatter2D(s=1.0, c=FD2.getDataChannel(0), cmap='jet');

################################################################################
# Or, I can interpolate the values to create a gridded "map". mapChannel will
# interpolate the specified channel number.

plt.figure()
FD1.mapData(3, system=0, method='ct', dx=200, dy=200, mask=250)
plt.axis('equal');

################################################################################
# Export the data to VTK

# FD1.toVTK('FD_one')
# FD2.toVTK('FD_two')

################################################################################
# We can get a specific line from the data set

print(np.unique(FD1.line))

################################################################################
L = FD1.getLine(30010.0)

################################################################################
# A summary will now show the properties of the line.

L.summary()

################################################################################
# And we can scatter2D the points in the line.

plt.figure()
L.scatter2D()

################################################################################

plt.figure()
L.plot(xAxis='r2d', log=10)
