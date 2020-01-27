"""
Frequency domain dataset
------------------------
"""
#%%
import matplotlib.pyplot as plt
from geobipy import FdemData
import numpy as np

#%%
# Reading in the Data
# +++++++++++++++++++

################################################################################
dataFolder = "..//supplementary//Data//"
# The data file name
dataFile = dataFolder + 'Resolve2.txt'
# The EM system file name
systemFile = dataFolder + 'FdemSystem2.stm'

################################################################################
# Read in a data set from file.
FD1 = FdemData()
FD1.read(dataFile, systemFile)

################################################################################
# Take a look at the channel names
for name in FD1.channelNames:
    print(name)

################################################################################
# Plot the locations of the data points
plt.figure(figsize=(8,6))
_ = FD1.scatter2D();

################################################################################
# Plot all the data along the specified line
plt.figure(figsize=(8,6))
_ = FD1.plotLine(30010.0, log=10);

################################################################################
# Or, plot specific channels in the data
plt.figure(figsize=(8,6))
_ = FD1.plot(channels=[0,11,8], log=10, linewidth=0.5);

################################################################################
# Read in a second data set
FD2 = FdemData()
FD2.read(dataFilename=dataFolder + 'Resolve1.txt', systemFilename=dataFolder + 'FdemSystem1.stm')

################################################################################
# We can create maps of the elevations in two separate figures
plt.figure(figsize=(8,6))
_ = FD1.mapPlot(dx=50.0, dy=50.0, mask = 200.0)
plt.axis('equal');

################################################################################

plt.figure(figsize=(8,6))
_ = FD2.mapPlot(dx=50.0, dy=50.0, mask = 200.0)
plt.axis('equal');

################################################################################
# Or, we can plot both data sets in one figure to see their positions relative
# to each other.
#
# In this case, I use a 2D scatter plot of the data point co-ordinates, and pass
# one of the channels as the colour.

plt.figure(figsize=(8,6))
_ = FD1.scatter2D(s=1.0, c=FD1.dataChannel(0))
_ = FD2.scatter2D(s=1.0, c=FD2.dataChannel(0), cmap='jet');

################################################################################
# Or, interpolate the values to create a gridded "map". mapChannel will
# interpolate the specified channel number.

plt.figure(figsize=(8,6))
_ = FD1.mapData(channel=3, system=0, dx=200, dy=200, mask=250)
plt.axis('equal');

################################################################################
# Export the data to VTK

# FD1.toVTK('FD_one')
# FD2.toVTK('FD_two')

#%%
# Obtain a line from the data set
# +++++++++++++++++++++++++++++++

################################################################################
# Take a look at the line numbers in the dataset
print(np.unique(FD1.lineNumber))

################################################################################
L = FD1.line(30010.0)

################################################################################
# A summary will now show the properties of the line.

L.summary()

################################################################################
# And we can scatter2D the points in the line.

plt.figure(figsize=(8,6))
_ = L.scatter2D();

################################################################################
# We can specify the axis along which to plot.
# xAxis can be index, x, y, z, r2d, r3d
plt.figure(figsize=(8,6))
_ = FD1.plot(channels=[0,11,8], log=10, linewidth=0.5);


#%%
# Obtain a single datapoint from the data set
# +++++++++++++++++++++++++++++++++++++++++++
#
# Checkout :ref:`Frequency domain datapoint` for an example
# about how to use a datapoint once it is instantiated.
dp = FD1.dataPoint(0)

################################################################################
# File Format for frequency domain data
# +++++++++++++++++++++++++++++++++++++
# Here we describe the file format for frequency domain data.
#
# The data columns are read in according to the column names in the first line
# The header line should contain at least the following column names. 
# Extra columns may exist, but will be ignored.
#
# In this description, the column name or its alternatives are given followed by what the name represents 
# Optional columns are also described 
#
# Required columns
# ________________
# line 
#     Line number for the data point
# fid
#     Fiducial of the data point, these be unique
# x or northing or n 
#     Northing co-ordinate of the data point, (m)
# y or easting or e 
#     Easting co-ordinate of the data point, (m)
# z or alt
#     Altitude of the transmitter coil above ground level (m)
# elevation
#     Elevation of the ground at the data point (m)
# I_<frequency[0]> Q_<frequency[0]> ... I_<frequency[last]> Q_<frequency[last]>  - with the number and square brackets
#     The measurements for each frequency specified in the accompanying system file.
#     I is the real inphase measurement in (ppm)
#     Q is the imaginary quadrature measurement in (ppm)
# Optional columns
# ________________
# InphaseErr[0] QuadratureErr[0] ... InphaseErr[nFrequencies] QuadratureErr[nFrequencies]
#     Estimates of standard deviation for each inphase and quadrature measurement.
#     These must appear after the data colums.
#
# Example Header
# ______________
# Line fid Easting Northing elevation height I_380 Q_380 ... ... I_129550 Q_129550

################################################################################
# File Format for a frequency domain system
# +++++++++++++++++++++++++++++++++++++++++
# .. role:: raw-html(raw)
#    :format: html
#
# The system file is structured using columns with the first line containing header information
#
# Each subsequent row contains the information for each measurement frequency
#
# freq
#     Frequency of the channel
# tor
#     Orientation of the transmitter loop 'x', or 'z'
# tmom
#     Transmitter moment
# tx, ty, tx
#     Offset of the transmitter with respect to the observation locations
# ror
#     Orientation of the receiver loop 'x', or 'z'
# rmom
#     Receiver moment
# rx, ry, rz
#     Offset of the receiver with respect to the transmitter location
#
# Example System File
# ___________________
#
# :raw-html:`freq  tor  tmom  tx   ty   tz   ror rmom  rx   ry   rz  <br />`
# :raw-html:`378   z    1     0.0  0.0  0.0  z   1     7.93 0.0  0.0 <br />`
# :raw-html:`1776  z    1     0.0  0.0  0.0  z   1     7.91 0.0  0.0 <br />`
# :raw-html:`...`
#