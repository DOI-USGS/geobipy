"""
Skytem dataset
--------------
"""
#%%
from geobipy import plotting as cP
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from geobipy import StatArray
from geobipy import TdemData
import h5py

#%%
# Reading in the Data
# +++++++++++++++++++

#%%
dataFolder = "..//..//supplementary//data//"
# The data file name
dataFiles=dataFolder + 'skytem_saline_clay.csv'
# dataFiles = dataFolder + 'Skytem.csv'
# The EM system file name
systemFiles=[dataFolder + 'SkytemHM.stm', dataFolder + 'SkytemLM.stm']

from pathlib import Path
for f in systemFiles[:1]:
    txt = Path(f).read_text()
    print(txt)

#%%
# Read in the data from file
TD = TdemData.read_csv(dataFiles, systemFiles)

#%%
# Plot the locations of the data points
plt.figure(1, figsize=(8,6))
_ = TD.scatter2D()

#%%
# Plot all the data along the specified line
plt.figure(2, figsize=(8,6))
_ = TD.plotLine(0.0, log=10)

#%%
# Or, plot specific channels in the data
plt.figure(3, figsize=(8,6))
_ = TD.plot_data(system=0, channels=[1, 3, 5], log=10)

#%%
plt.figure(4)
plt.subplot(211)
_ = TD.pcolor(system=0, xscale='log', log=10)
plt.subplot(212)
_ = TD.pcolor(system=1, xscale='log', log=10)

#%%
plt.figure(5)
ax = TD.scatter2D(c=TD.secondary_field[:, TD.channel_index(system=0, channel=6)], log=10)
plt.axis('equal')


# with h5py.File('tdem.h5', 'w') as f:
#     TD.createHdf(f, 'tdem')
#     TD.writeHdf(f, 'tdem')

# with h5py.File('tdem.h5', 'r') as f:
#     TD3 = TdemData.fromHdf(f['tdem'])

# with h5py.File('tdem.h5', 'r') as f:
#     tdp = TdemData.fromHdf(f['tdem'], index=0)


# #%%
# # Obtain a line from the data set
# # +++++++++++++++++++++++++++++++
# line = TD.line(0.0)

# #%%
# plt.figure(6)
# _ = line.scatter2D(c=line.secondary_field[:, line.channel_index(system=0, channel=6)], log=10)

# #%%
# plt.figure(7)
# _ = line.plot(xAxis='index', log=10)

# Prepare the dataset so that we can read a point at a time.
Dataset = TdemData._initialize_sequential_reading(dataFiles, systemFiles)
# Get a datapoint from the file.
DataPoint = Dataset._read_record()

plt.show()

#%%
# File Format for time domain data
# ++++++++++++++++++++++++++++++++
# Here we describe the file format for time domain data.
#
# The data columns are read in according to the column names in the first line
#
# In this description, the column name or its alternatives are given followed by what the name represents
# Optional columns are also described.
#
# Required columns
# ________________
# line
#     Line number for the data point
# fid
#     Unique identification number of the data point
# x or northing or n
#     Northing co-ordinate of the data point, (m)
# y or easting or e
#     Easting co-ordinate of the data point, (m)
# z or alt
#     Altitude of the transmitter coil above ground level (m)
# elevation
#     Elevation of the ground at the data point (m)
# txrx_dx
#     Distance in x between transmitter and reciever (m)
# txrx_dy
#     Distance in y between transmitter and reciever (m)
# txrx_dz
#     Distance in z between transmitter and reciever (m)
# Tx_Pitch
#     Pitch of the transmitter loop
# Tx_Roll
#     Roll of the transmitter loop
# Tx_Yaw
#     Yaw of the transmitter loop
# Rx_Pitch
#     Pitch of the receiver loop
# Rx_Roll
#     Roll of the receiver loop
# Rx_Yaw
#     Yaw of the receiver loop
# Off_time[0] Off_time[1] ... Off_time[last]  - with the number and square brackets
#     The measurements for each time gate specified in the accompanying system file under Receiver Window Times
#     The total number of off_time columns should equal the sum of the receiver windows in all system files.
# Optional columns
# ________________
# Off_time_Error[0] Off_time_Error[1] ... Off_time_Error[last]
#     Estimates of standard deviation for each off time measurement
# Example Header
# ______________
# Line fid easting northing elevation height txrx_dx txrx_dy txrx_dz TxPitch TxRoll TxYaw RxPitch RxRoll RxYaw Off[0] Off[1]

#%%
# File Format for a time domain system
# ++++++++++++++++++++++++++++++++++++
# Please see Page 13 of Ross Brodie's `instructions`_
#
# .. _instructions: https://github.com/GeoscienceAustralia/ga-aem/blob/master/docs/GA%20AEM%20Programs%20User%20Manual.pdf
#
# We use GA-AEM for our airborne time domain forward modeller.
#
# Example system files are contained in
# `the supplementary folder`_ in this repository
#
# .. _the supplementary folder: https://github.com/usgs/geobipy/tree/master/documentation_source/source/examples/supplementary/Data
