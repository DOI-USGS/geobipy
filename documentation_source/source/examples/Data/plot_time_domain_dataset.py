"""
Time Domain Data Set
--------------------
"""

#%%
from geobipy import customPlots as cP
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from geobipy.src.classes.core.StatArray import StatArray
from geobipy.src.classes.data.dataset.TdemData import TdemData

#%%
# Reading in the Data
# +++++++++++++++++++

################################################################################
dataFolder = "..//supplementary//Data//"
# The data file name
dataFiles=[dataFolder + 'Skytem_High.txt', dataFolder + 'Skytem_Low.txt']
# The EM system file name
systemFiles=[dataFolder + 'SkytemHM-SLV.stm', dataFolder + 'SkytemLM-SLV.stm']

################################################################################
# Read in the data from file
TD = TdemData()
TD.read(dataFiles, systemFiles)


################################################################################
# Plot the locations of the data points
plt.figure(figsize=(8,6))
_ = TD.scatter2D()


################################################################################
# Plot all the data along the specified line
plt.figure(figsize=(8,6))
_ = TD.plotLine(100101.0, log=10)

################################################################################
# Or, plot specific channels in the data
plt.figure(figsize=(8,6))
_ = TD.plot(system=0, channels=TD.active[:3], log=10)

################################################################################
plt.figure()
plt.subplot(211)
_ = TD.pcolor(system=0, log=10, xscale='log')
plt.subplot(212)
_ = TD.pcolor(system=1, log=10, xscale='log')

################################################################################
plt.figure()
ax = TD.scatter2D(s=1.0, c=TD.dataChannel(system=0, channel=23), equalize=True)
plt.axis('equal')

################################################################################
# TD.toVTK('TD1', format='binary')


#%%
# Obtain a line from the data set
# +++++++++++++++++++++++++++++++
line = TD.line(100601.0)

################################################################################
plt.figure()
_ = line.scatter2D(c = line.dataChannel(10, system=1))

################################################################################
plt.figure()
_ = line.plot(xAxis='x', log=10)

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
# TxPitch 
#     Pitch of the transmitter loop
# TxRoll 
#     Roll of the transmitter loop
# TxYaw 
#     Yaw of the transmitter loop
# RxPitch 
#     Pitch of the receiver loop
# RxRoll 
#     Roll of the receiver loop
# RxYaw 
#     Yaw of the receiver loop
# Off[0] Off[1] ... Off[last]  - with the number and square brackets
#     The measurements for each time gate specified in the accompanying system file under Receiver Window Times 
# Optional columns
# ________________
# OffErr[0] OffErr[1] ... OffErr[last]
#     Estimates of standard deviation for each off time measurement
# Example Header
# ______________
# Line fid easting northing elevation height txrx_dx txrx_dy txrx_dz TxPitch TxRoll TxYaw RxPitch RxRoll RxYaw Off[0] Off[1]

################################################################################
# File Format for a time domain system
# +++++++++++++++++++++++++++++++++++++++++
# .. role:: raw-html(raw)
#    :format: html
