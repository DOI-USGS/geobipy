"""
Time Domain Datapoint Class
---------------------------
"""

#%%
# There are three ways in which to create a time domain datapoint
#
# 1) :ref:`Instantiating a time domain datapoint`
#
# 2) :ref:`Reading a datapoint from a file`
#
# 3) :ref:`Obtaining a datapoint from a dataset`
#
# Once instantiated, see :ref:`Using a time domain datapoint`
#
# Credits:
# We would like to thank Ross Brodie at Geoscience Australia for his airborne time domain forward modeller
# https://github.com/GeoscienceAustralia/ga-aem
#
# For ground-based time domain data, we are using Dieter Werthmuller's python package Empymod
# https://empymod.github.io/
#
# Thanks to Dieter for his help getting Empymod ready for incorporation into GeoBIPy

#%%
from os.path import join
import numpy as np
import h5py
import matplotlib.pyplot as plt
from geobipy import hdfRead
from geobipy import Waveform
from geobipy import SquareLoop, CircularLoop
from geobipy import butterworth
from geobipy import TdemSystem
from geobipy import TempestData
# from geobipy import TemDataPoint
from geobipy import Model1D
from geobipy import StatArray
from geobipy import Distribution

dataFolder = "..//supplementary//Data//"
# dataFolder = "source//examples//supplementary//Data"

###############################################################################
# Obtaining a datapoint from a dataset
# ++++++++++++++++++++++++++++++++++++
# More often than not, our observed data is stored in a file on disk.
# We can read in a dataset and pull datapoints from it.
#
# For more information about the time domain data set, see :ref:`Time domain dataset`

# The data file name
dataFile = dataFolder + 'Tempest.nc'
# The EM system file name
systemFile = dataFolder + 'Tempest.stm'

################################################################################
# Initialize and read an EM data set
D = TempestData()
D.read(dataFile, systemFile)

################################################################################
# Get a datapoint from the dataset
tdp = D.datapoint(0)

print(tdp.primary_field)
print(tdp.secondary_field)

plt.figure()
tdp.plot()

plt.show()