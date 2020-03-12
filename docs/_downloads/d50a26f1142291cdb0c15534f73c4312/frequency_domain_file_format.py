"""
File Format for frequency domain data
+++++++++++++++++++++++++++++++++++++
"""

# Here we describe the file format for frequency domain data.
#
# The data columns are read in according to the column names in the first line.
#
# In this description, the column name or its alternatives are given followed by what the name represents.
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
# Line fid easting northing elevation height I_380 Q_380 ... ... I_129550 Q_129550

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
# Example system files are contained in
# `the supplementary folder`_ in this repository
#
# .. _the supplementary folder: https://github.com/usgs/geobipy/tree/master/documentation_source/source/examples/supplementary/Data
#
# See the Resolve.stm files.