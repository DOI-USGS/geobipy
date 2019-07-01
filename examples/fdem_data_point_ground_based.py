"""
Fdem Data Point Class
---------------------

Fdem Data contains entire data sets.
Fdem Data Points can forward model and evaluate themselves
"""
from os.path import join
import numpy as np
import h5py
import matplotlib.pyplot as plt
from geobipy import hdfRead
from geobipy import FdemData
from geobipy import FdemDataPoint
from geobipy import Model1D
from geobipy import StatArray

################################################################################

# The data file name
dataFile=join('supplementary','Data','Resolve2.txt')
# The EM system file name
systemFile=join('supplementary','Data','FdemSystem2.stm')

################################################################################
# Initialize and read an EM data set

D=FdemData()
D.read(dataFile,systemFile)

################################################################################
# Summarize the Data

D.summary()

################################################################################
# Grab a measurement from the data set

P=D.getDataPoint(0)
P.e -= P.z
P.z[0] = 0.0
P.sys.summary()
P.summary()
plt.figure()
P.plot()

################################################################################
# We can forward model the EM response of a 1D layered earth <a href="../Model/Model1D.ipynb">Model1D</a>

nCells = 19
par = StatArray(np.linspace(0.01, 0.1, nCells), "Conductivity", "$\\frac{S}{m}$")
thk = StatArray(np.ones(nCells) * 10.0)
mod = Model1D(nCells = nCells, parameters=par, thickness=thk)
mod.summary()
#plt.figure()
#mod.plotBlocks()

################################################################################
# Compute and plot the data from the model

P.forward(mod)
plt.figure()
P.plot()
P.plotPredicted()

################################################################################
# The errors are set to zero right now, so lets change that

# Set the Prior
addErrors = StatArray(np.full(2*P.sys.nFreq, 10.0))
P.p.setPrior('MVNormalLog', addErrors, addErrors)
P.updateErrors(1, P.s[:], 0.1, addErrors[:])

################################################################################
# With forward modelling, we can solve for the best fitting halfspace model

HSconductivity=P.FindBestHalfSpace()
print('Best half space conductivity is ', HSconductivity, ' $S/m$')
plt.figure()
P.plot()
P.plotPredicted()


################################################################################
# Compute the misfit between observed and predicted data

print(P.dataMisfit())

################################################################################
# Plot the misfits for a range of half space conductivities

plt.figure()
P.plotHalfSpaceResponses(-6.0,4.0,200)

################################################################################
# Compute the sensitivity matrix for a given model

J = P.sensitivity(mod)
plt.figure()
J.pcolor(equalize=True);

################################################################################
# We can save the FdemDataPoint to a HDF file

with h5py.File('FdemDataPoint.h5','w') as hf:
    P.writeHdf(hf, 'fdp')

################################################################################
# And then read it in

P1 = hdfRead.readKeyFromFiles('FdemDataPoint.h5','/','fdp')
P1.summary()
