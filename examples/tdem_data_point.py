"""
Tdem Data Point Class
---------------------

Tdem Data contains entire data sets

Tdem Data Points can forward model and evaluate themselves

"""

from os.path import join
import numpy as np
import h5py
import matplotlib.pyplot as plt
from geobipy import hdfRead
from geobipy import TdemData
from geobipy import TdemDataPoint
from geobipy import Model1D
from geobipy import StatArray


################################################################################


# The data file name
dataFile=[join('supplementary','Data','Skytem_High.txt'), join('supplementary','Data','Skytem_Low.txt')]
# The EM system file name
systemFile=[join('supplementary','Data','SkytemHM-SLV.stm'), join('supplementary','Data','SkytemLM-SLV.stm')]

################################################################################
# Initialize and read an EM data set

D = TdemData()
D.read(dataFile, systemFile)


################################################################################
# Summarize the Data
#
# Grab a measurement from the data set


P = D.getDataPoint(0)
P._std[:] = 1e-12
P.summary()
plt.figure()
P.plot()

################################################################################
# We can forward model the EM response of a 1D layered earth <a href="../Model/Model1D.ipynb">Model1D</a>

par = StatArray(np.linspace(0.01, 0.1, 19), "Conductivity", "$\\frac{S}{m}$")
thk = StatArray(np.ones(18) * 10.0)
mod = Model1D(nCells = 19, parameters=par, thickness=thk)
plt.figure()
mod.pcolor(grid=True)


################################################################################
# Compute and plot the data from the model

mod = Model1D(depth=np.asarray([125]), parameters=np.asarray([0.00327455, 0.00327455]))
mod.summary()


################################################################################


P.forward(mod)
plt.figure()
P.plot()
P.plotPredicted()


################################################################################


P.summary()


################################################################################


plt.figure()
P.plotDataResidual(xscale='log', log=10)


################################################################################
# The errors are set to zero right now, so lets change that

# Set the Prior
P._predictedData.setPrior('MVNormalLog' ,P._data[P.iActive], P._std[P.iActive]**2.0)
P.updateErrors(relativeErr=[0.05, 0.05], additiveErr=[1.0e-12, 1.0e-13])

################################################################################
# With forward modelling, we can solve for the best fitting halfspace model

HSconductivity=P.FindBestHalfSpace()
print(HSconductivity)
plt.figure()
P.plot(withErrorBars=True)
P.plotPredicted()


################################################################################


plt.figure()
P.plotDataResidual(xscale='log', log=10)


################################################################################
# Compute the misfit between observed and predicted data

print(P.dataMisfit())

################################################################################
# Plot the misfits for a range of half space conductivities

plt.figure()
P.plotHalfSpaceResponses(-6.0,4.0,200)


################################################################################
# Compute the sensitivity matrix for a given model

sensitivityMatrix = P.sensitivity(mod)
J = StatArray(np.abs(sensitivityMatrix),'|Sensitivity|')
plt.figure()
J.pcolor(grid=True, log=10, equalize=True, linewidth=1)


################################################################################


sensitivityMatrix = P.sensitivity(mod)

################################################################################
# We can save the FdemDataPoint to a HDF file

with h5py.File('TdemDataPoint.h5','w') as hf:
    P.createHdf(hf, 'tdp')
    P.writeHdf(hf, 'tdp')



################################################################################
# And then read it in

P1 = hdfRead.readKeyFromFiles('TdemDataPoint.h5','/','tdp', sysPath=join('supplementary','Data'))


################################################################################


P1.summary()
