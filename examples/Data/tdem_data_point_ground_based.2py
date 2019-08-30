"""
Tdem Data Point Class
--------------------

Tdem Data contains entire data sets

Tdem Data Points can forward model and evaluate themselves
"""

################################################################################

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
dataFile=[join('supplementary','Data','Walktem_High.txt'), join('supplementary','Data','Walktem_Low.txt')]
# The EM system file name
systemFile=[join('supplementary','Data','Walktem_HM.stm'), join('supplementary','Data','Walktem_LM.stm')]

################################################################################
# Initialize and read an EM data set

D=TdemData()
D.read(dataFile,systemFile)



################################################################################
# Summarize the Data
# Grab a measurement from the data set

P=D.getDataPoint(0)
P.s[:] = 1e-12
P.z[:] = 0.0
P.summary()
plt.figure()
P.plot()


################################################################################
# We can forward model the EM response of a 1D layered earth <a href="../Model/Model1D.ipynb">Model1D</a>

par = StatArray(np.asarray([10.0, 1.0]), "Conductivity", "$\\frac{S}{m}$")
thk = StatArray(np.ones(1) * 30.0)
mod = Model1D(nCells = 2, parameters=par, thickness=thk)
plt.figure()
mod.pcolor(grid=True)


################################################################################
# Compute and plot the data from the model

s=P.sys[0]
s.windows.centre


################################################################################


mod.depth


################################################################################


P.forward(mod)
plt.figure()
P.plot()
P.plotPredicted()

################################################################################
# The errors are set to zero right now, so lets change that

# Set the Prior
P.p.setPrior('MVNormalLog' ,P.d[P.iActive], P.s[P.iActive]**2.0)
P.updateErrors(relativeErr=[0.5,0.5], additiveErr=[1e-12,1e-13])


################################################################################


P.s[P.iActive]


################################################################################
# With forward modelling, we can solve for the best fitting halfspace model

HSconductivity=P.FindBestHalfSpace()
print(HSconductivity)
plt.figure()
P.plot(withErr=True)
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

sensitivityMatrix = P.sensitivity(mod)
J=StatArray(sensitivityMatrix,'|Sensitivity|')
plt.figure()
np.abs(J).pcolor(grid=True, log=10, equalize=True, linewidth=1)


################################################################################

sensitivityMatrix = P.sensitivity(mod)
sensitivityMatrix


################################################################################
# We can save the FdemDataPoint to a HDF file

#with h5py.File('TdemDataPoint.h5','w') as hf:
#    P.writeHdf(hf, 'tdp')


################################################################################
# And then read it in

#P1=hdfRead.readKeyFromFiles('TdemDataPoint.h5','/','tdp', sysPath=join('supplementary','Data'))


################################################################################


#P1.summary()
