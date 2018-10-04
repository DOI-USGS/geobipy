""" @MTDataPoint_Class
Module describing a Magneto-Telluric Data Point that contains a single measurement.
"""
from copy import copy, deepcopy
from ....classes.core.StatArray import StatArray
#from ...forwardmodelling.EMfor1D_F import fdem1dfwd
from ...forwardmodelling.MTfor1D import mt1dfwd, mt1dsen
from .EmDataPoint import EmDataPoint
from ...model.Model import Model
from ...model.Model1D import Model1D
from ...system.MTSystem import MTSystem
import matplotlib.pyplot as plt
import numpy as np
#from ....base import Error as Err
from ....base.customFunctions import safeEval
from ....base import customFunctions as cf

from ....base import customPlots as cp


class MTDataPoint(EmDataPoint):
    """Class extension to geobipy.EmDataPoint

    Class describes a magneto-telluric electro-magnetic data point with 3D coordinates, observed data, 
    error estimates, predicted data, and a system that describes the aquisition parameters.

    MTDataPoint(x, y, z)

    """

    def __init__(self, x, y, z, e=0.0, d=None, s=None, sys=None):
        # x coordinate
        self.x = StatArray(1) + x
        # y coordinate
        self.y = StatArray(1) + y
        # z coordinate
        self.z = StatArray(1, 'Height above ground', 'm') + z
        # Elevation of data point
        self.e = StatArray(1) + e

        # Assign the number of systems as 1
        self.nSystems = 1
        if (sys is None):
            return

        # EMSystem Class
        if (isinstance(sys, str)):
            self.sys = MTSystem()
            self.sys.read(sys)
        elif (isinstance(sys, MTSystem)):
            self.sys = deepcopy(sys)
        else:
            assert False, TypeError("Sys must be a path to the system file or an MTSystem class")

        # StatArray of apparent Resisitivity and Phase Data
        if (not d is None):
            assert d.size == 2 * self.sys.nFreq, ValueError("Number of data do not match the number of frequencies in the system file {} \n You need to provide an apparent resisitivity and phase for each frequency".format(2*self.sys.nFreq))
            self.d = deepcopy(d)
        else:
            self.d = StatArray(np.zeros(2 * self.sys.nFreq), name='MT data')

        # StatArray of Standard Deviations
        if (not s is None):
            assert s.size == 2 * self.sys.nFreq, ValueError("Number of standard deviations do not match the number of frequencies in the system file {} \n You need to provide an apparent resisitivity and phase for each frequency".format(2*self.sys.nFreq))
            self.s = StatArray(s, 'Standard Deviation', self.d.units)
        else:
            self.s = StatArray(np.ones(2 * self.sys.nFreq), 'Standard Deviation')

        # StatArray of Predicted Data
        self.p = StatArray(2 * self.sys.nFreq, 'Predicted Data')
        # StatArray of Relative Errors
        self.relErr = StatArray(self.nSystems, '$\epsilon_{Relative}x10^{2}$','%')
        # StatArray of Additive Errors
        self.addErr = StatArray(self.nSystems, '$\epsilon_{Additive}$')
        # Initialize the sensitivity matrix
        self.J = None
        # Index to non NaN values
        self.iActive = self.getActiveData()
        # Complex C.  Coefficients to save between forward modelling and sensitivity
        self.c = None


    @property
    def apparentResistivity(self):
        """Returns the apparent resistivity
        
        Returns
        -------
        out : geobipy.StatArray
            Apparent Resistivity

        """

        return StatArray(self.d[:self.sys.nFreq], name='Apparent Resistivity', units='$\Omega m$')
        

    @property
    def phase(self):
        """Returns the phase
        
        Returns
        -------
        out : geobipy.StatArray
            Impedance Phase
            
        """

        return StatArray(self.d[self.sys.nFreq:], name='Impedance Phase', units='$^{o}$')


    @property
    def nFrequencies(self):
        """The number of measurement frequencies."""
        return self.sys.nFreq


    @property
    def noise(self):
        """Generates values from a random normal distribution with a mean of 0.0 and standard deviation self.s
        
        Returns
        -------
        out : array_like
            Noise values of size 2 * self.nFrequencies

        """
        norm = 0.0
        tmp = 2.0 * self.nFrequencies
        while np.abs(norm - tmp) / tmp > 0.1:
            x = np.random.randn(2*self.nFrequencies) * self.s[:]
            norm = np.linalg.norm(x)
        x.name = None
        return x


    @property
    def predictedApparentResistivity(self):
        # Returns the apparent resistivity
        return StatArray(self.p[:self.sys.nFreq], name='Predicted Apparent Resistivity', units='$\Omega m$')
        
    @property
    def predictedPhase(self):
        # Returns the phase
        return StatArray(self.p[self.sys.nFreq:], name='Predcited Impedance Phase', units='$^{o}$')


    def deepcopy(self):
        """ Define a deepcopy routine """
        tmp = MTDataPoint(self.x, self.y, self.z, self.e, self.d, self.s)
        tmp.z = self.z.deepcopy()
        tmp.nSystems = self.nSystems        
        # Initialize the sensitivity matrix
        tmp.J = deepcopy(self.J)
        # EMSystem Class
        tmp.sys = self.sys
        # StatArray of Predicted Data
        tmp.p = self.p.deepcopy()
        # StatArray of Relative Errors
        tmp.relErr = self.relErr.deepcopy()
        # StatArray of Additive Errors
        tmp.addErr = self.addErr.deepcopy()
        # Index to non NaN values
        tmp.iActive = tmp.getActiveData()
        tmp.c = deepcopy(self.c)
        return tmp


    def getChannels(self, system=0):
        """ Return the frequencies in a StatArray """
        return self.sys.freq


    def hdfName(self):
        """ Reprodicibility procedure """
        return('MTDataPoint(0.0,0.0,0.0,0.0)')

      
    def summary(self, out=False):
        """ Print a summary of the MTDataPoint """
        msg = 'MT Data Point: \n'
        msg += 'x: :' + str(self.x) + '\n'
        msg += 'y: :' + str(self.y) + '\n'
        msg += 'z: :' + str(self.z) + '\n'
        msg += 'e: :' + str(self.e) + '\n'
        msg += self.apparentResistivity.summary(True)
        msg += self.phase.summary(True)
        msg += self.predictedApparentResistivity.summary(True)
        msg += self.predictedPhase.summary(True)
        msg += self.s.summary(True)
        if (out):
            return msg
        print(msg)


    # def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
    #     """ Create the hdf group metadata in file
    #     parent: HDF object to create a group inside
    #     myName: Name of the group
    #     """
    #     # create a new group inside h5obj
    #     grp = parent.create_group(myName)
    #     grp.attrs["repr"] = self.hdfName()
    #     self.x.createHdf(grp, 'x', nRepeats=nRepeats, fillvalue=fillvalue)
    #     self.y.createHdf(grp, 'y', nRepeats=nRepeats, fillvalue=fillvalue)
    #     self.z.createHdf(grp, 'z', nRepeats=nRepeats, fillvalue=fillvalue)
    #     self.e.createHdf(grp, 'e', nRepeats=nRepeats, fillvalue=fillvalue)
    #     self.d.createHdf(grp, 'd', nRepeats=nRepeats, fillvalue=fillvalue)
    #     self.s.createHdf(grp, 's', nRepeats=nRepeats, fillvalue=fillvalue)
    #     self.p.createHdf(grp, 'p', nRepeats=nRepeats, fillvalue=fillvalue)
    #     self.relErr.createHdf(grp, 'relErr', nRepeats=nRepeats, fillvalue=fillvalue)
    #     self.addErr.createHdf(grp, 'addErr', nRepeats=nRepeats, fillvalue=fillvalue)
    #     self.calibration.createHdf(grp, 'calibration', nRepeats=nRepeats, fillvalue=fillvalue)
    #     self.sys.toHdf(grp, 'sys')


    # def writeHdf(self, parent, myName, index=None):
    #     """ Write the StatArray to an HDF object
    #     parent: Upper hdf file or group
    #     myName: object hdf name. Assumes createHdf has already been called
    #     create: optionally create the data set as well before writing
    #     """
    #     grp = parent.get(myName)

    #     self.x.writeHdf(grp, 'x',  index=index)
    #     self.y.writeHdf(grp, 'y',  index=index)
    #     self.z.writeHdf(grp, 'z',  index=index)
    #     self.e.writeHdf(grp, 'e',  index=index)

    #     self.d.writeHdf(grp, 'd',  index=index)
    #     self.s.writeHdf(grp, 's',  index=index)
    #     self.p.writeHdf(grp, 'p',  index=index)
    #     self.relErr.writeHdf(grp, 'relErr',  index=index)
    #     self.addErr.writeHdf(grp, 'addErr',  index=index)
    #     self.calibration.writeHdf(grp, 'calibration',  index=index)


    # def fromHdf(self, grp, index=None, **kwargs):
    #     """ Reads the object from a HDF group """

    #     item = grp.get('x')
    #     obj = eval(safeEval(item.attrs.get('repr')))
    #     x = obj.fromHdf(item, index=index)

    #     item = grp.get('y')
    #     obj = eval(safeEval(item.attrs.get('repr')))
    #     y = obj.fromHdf(item, index=index)

    #     item = grp.get('z')
    #     obj = eval(safeEval(item.attrs.get('repr')))
    #     z = obj.fromHdf(item, index=index)

    #     item = grp.get('e')
    #     obj = eval(safeEval(item.attrs.get('repr')))
    #     e = obj.fromHdf(item, index=index)

    #     _aPoint = FdemDataPoint(x, y, z, e)

    #     item = grp.get('sys')
    #     obj = eval(safeEval(item.attrs.get('repr')))
    #     _aPoint.sys = obj.fromHdf(item)

    #     slic = None
    #     if not index is None:
    #         slic=np.s_[index,:]
    #     item = grp.get('d')
    #     obj = eval(safeEval(item.attrs.get('repr')))
    #     _aPoint.d = obj.fromHdf(item, index=slic)

    #     item = grp.get('s')
    #     obj = eval(safeEval(item.attrs.get('repr')))
    #     _aPoint.s = obj.fromHdf(item, index=slic)

    #     item = grp.get('p')
    #     obj = eval(safeEval(item.attrs.get('repr')))
    #     _aPoint.p = obj.fromHdf(item, index=slic)

    #     item = grp.get('relErr')
    #     obj = eval(safeEval(item.attrs.get('repr')))
    #     _aPoint.relErr = obj.fromHdf(item, index=index)

    #     item = grp.get('addErr')
    #     obj = eval(safeEval(item.attrs.get('repr')))
    #     _aPoint.addErr = obj.fromHdf(item, index=index)

    #     item = grp.get('calibration')
    #     obj = eval(safeEval(item.attrs.get('repr')))
    #     _aPoint.calibration = obj.fromHdf(item, index=slic)

    #     _aPoint.iActive = _aPoint.getActiveData()
    #     return _aPoint
        

    def plot(self, title='Magneto Telluric Data', **kwargs):
        """ Plot the apparent resistivity and phase Data
        if plotPredicted then the predicted data are plotted as a line, with points for the observed data
        else the observed data with error bars and linear interpolation are shown.
        Additional options
        incolor
        inmarker
        quadcolor
        quadmarker
        """

        arColor = kwargs.pop('arcolor',cp.wellSeparated[0])
        phaseColor = kwargs.pop('phasecolor',cp.wellSeparated[1])
        arm = kwargs.pop('armarker','v')
        pm = kwargs.pop('phasemarker','o')
        ms = kwargs.pop('markersize',7)
        mec = kwargs.pop('markeredgecolor','k')
        mew = kwargs.pop('markeredgewidth',1.0)
        a = kwargs.pop('alpha',0.8)
        ls = kwargs.pop('linestyle','none')
        lw = kwargs.pop('linewidth',2)
        xs = kwargs.pop('xscale', 'log')
        ys = kwargs.pop('yscale', 'log')

        ax = []
        ax1 = plt.gca()

        cp.xlabel('Frequency (Hz)')
        cp.title(title)

        appRes = self.apparentResistivity
        cp.ylabel(appRes.getNameUnits(), color = arColor)
        ax1.tick_params('y', colors=arColor)

        ax1.errorbar(self.sys.freq, appRes, yerr=self.s[:self.sys.nFreq],
            marker=arm,
            markersize=ms,
            color=arColor,
            markerfacecolor=arColor,
            markeredgecolor=mec,
            markeredgewidth=mew,
            alpha=a,
            linestyle=ls,
            linewidth=lw,
            **kwargs)

        phase = self.phase
        ax2 = ax1.twinx()
        cp.ylabel(phase.getNameUnits(), color = phaseColor)
        ax2.tick_params('y', colors=phaseColor)
        ax2.get_yaxis().tick_right()

        ax2.errorbar(self.sys.freq, phase, yerr=self.s[self.sys.nFreq:],
            marker=pm,
            markersize=ms,
            color=phaseColor,
            markerfacecolor=phaseColor,
            markeredgecolor=mec,
            markeredgewidth=mew,
            alpha=a,
            linestyle=ls,
            linewidth=lw,
            **kwargs)

        plt.xscale(xs)

        ax.append(ax1)
        ax.append(ax2)

        return ax


    def plotPredicted(self, title='Magneto Telluric Data', ax=None, **kwargs):

        if (ax is None):
            ax1 = plt.gca()
        else:
            ax1 = ax[0]

        noLabels = kwargs.pop('nolabels', False)

        if (not noLabels):
            cp.xlabel('Frequency (Hz)')
            cp.title(title)

        arColor = kwargs.pop('arcolor', cp.wellSeparated[3])
        lw = kwargs.pop('linewidth', 2)
        a = kwargs.pop('alpha', 0.7)

        xscale = kwargs.pop('xscale','log')
        yscale = kwargs.pop('yscale','linear')

        appRes = self.predictedApparentResistivity

        ax1.plot(self.sys.freq, appRes, color=arColor, linewidth=lw, alpha=a, **kwargs)

        if (not noLabels):
            cp.ylabel(appRes.getNameUnits(), color = arColor)
            ax1.tick_params('y', colors = arColor)

        if (ax is None):
            ax2 = ax1.twinx()
        else:
            ax2 = ax[1]

        phaseColor = kwargs.pop('phasecolor', cp.wellSeparated[3])
        phase = self.predictedPhase
        ax2.plot(self.sys.freq, phase, color=phaseColor, linewidth=lw, alpha=a, **kwargs)

        if (not noLabels):
            cp.ylabel(phase.getNameUnits(), color = phaseColor)
            ax2.tick_params('y', colors = phaseColor)
            ax2.get_yaxis().tick_right()

        plt.xscale(xscale)
        plt.yscale(yscale)

    
    def updateErrors(self, relativeErr, additiveErr):
        """Updates the data errors

        Updates the standard deviation of the data errors using the following model

        .. math::
            \sqrt{(\mathbf{\epsilon}_{rel} \mathbf{d}^{obs})^{2} + \mathbf{\epsilon}^{2}_{add}},
        where :math:`\mathbf{\epsilon}_{rel}` is the relative error, a percentage fraction and :math:`\mathbf{\epsilon}_{add}` is the additive error.
        
        Parameters
        ----------  
        relativeErr : array_like or list of array_like
            A fraction percentage that is multiplied by the observed data. 
            If array_like: should have length 2. The entries correspond to the apparent resisitivity and phase relative errors. 
            If list, each item should have length equal to the number of frequencies. First array for apparent resistivity, the second for phase.
        additiveErr : array_like or list of array_like
            An absolute value of additive error. 
            If array_like: should have length 2. The entries correspond to the apparent resisitivity and phase additive errors. 
            If list, each item should have length equal to the number of frequencies. First array for apparent resistivity, the second for phase.

        Raises
        ------
        TypeError
            If the length of relativeErr or additiveErr is not equal to 2 if arrays are given.
        TypeError
            If any item in the relativeErr or additiveErr lists are not of length equal to the number of frequencies
        ValueError
            If any relative or additive errors are <= 0.0

        """

        assert (np.size(relativeErr) == 2), TypeError("relativeErr must have size equal to 2.")
        assert (np.size(additiveErr) == 2), TypeError("additiveErr must have size equal to 2.")

        # For each system assign error levels using the user inputs
        assert np.all([rel > 0.0 for rel in relativeErr]), ValueError("relativeErr must be > 0.0")
        assert np.all([add > 0.0 for add in additiveErr]), ValueError("additiveErr must be > 0.0")

        if isinstance(relativeErr, list):
            for rel in relativeErr:
                assert np.size(rel) == self.nFrequencies, ValueError("Each item in relativeErr must have length equal to {}".format(self.nFrequencies))
        if isinstance(additiveErr, list):
            for add in additiveErr:
                assert np.size(add) == self.nFrequencies, ValueError("Each item in additiveErr must have length equal to {}".format(self.nFrequencies))
        
        tmp = np.zeros(2 * self.nFrequencies)
        tmp[:self.nFrequencies] = (relativeErr[0] * self.d[:self.sys.nFreq])**2.0 + additiveErr[0]**2.0
        tmp[self.nFrequencies:] = (relativeErr[1] * self.d[self.sys.nFreq:])**2.0 + additiveErr[1]**2.0

        # Update the variance of the predicted data prior
        if self.p.hasPrior():
            self.p.prior.variance[:] = tmp[self.iActive]

        self.s[:] = np.sqrt(tmp)


    # def addErrors(self):
    #         """ Add errors using stuff
    #         """
    #         assert 0 <= option <= 2, ValueError("Use an option [0,1,2]")

    #         if (not relativeErr is None):
    #             assert 0.0 <= relativeErr <= 1.0, ValueError("0.0 <= relativeErr <= 1.0")

    #         if (option == 0):
    #             self.s[:] = err
    #         elif (option == 1):
    #             self.s[:] = np.maximum(relativeErr * self.d, addError)
    #         elif (option == 2):
    #             tmp = (relativeErr * self.d)**2.0
    #             self.s[:] = np.sqrt(tmp + addError**2.0)

    #         assert not self.p.prior is None, "No prior has been assigned to the predicted data"

    #         self.p.prior.variance[:] = self.s[self.iActive]**2.0

    # def updateSensitivity(self, J, mod, option, scale=False):
    #     """ Compute an updated sensitivity matrix based on the one already containined in the TdemDataPoint object  """
    #     # If there is no matrix saved in the data object, compute the entire
    #     # thing
    #     return self.sensitivity(mod, scale=scale)


    def FindBestHalfSpace(self):
        """ Uses the bisection approach to find a half space conductivity that best matches the EM data by minimizing the data misfit """
        # ####lg.myLogger('Global');####lg.indent()
        ####lg.info('Finding Best Half Space Model')
        c1 = 2.0
        c2 = -6.0
        cnew = 0.5 * (c1 + c2)
        # Initialize a single layer model
        mod = Model1D(1)
        # Initialize the first conductivity
        mod.par[0] = 10.0**c1
        self.forward(mod)  # Forward model the EM data
        PhiD1 = self.dataMisfit(squared=True)  # Compute the measure between observed and predicted data
        # Initialize the second conductivity
        mod.par[0] = 10.0**c2
        self.forward(mod)  # Forward model the EM data
        PhiD2 = self.dataMisfit(squared=True)  # Compute the measure between observed and predicted data
        # Compute a relative change in the data misfit
        dPhiD = abs(PhiD2 - PhiD1) / PhiD2
        i = 1
        ####lg.debug('Entering Bisection')
        # Continue until there is less than 1% change
        while (dPhiD > 0.01 and i < 100):
            ####lg.debug('c1,c2,cnew: '+str([c1,c2,cnew]))
            cnew = 0.5 * (c1 + c2)  # Bisect the conductivities
            mod.par[0] = 10.0**cnew
            self.forward(mod)  # Forward model the EM data
            PhiDnew = self.dataMisfit(squared=True)
            if (PhiD2 > PhiDnew):
                c2 = cnew
                PhiD2 = PhiDnew
            elif (PhiD1 > PhiDnew):
                c1 = cnew
                PhiD1 = PhiDnew
            dPhiD = abs(PhiD2 - PhiD1) / PhiD2
            i += 1
        # ####lg.dedent()
        return np.float64(10.0**cnew)


    def forward(self, mod):
        """ Forward model the data from the given model """

        assert isinstance(mod, Model), TypeError("Invalid model class for forward modeling [1D]")

        self._forward1D(mod)


    def sensitivity(self, mod, scale=False):
        """ Compute the sensitivty matrix for the given model """

        assert isinstance(mod, Model), TypeError("Invalid model class for sensitivity matrix [1D]")

        return StatArray(self._sensitivity1D(mod, scale), 'Sensitivity')


    def _forward1D(self, mod):
        """ Forward model the data from a 1D layered earth model """
        self.p[:], self.c = mt1dfwd(self.sys, mod, -self.z[0])


    def _sensitivity1D(self, mod, scale=False):
        """ Compute the sensitivty matrix for a 1D layered earth model """

        if (mod.nCells[0] != self.c.shape[1]):
            self.p[:], self.c = mt1dfwd(self.sys, mod, -self.z[0])

        J = mt1dsen(self.sys, mod, -self.z[0], self.c)

        # # Re-arrange the sensitivity matrix to Real:Imaginary vertical
        # # concatenation
        # J = np.zeros([2 * self.sys.nFreq, mod.nCells[0]])
        # J[:self.sys.nFreq, :] = Jtmp.real
        # J[self.sys.nFreq:, :] = Jtmp.imag

        # Scale the sensitivity matrix rows by the data weights if required
        if scale:
            J *= (np.repeat(self.s[:, np.newaxis]**-1.0, np.size(J, 1), 1))

        J = J[self.iActive, :]
        return J
