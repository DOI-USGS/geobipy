""" @FdemDataPoint_Class
Module describing a frequency domain EMData Point that contains a single measurement.
"""
from copy import copy, deepcopy
from ....classes.core.StatArray import StatArray
#from ...forwardmodelling.EMfor1D_F import fdem1dfwd
from ...forwardmodelling.EMfor1D_F import fdem1dfwd, fdem1dsen
from .EmDataPoint import EmDataPoint
from ...model.Model import Model
from ...model.Model1D import Model1D
from ....base.logging import myLogger
from ...system.FdemSystem import FdemSystem
import matplotlib.pyplot as plt
import numpy as np
#from ....base import Error as Err
from ....base.customFunctions import safeEval
from ....base import customFunctions as cf

from ....base import customPlots as cp


class FdemDataPoint(EmDataPoint):
    """Class extension to geobipy.EmDataPoint

    Class describes a frequency domain electro magnetic data point with 3D coordinates, observed data, 
    error estimates, predicted data, and a system that describes the aquisition parameters.

    FdemDataPoint(x, y, z)

    """

    def __init__(self, x, y, z, e=0.0, d=None, s=None, sys=None, dataUnits='ppm'):
        # x coordinate
        self.x = StatArray(1)+x
        # y coordinate
        self.y = StatArray(1)+y
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
            tmpsys = FdemSystem()
            tmpsys.read(sys)
            self.sys = deepcopy(tmpsys)
        elif (isinstance(sys, FdemSystem)):
            self.sys = deepcopy(sys)
        else:
            assert False, TypeError("Sys must be a path to the system file or an FdemSystem class")

        # StatArray of InPhase and Quadrature Data
        if (not d is None):
            assert d.size == 2 * self.sys.nFreq, ValueError("Number of data do not match the number of frequencies in the system file "+str(2*self.sys.nFreq))
            self.d = deepcopy(d)
        else:
            self.d = StatArray(np.zeros(2 * self.sys.nFreq), name='Frequency domain data', units=dataUnits)
        #StatArray(2*sys.nFreq,'Observed Data',d.units);self.d+=d

        # StatArray of Standard Deviations

        if (not s is None):
            assert s.size == 2 * self.sys.nFreq, ValueError("Number of standard deviations do not match the number of frequencies in the system file "+str(2*self.sys.nFreq))
            self.s = StatArray(s, 'Standard Deviation', self.d.units)
        else:
            self.s = StatArray(np.ones(2 * self.sys.nFreq), 'Standard Deviation', self.d.units)

        # StatArray of Predicted Data
        self.p = StatArray(2 * self.sys.nFreq, 'Predicted Data', self.d.units)
        # StatArray of Relative Errors
        self.relErr = StatArray(self.nSystems, '$\epsilon_{Relative}x10^{2}$','%')
        # StatArray of Additive Errors
        self.addErr = StatArray(self.nSystems, '$\epsilon_{Additive}$',self.d.units)
        # StatArray of calibration parameters
        # The four columns are Bias,Variance,InphaseBias,QuadratureBias.
        self.calibration = StatArray([self.sys.nFreq * 4], 'Calibration Parameters')
        # Initialize the sensitivity matrix
        self.J = None
        # Index to non NaN values
        self.iActive = self.getActiveData()


    def deepcopy(self):
        """ Define a deepcopy routine """
        tmp = FdemDataPoint(self.x, self.y, self.z, self.e)
        tmp.z = self.z.deepcopy()
        # Assign the number of systems as 1
        tmp.nSystems = 1
        # Initialize the sensitivity matrix
        tmp.J = deepcopy(self.J)
        # EMSystem Class
        tmp.sys = self.sys
        # StatArray of Data
        tmp.d = self.d.deepcopy()
        # StatArray of Standard Deviations
        tmp.s = self.s.deepcopy()
        # StatArray of Predicted Data
        tmp.p = self.p.deepcopy()
        # StatArray of Relative Errors
        tmp.relErr = self.relErr.deepcopy()
#    tmp.relErr.name='Relative Errors';tmp.relErr.units='ppm'
        # StatArray of Additive Errors
        tmp.addErr = self.addErr.deepcopy()
#    tmp.name='Relative Errors';tmp.units='ppm'
        # StatArray of calibration parameters
        # The four columns are Bias,Variance,InphaseBias,QuadratureBias.
        tmp.calibration = self.calibration.deepcopy()
        # Index to non NaN values
        tmp.iActive = tmp.getActiveData()
        return tmp


    def getChannels(self, system=0):
        """ Return the frequencies in an StatArray """
        return StatArray(self.sys.freq, name='Frequency', units='Hz')


    def hdfName(self):
        """ Reprodicibility procedure """
        return('FdemDataPoint(0.0,0.0,0.0,0.0)')


    def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self.x.createHdf(grp, 'x', nRepeats=nRepeats, fillvalue=fillvalue)
        self.y.createHdf(grp, 'y', nRepeats=nRepeats, fillvalue=fillvalue)
        self.z.createHdf(grp, 'z', nRepeats=nRepeats, fillvalue=fillvalue)
        self.e.createHdf(grp, 'e', nRepeats=nRepeats, fillvalue=fillvalue)
        self.d.createHdf(grp, 'd', nRepeats=nRepeats, fillvalue=fillvalue)
        self.s.createHdf(grp, 's', nRepeats=nRepeats, fillvalue=fillvalue)
        self.p.createHdf(grp, 'p', nRepeats=nRepeats, fillvalue=fillvalue)
        self.relErr.createHdf(grp, 'relErr', nRepeats=nRepeats, fillvalue=fillvalue)
        self.addErr.createHdf(grp, 'addErr', nRepeats=nRepeats, fillvalue=fillvalue)
        self.calibration.createHdf(grp, 'calibration', nRepeats=nRepeats, fillvalue=fillvalue)
        self.sys.toHdf(grp, 'sys')


    def writeHdf(self, parent, myName, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        grp = parent.get(myName)

        self.x.writeHdf(grp, 'x',  index=index)
        self.y.writeHdf(grp, 'y',  index=index)
        self.z.writeHdf(grp, 'z',  index=index)
        self.e.writeHdf(grp, 'e',  index=index)

        self.d.writeHdf(grp, 'd',  index=index)
        self.s.writeHdf(grp, 's',  index=index)
        self.p.writeHdf(grp, 'p',  index=index)
        self.relErr.writeHdf(grp, 'relErr',  index=index)
        self.addErr.writeHdf(grp, 'addErr',  index=index)
        self.calibration.writeHdf(grp, 'calibration',  index=index)


    def fromHdf(self, grp, index=None, **kwargs):
        """ Reads the object from a HDF group """

        item = grp.get('x')
        obj = eval(safeEval(item.attrs.get('repr')))
        x = obj.fromHdf(item, index=index)

        item = grp.get('y')
        obj = eval(safeEval(item.attrs.get('repr')))
        y = obj.fromHdf(item, index=index)

        item = grp.get('z')
        obj = eval(safeEval(item.attrs.get('repr')))
        z = obj.fromHdf(item, index=index)

        item = grp.get('e')
        obj = eval(safeEval(item.attrs.get('repr')))
        e = obj.fromHdf(item, index=index)

        _aPoint = FdemDataPoint(x, y, z, e)

        item = grp.get('sys')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.sys = obj.fromHdf(item)

        slic = None
        if not index is None:
            slic=np.s_[index,:]
        item = grp.get('d')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.d = obj.fromHdf(item, index=slic)

        item = grp.get('s')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.s = obj.fromHdf(item, index=slic)

        item = grp.get('p')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.p = obj.fromHdf(item, index=slic)

        item = grp.get('relErr')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.relErr = obj.fromHdf(item, index=index)

        item = grp.get('addErr')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.addErr = obj.fromHdf(item, index=index)

        item = grp.get('calibration')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.calibration = obj.fromHdf(item, index=slic)

        _aPoint.iActive = _aPoint.getActiveData()
        return _aPoint
        

    def calibrate(self, Predicted=True):
        """ Apply calibration factors to the data point """
        # Make complex numbers from the data
        if (Predicted):
            tmp = cf.mergeComplex(self.p)
        else:
            tmp = cf.mergeComplex(self.d)

        # Get the calibration factors for each frequency
        i1 = 0
        i2 = self.sys.nFreq
        G = self.calibration[i1:i2]
        i1 += self.sys.nFreq
        i2 += self.sys.nFreq
        Phi = self.calibration[i1:i2]
        i1 += self.sys.nFreq
        i2 += self.sys.nFreq
        Bi = self.calibration[i1:i2]
        i1 += self.sys.nFreq
        i2 += self.sys.nFreq
        Bq = self.calibration[i1:i2]

        # Calibrate the data
        tmp[:] = G * np.exp(1j * Phi) * tmp + Bi + (1j * Bq)

        # Split the complex numbers back out
        if (Predicted):
            self.p[:] = cf.splitComplex(tmp)
        else:
            self.d[:] = cf.splitComplex(tmp)

#  def evaluatePrior(self,sErr,sZ,sCal):
#    """ Evaluate the prior for the EM data point
#    sErr: :Include the prior for the relative error
#    sZ:   :Include the prior for elevation
#    sCal: :Include the prior for calibration parameters
#    """
#    prior=0.0
#    if sErr: # Relative Errors
#      tmp=np.log(self.relErr.probability())
#      prior+=tmp
##      print('Errors: ',tmp)
#    if sZ:# Elevation
#      tmp=np.log(self.z.probability())
#      prior+=tmp
##      print('Elevation: ',tmp)
#    if sCal: # Calibration parameters
#      tmp=self.calibration.probability()
#      prior+=tmp
##      print('Calibration: ',tmp)
#    return np.float64(prior)

#  def summary(self,out=False):
#    """ Print a summary of the EMdataPoint """
#    msg='EM Data Point: \n'
#    msg+='x: :'+str(self.x)+'\n'
#    print('y: :'+str(self.y)+'\n'
#    print('z: :'+str(self.z)+'\n'
#    self.d.summary()
#    self.p.summary()
#    self.s.summary()
#    self.sys.summary()
#    print('')

    def plot(self, title='', **kwargs):
        """ Plot the Inphase and Quadrature Data for an EM measurement
        if plotPredicted then the predicted data are plotted as a line, with points for the observed data
        else the observed data with error bars and linear interpolation are shown.
        Additional options
        incolor
        inmarker
        quadcolor
        quadmarker
        """

        ax = plt.gca()
        cp.pretty(ax)

        cp.xlabel('Frequency (Hz)')
        cp.ylabel('Data (ppm)')
        cp.title(title)

        inColor = kwargs.pop('incolor',cp.wellSeparated[0])
        quadColor = kwargs.pop('quadcolor',cp.wellSeparated[1])
        im = kwargs.pop('inmarker','v')
        qm = kwargs.pop('quadmarker','o')
        ms = kwargs.pop('markersize',7)
        mec = kwargs.pop('markeredgecolor','k')
        mew = kwargs.pop('markeredgewidth',1.0)
        a = kwargs.pop('alpha',0.8)
        ls = kwargs.pop('linestyle','none')
        lw = kwargs.pop('linewidth',2)

        xscale = kwargs.pop('xscale','log')
        yscale = kwargs.pop('yscale','log')

        plt.errorbar(self.sys.freq, self.d[:self.sys.nFreq], yerr=self.s[:self.sys.nFreq],
            marker=im,
            markersize=ms,
            color=inColor,
            markerfacecolor=inColor,
            markeredgecolor=mec,
            markeredgewidth=mew,
            alpha=a,
            linestyle=ls,
            linewidth=lw,
            label='In-Phase', **kwargs)

        plt.errorbar(self.sys.freq, self.d[self.sys.nFreq:], yerr=self.s[self.sys.nFreq:],
            marker=qm,
            markersize=ms,
            color=quadColor,
            markerfacecolor=quadColor,
            markeredgecolor=mec,
            markeredgewidth=mew,
            alpha=a,
            linestyle=ls,
            linewidth=lw,
            label='Quadrature', **kwargs)

        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.legend()

        return ax


    def plotPredicted(self, title='',**kwargs):

        ax = plt.gca()
        cp.pretty(ax)

        cp.xlabel('Frequency (Hz)')
        cp.ylabel('Data (ppm)')
        cp.title(title)

        c = kwargs.pop('color',cp.wellSeparated[3])
        lw = kwargs.pop('linewidth',2)
        a = kwargs.pop('alpha',0.7)

        xscale = kwargs.pop('xscale','log')
        yscale = kwargs.pop('yscale','log')

        plt.semilogx(self.sys.freq, self.p[:self.sys.nFreq], color=c, linewidth=lw, alpha=a, **kwargs)
        plt.semilogx(self.sys.freq, self.p[self.sys.nFreq:], color=c, linewidth=lw, alpha=a, **kwargs)

        plt.xscale(xscale)
        plt.yscale(yscale)

        return ax


    def scaleJ(self, Jin, power=1.0):
        """ Scales a matrix by the errors in the given data
        Useful if a sensitivity matrix is generated using one data point, but must be scaled by the errors in another """
        J1 = np.zeros(Jin.shape)
        J1[:, :] = Jin * (np.repeat(self.s[self.iActive, np.newaxis] ** -power, np.size(J1, 1), 1))
        return J1


    def updateErrors(self, option, err=None, relativeErr=None, addError=None):
        """ Updates the data errors
        option:      :[0,1,2]
                   0 :Assign err to the data errors
                   1 :Use a percentage of the data with a minimum floor
                   2 :Norm of the percentage and minimum floor
        err:         :Specified Errors (only used with option 1)
        relativeErr: :Percentage of the data
        addError:    :StatArray of Minimum error floors
        """
        assert 0 <= option <= 2, ValueError("Use an option [0,1,2]")

        if (not relativeErr is None):
            assert 0.0 <= relativeErr <= 1.0, ValueError("0.0 <= relativeErr <= 1.0")

        if (option == 0):
            self.s[:] = err
        elif (option == 1):
            self.s[:] = np.maximum(relativeErr * self.d, addError)
        elif (option == 2):
            tmp = (relativeErr * self.d)**2.0
            self.s[:] = np.sqrt(tmp + addError**2.0)

        assert not self.p.prior is None, "No prior has been assigned to the predicted data"

        self.p.prior.variance[:] = self.s[self.iActive]**2.0

    def addErrors(self):
            """ Add errors using stuff
            """
            assert 0 <= option <= 2, ValueError("Use an option [0,1,2]")

            if (not relativeErr is None):
                assert 0.0 <= relativeErr <= 1.0, ValueError("0.0 <= relativeErr <= 1.0")

            if (option == 0):
                self.s[:] = err
            elif (option == 1):
                self.s[:] = np.maximum(relativeErr * self.d, addError)
            elif (option == 2):
                tmp = (relativeErr * self.d)**2.0
                self.s[:] = np.sqrt(tmp + addError**2.0)

            assert not self.p.prior is None, "No prior has been assigned to the predicted data"

            self.p.prior.variance[:] = self.s[self.iActive]**2.0

    def updateSensitivity(self, J, mod, option, scale=False):
        """ Compute an updated sensitivity matrix based on the one already containined in the TdemDataPoint object  """
        # If there is no matrix saved in the data object, compute the entire
        # thing
        return self.sensitivity(mod, scale=scale)


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

        return StatArray(self._sensitivity1D(mod, scale), 'Sensitivity', '$\\frac{ppm.m}{S}$')


    def _forward1D(self, mod):
        """ Forward model the data from a 1D layered earth model """
        tmp = fdem1dfwd(self.sys, mod, -self.z[0])
        self.p[:self.sys.nFreq] = tmp.real
        self.p[self.sys.nFreq:] = tmp.imag


    def _sensitivity1D(self, mod, scale=False):
        """ Compute the sensitivty matrix for a 1D layered earth model """
        Jtmp = fdem1dsen(self.sys, mod, -self.z[0])
        # Re-arrange the sensitivity matrix to Real:Imaginary vertical
        # concatenation
        J = np.zeros([2 * self.sys.nFreq, mod.nCells[0]])
        J[:self.sys.nFreq, :] = Jtmp.real
        J[self.sys.nFreq:, :] = Jtmp.imag
        # Scale the sensitivity matrix rows by the data weights if required
        if scale:
            J *= (np.repeat(self.s[:, np.newaxis]**-1.0, np.size(J, 1), 1))

        J = J[self.iActive, :]
        return J
