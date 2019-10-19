from copy import deepcopy
from ....classes.core import StatArray
from ...model.Model import Model
from .EmDataPoint import EmDataPoint
from ...system.EmLoop import EmLoop
from ...system.CircularLoop import CircularLoop
from ....base.logging import myLogger
from ...system.TdemSystem import TdemSystem
from ...statistics.Histogram1D import Histogram1D

try:
    from gatdaem1d import Earth
    from gatdaem1d import Geometry
except:
    h=("Could not find a Time Domain forward modeller. \n"
       "Please see the package's README for instructions on how to install one")
    print(Warning(h))
import matplotlib.pyplot as plt
import numpy as np
#from ....base import Error as Err
from ....base import fileIO as fIO
from ....base import customFunctions as cf
from ....base import customPlots as cp
from ....base import MPI as myMPI
from os.path import split as psplt
from os.path import join


class TdemDataPoint(EmDataPoint):
    """ Initialize a Time domain EMData Point


    TdemDataPoint(x, y, z, elevation, data, std, system, T, R, lineNumber, fiducial)

    Parameters
    ----------
    x : np.float64
        The easting co-ordinate of the data point
    y : np.float64
        The northing co-ordinate of the data point
    z : np.float64
        The height of the data point above ground
    elevation : np.float64, optional
        The elevation of the data point, default is 0.0
    data : list of arrays, optional
        A list of 1D arrays, where each array contains the data in each system.
        The arrays are vertically concatenated inside the TdemDataPoint object
    std : list of arrays, optional
        A list of 1D arrays, where each array contains the errors in each system.
        The arrays are vertically concatenated inside the TdemDataPoint object
    system : TdemSystem, optional
        Time domain system class
    T : EmLoop, optional
        Transmitter loop class
    R : EmLoop, optional
        Receiver loop class
    lineNumber : float, optional
        The line number associated with the datapoint
    fiducial : float, optional
        The fiducial associated with the datapoint

    Returns
    -------
    out : TdemDataPoint
        A time domain EM sounding

    Notes
    -----
    The data argument is a set of lists with length equal to the number of systems.  
    These data are unpacked and vertically concatenated in this class. 
    The parameter self._data will have length equal to the sum of the number of time gates in each system.  
    The same is true for the errors, and the predicted data vector.

    """

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=0.0, data=None, std=None, predictedData=None, system=None, T=None, R=None, loopOffset=[0.0, 0.0, 0.0], lineNumber=0.0, fiducial=0.0):
        """Initializer. """

        if not T is None:
            assert isinstance(T, EmLoop), TypeError("Transmitter must be of type EmLoop")

        if not R is None:
            assert isinstance(R, EmLoop), TypeError("Receiver must be of type EmLoop")

        if (system is None):
            return
        else:
            if isinstance(system, (str, TdemSystem)):
                system = [system]
            assert all((isinstance(sys, (str, TdemSystem)) for sys in system)), TypeError("System must be list with items of type TdemSystem")

        nSystems = len(system)
        nTimes = np.empty(nSystems, dtype=np.int32)
        systems = []
        for j, sys in enumerate(system):
            if isinstance(sys, str):
                systems.append(TdemSystem(sys))
            elif isinstance(sys, TdemSystem):
                systems.append(sys)
            # Number of time gates
            nTimes[j] = systems[j].nwindows()

        nChannels = np.sum(nTimes)

        if not data is None:
            assert data.size == nChannels, ValueError("Size of data must equal total number of time channels {}".format(nChannels))
            # Mask invalid data values less than 0.0 to NaN
            i = np.where(~np.isnan(data))[0]
            i1 = np.where(data[i] <= 0.0)[0]
            data[i[i1]] = np.nan
        if not std is None:
            assert std.size == nChannels, ValueError("Size of std must equal total number of time channels {}".format(nChannels))
        if not predictedData is None:
            assert predictedData.size == nChannels, ValueError("Size of predictedData must equal total number of time channels {}".format(nChannels))

        EmDataPoint.__init__(self, nChannelsPerSystem=nTimes, x=x, y=y, z=z, elevation=elevation, data=data, std=std, predictedData=predictedData, dataUnits=r'$\frac{V}{Am^{4}}$', lineNumber=lineNumber, fiducial=fiducial)

        self._data.name = "Time domain data"

        self.nSystems = nSystems
        self.system = systems

        self.getIplotActive()
                
        # EmLoop Transnmitter
        self.T = deepcopy(T)
        # EmLoop Reciever
        self.R = deepcopy(R)
        # Set the loop offset
        self.loopOffset = StatArray.StatArray(np.asarray(loopOffset), 'Loop Offset', 'm')

        k = 0
        for i in range(self.nSystems):
            # Set the channel names
            for iTime in range(self.nTimes[i]):
                self._channelNames[k] = 'Time {:.3e} s'.format(self.system[i].windows.centre[iTime])
                k += 1


    @property
    def nTimes(self):
        return self.nChannelsPerSystem

    @property
    def nWindows(self):
        return self.nChannels


    def times(self, system=0):
        """ Return the window times in an StatArray """
        return self.system[system].times


    def deepcopy(self):
        """ Define a deepcopy routine """
        return deepcopy(self)


    def __deepcopy__(self, memo):
        out = TdemDataPoint(self.x, self.y, self.z, self.elevation, self._data, self._std, self._predictedData, self.system, self.T, self.R, self.loopOffset, self.lineNumber, self.fiducial)
        # StatArray of Relative Errors
        out.relErr = self.relErr.deepcopy()
        # StatArray of Additive Errors
        out.addErr = self.addErr.deepcopy()
        # Initialize the sensitivity matrix
        out.J = deepcopy(self.J)

        return out


    def getIplotActive(self):
        """ Get the active data indices per system.  Used for plotting. """
        self.iplotActive = []
        i0 = 0
        for i in range(self.nSystems):
            i1 = i0 + self.nTimes[i]
            self.iplotActive.append(cf.findNotNans(self._data[i0:i1]))
            i0 = i1


    def dualMoment(self):
        """ Returns True if the number of systems is > 1 """
        return len(self.system) > 1


    def hdfName(self):
        """ Reprodicibility procedure """
        return('TdemDataPoint(0.0,0.0,0.0,0.0)')


    def createHdf(self, parent, myName, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        grp.create_dataset('nSystems', data=self.nSystems)
        for i in range(self.nSystems):
            grp.create_dataset('System{}'.format(i), data=np.string_(psplt(self.system[i].fileName)[-1]))
        self.x.createHdf(grp, 'x', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.y.createHdf(grp, 'y', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.z.createHdf(grp, 'z', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.elevation.createHdf(grp, 'e', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self._data.createHdf(grp, 'd', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self._std.createHdf(grp, 's', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self._predictedData.createHdf(grp, 'p', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.relErr.createHdf(grp, 'relErr', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.addErr.createHdf(grp, 'addErr', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.T.createHdf(grp, 'T', nRepeats=nRepeats, fillvalue=fillvalue)
        self.R.createHdf(grp, 'R', nRepeats=nRepeats, fillvalue=fillvalue)
        self.loopOffset.createHdf(grp, 'loop_offset', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        if (not index is None):
            assert cf.isInt(index), TypeError('Index must be an int')

        grp = parent.get(myName)

        self.x.writeHdf(grp, 'x', withPosterior=withPosterior, index=index)
        self.y.writeHdf(grp, 'y',  withPosterior=withPosterior, index=index)
        self.z.writeHdf(grp, 'z',  withPosterior=withPosterior, index=index)
        self.elevation.writeHdf(grp, 'e',  withPosterior=withPosterior, index=index)
        self._data.writeHdf(grp, 'd',  withPosterior=withPosterior, index=index)
        self._std.writeHdf(grp, 's',  withPosterior=withPosterior, index=index)
        self._predictedData.writeHdf(grp, 'p',  withPosterior=withPosterior, index=index)
        self.relErr.writeHdf(grp, 'relErr',  withPosterior=withPosterior, index=index)
        self.addErr.writeHdf(grp, 'addErr',  withPosterior=withPosterior, index=index)
        self.T.writeHdf(grp, 'T', index=index)
        self.R.writeHdf(grp, 'R', index=index)
        self.loopOffset.writeHdf(grp, 'loop_offset', index=index)
        #writeNumpy(self.iActive, grp, 'iActive')

#    def toHdf(self, parent, myName):
#        """ Write the TdemDataPoint to an HDF object
#        h5obj: :An HDF File or Group Object.
#        """
#        self.writeHdf(parent, myName, index=np.s_[0])

    def fromHdf(self, grp, index=None, **kwargs):
        """ Reads the object from a HDF group """

        assert ('sysPath' in kwargs), ValueError("missing 1 required argument 'sysPath', the path to directory containing system files")

        sysPath = kwargs.pop('sysPath', None)
        assert (not sysPath is None), ValueError("missing 1 required argument 'sysPath', the path to directory containing system files")
        if (not index is None):
            assert cf.isInt(index), ValueError("index must be of type int")

        item = grp.get('x')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        x = obj.fromHdf(item, index=index)
        item = grp.get('y')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        y = obj.fromHdf(item, index=index)
        item = grp.get('z')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        z = obj.fromHdf(item, index=index)
        item = grp.get('e')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        e = obj.fromHdf(item, index=index)

        nSystems = np.int(np.asarray(grp.get('nSystems')))
        systems = []
        for i in range(nSystems):
            # Get the system file name. h5py has to encode strings using utf-8, so decode it!
            systems.append(TdemSystem(join(sysPath, str(np.asarray(grp.get('System{}'.format(i))), 'utf-8'))))

        _aPoint = TdemDataPoint(x, y, z, e, system=systems)

        slic = None
        if (not index is None):
            slic = np.s_[index,:]

        item = grp.get('d')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        _aPoint._data = obj.fromHdf(item, index=slic)
        item = grp.get('s')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        _aPoint._std = obj.fromHdf(item, index=slic)
        item = grp.get('p')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        _aPoint._predictedData = obj.fromHdf(item, index=slic)
        if (_aPoint.nSystems == 1):
            slic = index
        item = grp.get('relErr')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        _aPoint.relErr = obj.fromHdf(item, index=slic)
        item = grp.get('addErr')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        _aPoint.addErr = obj.fromHdf(item, index=slic)
        item = grp.get('T')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        _aPoint.T = obj.fromHdf(item, index=index)
        item = grp.get('R')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        _aPoint.R = obj.fromHdf(item, index=index)

        try:
            item = grp.get('loop_offset')
            obj = eval(cf.safeEval(item.attrs.get('repr')))
            _aPoint.loopOffset = obj.fromHdf(item, index=index)
        except:
            pass

        _aPoint.iActive = _aPoint.getActiveData()

        _aPoint.getIplotActive()

        return _aPoint


#  def calibrate(self,Predicted=True):
#    """ Apply calibration factors to the data point """
#    # Make complex numbers from the data
#    if (Predicted):
#      tmp=cf.mergeComplex(self._predictedData)
#    else:
#      tmp=cf.mergeComplex(self._data)
#
#    # Get the calibration factors for each frequency
#    i1=0;i2=self.system.nFreq
#    G=self.calibration[i1:i2];i1+=self.system.nFreq;i2+=self.system.nFreq
#    Phi=self.calibration[i1:i2];i1+=self.system.nFreq;i2+=self.system.nFreq
#    Bi=self.calibration[i1:i2];i1+=self.system.nFreq;i2+=self.system.nFreq
#    Bq=self.calibration[i1:i2]
#
#    # Calibrate the data
#    tmp[:]=G*np.exp(1j*Phi)*tmp+Bi+(1j*Bq)
#
#    # Split the complex numbers back out
#    if (Predicted):
#      self._predictedData[:]=cf.splitComplex(tmp)
#    else:
#      self._data[:]=cf.splitComplex(tmp)
#

    def plotWaveform(self,**kwargs):
        for i in range(self.nSystems):
            if (self.nSystems > 1):
                plt.subplot(2, 1, i + 1)
            plt.plot(self.system[i].waveform.time, self.system[i].waveform.current, **kwargs)
            cp.xlabel('Time (s)')
            cp.ylabel('Normalized Current (A)')
            plt.margins(0.1, 0.1)


    def plot(self, title='Time Domain EM Data', with_error_bars=True, **kwargs):
        """ Plot the Inphase and Quadrature Data for an EM measurement
        """
        ax=plt.gca()

        kwargs['marker'] = kwargs.pop('marker', 'v')
        kwargs['markersize'] = kwargs.pop('markersize', 7)
        c = kwargs.pop('color', [cp.wellSeparated[i+1] for i in range(self.nSystems)])
        mfc = kwargs.pop('markerfacecolor',[cp.wellSeparated[i+1] for i in range(self.nSystems)])
        assert len(c) == self.nSystems, ValueError("color must be a list of length {}".format(self.nSystems))
        assert len(mfc) == self.nSystems, ValueError("markerfacecolor must be a list of length {}".format(self.nSystems))
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor', 'k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth', 1.0)
        kwargs['alpha'] = kwargs.pop('alpha', 0.8)
        kwargs['linestyle'] = kwargs.pop('linestyle', 'none')
        kwargs['linewidth'] = kwargs.pop('linewidth', 2)

        xscale = kwargs.pop('xscale', 'log')
        yscale = kwargs.pop('yscale', 'log')

        iJ0 = 0
        for j in range(self.nSystems):
            iAct = self.iplotActive[j]
            iS = self._systemIndices(j)
            d = self._data[iS]
            if (with_error_bars):
                s = self._std[iS]
                plt.errorbar(self.times(j)[iAct], d[iAct], yerr=s[iAct],
                color=c[j],
                markerfacecolor=mfc[j],
                label='System: {}'.format(j+1),
                **kwargs)
            else:
                plt.plot(self.times(j)[iAct], d[iAct],
                markerfacecolor=mfc[j],
                label='System: {}'.format(j+1),
                **kwargs)
            iJ0 += self.system[j].nwindows()


        plt.xscale(xscale)
        plt.yscale(yscale)
        cp.xlabel('Time (s)')
        cp.ylabel(cf.getNameUnits(self._data))
        cp.title(title)
        
        if self.nSystems > 1:
            plt.legend()

        return ax
    

    def plotPredicted(self, title='Time Domain EM Data', **kwargs):

        noLabels = kwargs.pop('nolabels', False)

        if (not noLabels):
            cp.xlabel('Time (s)')
            cp.ylabel(cf.getNameUnits(self._predictedData))
            cp.title(title)
        
        kwargs['color'] = kwargs.pop('color', cp.wellSeparated[3])
        kwargs['linewidth'] = kwargs.pop('linewidth', 2)
        kwargs['alpha'] = kwargs.pop('alpha', 0.7)
        xscale = kwargs.pop('xscale', 'log')
        yscale = kwargs.pop('yscale', 'log')
        for i in range(self.nSystems):
            iAct = self.iplotActive[i]

            p = self._predictedData[self._systemIndices(i)]
            p[iAct].plot(x=self.times(i)[iAct], **kwargs)

        plt.xscale(xscale)
        plt.yscale(yscale)


    def plotDataResidual(self, title='', **kwargs):

        
        lw = kwargs.pop('linewidth',2)
        a = kwargs.pop('alpha',0.7)

        xscale = kwargs.pop('xscale','linear')
        yscale = kwargs.pop('yscale','linear')

        for i in range(self.nSystems):
            iAct = self.iplotActive[i]
            dD = self.deltaD[self._systemIndices(i)]
            np.abs(dD[iAct]).plot(x=self.times(i)[iAct], **kwargs)

        plt.xscale(xscale)
        plt.yscale(yscale)

        plt.ylabel("|{}| ({})".format(dD.getName(), dD.getUnits()))

        cp.title(title)

    
    def priorProbability(self, rErr, aErr, height, calibration, verbose=False):
        """Evaluate the probability for the EM data point given the specified attached priors

        Parameters
        ----------
        rEerr : bool
            Include the relative error when evaluating the prior
        aEerr : bool
            Include the additive error when evaluating the prior
        height : bool
            Include the elevation when evaluating the prior
        calibration : bool
            Include the calibration parameters when evaluating the prior
        verbose : bool
            Return the components of the probability, i.e. the individually evaluated priors

        Returns
        -------
        out : np.float64
            The evaluation of the probability using all assigned priors

        Notes
        -----
        For each boolean, the associated prior must have been set.

        Raises
        ------
        TypeError
            If a prior has not been set on a requested parameter

        """
        probability = np.float64(0.0)
        errProbability = np.float64(0.0)

        P_relative = np.float64(0.0)
        P_additive = np.float64(0.0)
        P_height = np.float64(0.0)
        P_calibration = np.float64(0.0)

        if rErr:  # Relative Errors
            P_relative = self.relErr.probability()
            errProbability += P_relative
        if aErr:  # Additive Errors
            P_additive = self.addErr.probability(np.log(self.addErr))
            errProbability += P_additive

        probability += errProbability
        if height:  # Elevation
            P_height = (self.z.probability())
            probability += P_height
        if calibration:  # Calibration parameters
            P_calibration = self.calibration.probability()
            probability += P_calibration

        probability = np.float64(probability)
        
        if verbose:
            return probability, np.asarray([P_relative, P_additive, P_height, P_calibration])
        return probability


    def proposeAdditiveError(self):

        # Generate a new error
        tmp = self.addErr.proposal.rng(1)
        if self.addErr.hasPrior():
            p = self.addErr.probability(tmp)
            while p == 0.0 or p == -np.inf:
                tmp = self.addErr.proposal.rng(1)
                p = self.addErr.probability(tmp)
        # Update the mean of the proposed errors
        self.addErr.proposal.mean[:] = tmp
        self.addErr[:] = np.exp(tmp)


    def setAdditiveErrorPosterior(self):

        assert self.addErr.hasPrior(), Exception("Must set a prior on the additive error")

        aBins = self.addErr.prior.getBinEdges()

        log = 10
        aBins = np.exp(aBins)
        binsMidpoint = 0.5 * aBins.max(axis=-1) + aBins.min(axis=-1)

        ab = np.atleast_2d(aBins)
        binsMidpoint = np.atleast_1d(binsMidpoint)

        self.addErr.setPosterior([Histogram1D(bins = StatArray.StatArray(ab[i, :], name=self.addErr.name, units=self.data.units), log=log, relativeTo=binsMidpoint[i]) for i in range(self.nSystems)])



    def setAdditiveErrorPrior(self, minimum, maximum, prng=None):
        minimum = np.atleast_1d(minimum)
        assert minimum.size == self.nSystems, ValueError("minimum must have {} entries".format(self.nSystems))
        assert np.all(minimum > 0.0), ValueError("minimum values must be in linear space i.e. > 0.0")
        maximum = np.atleast_1d(maximum)
        assert maximum.size == self.nSystems, ValueError("maximum must have {} entries".format(self.nSystems))
        assert np.all(maximum > 0.0), ValueError("maximum values must be in linear space i.e. > 0.0")
        self.addErr.setPrior('UniformLog', np.log(minimum), np.log(maximum), prng=prng)

    
    def setAdditiveErrorProposal(self, means, variances, prng=None):
        means = np.atleast_1d(means)
        assert means.size == self.nSystems, ValueError("means must have {} entries".format(self.nSystems))
        variances = np.atleast_1d(variances)
        assert variances.size == self.nSystems, ValueError("variances must have {} entries".format(self.nSystems))
        self.addErr.setProposal('MvNormal', np.log(means), variances, prng=prng)

    # def scaleJ(self, Jin, power=1.0):
    #     """ Scales a matrix by the errors in the given data
    #     Useful if a sensitivity matrix is generated using one data point, but must be scaled by the errors in another """
    #     J1 = np.zeros(Jin.shape)
    #     J1[:, :] = Jin * (np.repeat(self._std[self.iActive, np.newaxis]**-power, np.size(J1, 1), 1))
    #     return J1


    def updateErrors(self, relativeErr, additiveErr):
        """ Updates the data errors

        Assumes a t^-0.5 behaviour e.g. logarithmic gate averaging
        V0 is assumed to be ln(Error @ 1ms)

        Parameters
        ----------  
        relativeErr : list of scalars or list of array_like
            A fraction percentage that is multiplied by the observed data. The list should have length equal to the number of systems. The entries in each item can be scalar or array_like.
        additiveErr : list of scalars or list of array_like
            An absolute value of additive error. The list should have length equal to the number of systems. The entries in each item can be scalar or array_like.

        Raises
        ------
        TypeError
            If relativeErr or additiveErr is not a list
        TypeError
            If the length of relativeErr or additiveErr is not equal to the number of systems
        TypeError
            If any item in the relativeErr or additiveErr lists is not a scalar or array_like of length equal to the number of time channels
        ValueError
            If any relative or additive errors are <= 0.0
        """
        relativeErr = np.atleast_1d(relativeErr)
        additiveErr = np.atleast_1d(additiveErr)

        #assert (isinstance(relativeErr, list)), TypeError("relativeErr must be a list of size equal to the number of systems {}".format(self.nSystems))
        assert (relativeErr.size == self.nSystems), TypeError("relativeErr must be a list of size equal to the number of systems {}".format(self.nSystems))

        #assert (isinstance(additiveErr, list)), TypeError("additiveErr must be a list of size equal to the number of systems {}".format(self.nSystems))
        assert (relativeErr.size == self.nSystems), TypeError("additiveErr must be a list of size equal to the number of systems {}".format(self.nSystems))

        t0 = 0.5 * np.log(1e-3)  # Assign fixed t0 at 1ms
        # For each system assign error levels using the user inputs
        for i in range(self.nSystems):
            assert (isinstance(relativeErr[i], float) or isinstance(relativeErr[i], np.ndarray)), TypeError("relativeErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.nTimes[i]))
            assert (isinstance(additiveErr[i], float) or isinstance(additiveErr[i], np.ndarray)), TypeError("additiveErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.nTimes[i]))
            assert (np.all(relativeErr[i] > 0.0)), ValueError("relativeErr for system {} cannot contain values <= 0.0.".format(i+1))
            assert (np.all(additiveErr[i] > 0.0)), ValueError("additiveErr for system {} should contain values > 0.0. Make sure the values are in linear space".format(i+1))
            iSys = self._systemIndices(system=i)
            
            # Compute the relative error
            rErr = relativeErr[i] * self._data[iSys]
            aErr = np.exp(np.log(additiveErr[i]) - 0.5 * np.log(self.times(i)) + t0)

            self._std[iSys] = np.sqrt((rErr**2.0) + (aErr**2.0))

        # Update the variance of the predicted data prior
        if self._predictedData.hasPrior():
            self._predictedData.prior.variance[:] = self._std[self.iActive]**2.0


    def updateSensitivity(self, J, mod, action, scale=False):
        """ Compute an updated sensitivity matrix using a new model based on an existing matrix """
        J1 = np.zeros([np.size(self.iActive), mod.nCells[0]])

        if(action == 'none'):  # Do Nothing!
            J1[:, :] = J[:, :]
            return J1

        perturbedLayer = mod.action[1]

        if (action == 'birth'):  # Created a layer
            J1[:, :perturbedLayer] = J[:, :perturbedLayer]
            J1[:, perturbedLayer + 2:] = J[:, perturbedLayer + 1:]
            tmp = self.sensitivity(mod, ix=[perturbedLayer, perturbedLayer + 1], scale=scale, modelChanged=True)
            J1[:, perturbedLayer:perturbedLayer + 2] = tmp
            return J1

        if(action == 'death'):  # Deleted a layer
            J1[:, :perturbedLayer] = J[:, :perturbedLayer]
            J1[:, perturbedLayer + 1:] = J[:, perturbedLayer + 2:]
            tmp = self.sensitivity(mod, ix=[perturbedLayer], scale=scale, modelChanged=True)
            J1[:, perturbedLayer] = tmp[:, 0]
            return J1

        if(action == 'perturb'):  # Perturbed a layer
            J1[:, :perturbedLayer] = J[:, :perturbedLayer]
            J1[:, perturbedLayer + 1:] = J[:, perturbedLayer + 1:]
            tmp = self.sensitivity(mod, ix=[perturbedLayer], scale=scale, modelChanged=True)
            J1[:, perturbedLayer] = tmp[:, 0]
            return J1

        assert False, __name__ + '.updateSensitivity: Invalid option [0,1,2]'


    def forward(self, mod):
        """ Forward model the data from the given model """

        assert isinstance(mod, Model), TypeError("Invalid model class for forward modeling [1D]")

        self._forward1D(mod)


    def sensitivity(self, mod, ix=None, scale=False, modelChanged=True):
        """ Compute the sensitivty matrix for the given model """

        assert isinstance(mod, Model), TypeError("Invalid model class for sensitivity matrix [1D]")

        return StatArray.StatArray(self._sensitivity1D(mod, ix, scale, modelChanged), 'Sensitivity', '$\\frac{V}{ASm^{3}}$')


    def _forward1D(self, mod):
        """ Forward model the data from a 1D layered earth model """
        heightTolerance = 0.0
        if (self.z > heightTolerance):
            self._BrodieForward(mod)
        else:
            self._simPEGForward(mod)


    def _BrodieForward(self, mod):
        # Generate the Brodie Earth class
        E = Earth(mod.par[:], mod.thk[:-1])
        # Generate the Brodie Geometry class
        G = Geometry(self.z[0], self.T.roll, self.T.pitch, self.T.yaw, 
                     self.loopOffset[0], self.loopOffset[1], self.loopOffset[2],
                     self.R.roll, self.R.pitch, self.R.yaw)

        # Forward model the data for each system
        for i in range(self.nSystems):
            iSys = self._systemIndices(i)
            fm = self.system[i].forwardmodel(G, E)
            self._predictedData[iSys] = -fm.SZ[:]  # Store the necessary component


    def _simPEGForward(self, mod):
        
        from SimPEG import Maps
        from simpegEM1D import (EM1DSurveyTD, EM1D, set_mesh_1d)

        mesh1D = set_mesh_1d(mod.depth)
        expmap = Maps.ExpMap(mesh1D)
        prob = EM1D(mesh1D, sigmaMap = expmap, chi = mod.chim)

        if (self.dualMoment()):

            print(self.system[0].loopRadius(), self.system[0].peakCurrent())

            simPEG_survey = EM1DSurveyTD(
                rx_location=np.array([0., 0., 0.]),
                src_location=np.array([0., 0., 0.]),
                topo=np.r_[0., 0., 0.],
                depth=-mod.depth,
                rx_type='dBzdt',
                wave_type='general',
                src_type='CircularLoop',
                a=self.system[0].loopRadius(),
                I=self.system[0].peakCurrent(),
                time=self.system[0].windows.centre,
                time_input_currents=self.system[0].waveform.transmitterTime,
                input_currents=self.system[0].waveform.transmitterCurrent,
                n_pulse=2,
                base_frequency=self.system[0].baseFrequency(),
                use_lowpass_filter=True,
                high_cut_frequency=450000,
                moment_type='dual',
                time_dual_moment=self.system[1].windows.centre,
                time_input_currents_dual_moment=self.system[1].waveform.transmitterTime,
                input_currents_dual_moment=self.system[1].waveform.transmitterCurrent,
                base_frequency_dual_moment=self.system[1].baseFrequency(),
            )
        else:

            simPEG_survey = EM1DSurveyTD(
                rx_location=np.array([0., 0., 0.]),
                src_location=np.array([0., 0., 0.]),
                topo=np.r_[0., 0., 0.],
                depth=-mod.depth,
                rx_type='dBzdt',
                wave_type='general',
                src_type='CircularLoop',
                a=self.system[0].loopRadius(),
                I=self.system[0].peakCurrent(),
                time=self.system[0].windows.centre,
                time_input_currents=self.system[0].waveform.transmitterTime,
                input_currents=self.system[0].waveform.transmitterCurrent,
                n_pulse=1,
                base_frequency=self.system[0].baseFrequency(),
                use_lowpass_filter=True,
                high_cut_frequency=7e4,
                moment_type='single',
            )

        prob.pair(simPEG_survey)
            
        self._predictedData[:] = -simPEG_survey.dpred(mod.par)


    def _sensitivity1D(self, mod, ix=None, scale=False, modelChanged=True):
        """ Compute the sensitivty matrix for a 1D layered earth model, optionally compute the responses for only the layers in ix """
        # Unfortunately the code requires forward modelled data to compute the
        # sensitivity if the model has changed since last time
        if modelChanged:
            E = Earth(mod.par[:], mod.thk[:-1])
            G = Geometry(self.z[0], self.T.roll, self.T.pitch, self.T.yaw, 
                         self.loopOffset[0], self.loopOffset[1], self.loopOffset[2],
                         self.R.roll, self.R.pitch, self.R.yaw)

            for i in range(self.nSystems):
                self.system[i].forwardmodel(G, E)

        if (ix is None):  # Generate a full matrix if the layers are not specified
            ix = range(mod.nCells[0])
            J = np.zeros([self.nWindows, mod.nCells[0]])
        else:  # Partial matrix for specified layers
            J = np.zeros([self.nWindows, len(ix)])

        for j in range(self.nSystems):  # For each system
            iSys = self._systemIndices(j)
            for i in range(len(ix)):  # For the specified layers
                tmp = self.system[j].derivative(
                    self.system[j].CONDUCTIVITYDERIVATIVE, ix[i] + 1)
                # Store the necessary component
                J[iSys, i] = -mod.par[ix[i]] * tmp.SZ[:]

        if scale:
            for j in range(self.nSystems):  # For each system
                iSys = self._systemIndices(j)
                for i in range(len(ix)):  # For the specified layers
                    # Scale the sensitivity matix rows by the data weights if
                    # required
                    J[iSys, i] /= self._std[ISYS]

        J = J[self.iActive, :]
        return J


    def Isend(self, dest, world, systems=None):
        tmp = np.empty(7, dtype=np.float64)
        tmp[:] = np.asarray([self.x, self.y, self.z, self.elevation, self.nSystems, self.lineNumber, self.fiducial])
        myMPI.Isend(tmp, dest=dest, world=world)
        if systems is None:
            for i in range(self.nSystems):
                world.isend(self.system[i].fileName, dest=dest)
        self._data.Isend(dest, world)
        self._std.Isend(dest, world)
        self._predictedData.Isend(dest, world)
        self.T.Isend(dest, world)
        self.R.Isend(dest, world)
        self.loopOffset.Isend(dest, world)



    def Irecv(self, source, world, systems=None):

        tmp = np.empty(7, dtype=np.float64)
        tmp = myMPI.Irecv(source=source, world=world)

        if systems is None:
            nSystems = np.int32(tmp[4])

            systems = []
            for i in range(nSystems):
                systems.append(world.irecv(source=source).wait())

        s = StatArray.StatArray(0)
        d = s.Irecv(source, world)
        s = s.Irecv(source, world)
        p = s.Irecv(source, world)
        c = CircularLoop()
        T = c.Irecv(source, world)
        R = c.Irecv(source, world)
        loopOffset = s.Irecv(source, world)

        return TdemDataPoint(tmp[0], tmp[1], tmp[2], tmp[3], data=d, std=s, predictedData=p, system=systems, T=T, R=R, loopOffset=loopOffset, lineNumber=tmp[5], fiducial=tmp[6])
