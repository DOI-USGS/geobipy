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
from ....base import MPI as myMPI
from ....base import customPlots as cp


class FdemDataPoint(EmDataPoint):
    """Class defines a Frequency domain electromagnetic data point.

    Contains an easting, northing, height, elevation, observed and predicted data, and uncertainty estimates for the data.

    FdemDataPoint(x, y, z, elevation, data, std, system, lineNumber, fiducial)

    Parameters
    ----------
    x : float
        Easting co-ordinate of the data point
    y : float
        Northing co-ordinate of the data point
    z : float
        Height above ground of the data point
    elevation : float, optional
        Elevation from sea level of the data point
    data : geobipy.StatArray or array_like, optional
        Data values to assign the data of length 2*number of frequencies.
        * If None, initialized with zeros.
    std : geobipy.StatArray or array_like, optional
        Estimated uncertainty standard deviation of the data of length 2*number of frequencies.
        * If None, initialized with ones if data is None, else 0.1*data values.
    system : str or geobipy.FdemSystem, optional
        Describes the acquisition system with loop orientation and frequencies.
        * If str should be the path to a system file to read in.
        * If geobipy.FdemSystem, will be deepcopied.
    lineNumber : float, optional
        The line number associated with the datapoint
    fiducial : float, optional
        The fiducial associated with the datapoint

    """

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=0.0, data=None, std=None, predictedData=None, system=None, lineNumber=0.0, fiducial=0.0):
        """Define initializer. """
        if (system is None):
            return
        else:
            if isinstance(system, (str, FdemSystem)):
                system = [system]
            assert all((isinstance(sys, (str, FdemSystem)) for sys in system)), TypeError("System must have items of type str or FdemSystem")

        # Assign the number of systems as 1
        nSystems = len(system)
        nFrequencies = np.empty(nSystems, dtype=np.int32)

        systems = []
        for j, sys in enumerate(system):
            # EMSystem Class
            if (isinstance(sys, str)):
                tmpsys = FdemSystem()
                tmpsys.read(sys)
                systems.append(tmpsys)
            elif (isinstance(sys, FdemSystem)):
                systems.append(sys)
            nFrequencies[j] = systems[j].nFrequencies

        nChannels = np.sum(2*nFrequencies)

        if not data is None:
            assert np.size(data) == nChannels, ValueError("Size of data {}, must equal 2 * total number of frequencies {}".format(np.size(data), nChannels))
        if not std is None:
            assert np.size(std) == nChannels, ValueError("Size of std {}, must equal 2 * total number of frequencies {}".format(np.size(std), nChannels))
        if not predictedData is None:
            assert np.size(predictedData) == nChannels, ValueError("Size of predictedData {}, must equal 2 * total number of frequencies {}".format(np.size(predictedData), nChannels))
    
        EmDataPoint.__init__(self, nChannelsPerSystem=2*nFrequencies, x=x, y=y, z=z, elevation=elevation, data=data, std=std, predictedData=predictedData, dataUnits="ppm", lineNumber=lineNumber, fiducial=fiducial)

        self._data.name = 'Frequency domain data'

        self.nSystems = nSystems
        self.system = systems

        # StatArray of calibration parameters
        # The four columns are Bias,Variance,InphaseBias,QuadratureBias.
        self.calibration = StatArray([self.nChannels * 2], 'Calibration Parameters')

        k = 0
        for i in range(self.nSystems):
            # Set the channel names
            for iFrequency in range(self.nChannelsPerSystem[i]):
                self._channelNames[k] = '{} {} (Hz)'.format(self.getMeasurementType(iFrequency, i), self.getFrequency(iFrequency, i))
                k += 1


    def _inphaseIndices(self, system=0):
        """The slice indices for the requested in-phase data.
        
        Parameters
        ----------
        system : int
            Requested system index.
            
        Returns
        -------
        out : numpy.slice
            The slice pertaining to the requested system.
            
        """

        assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))

        return np.s_[self._systemOffset[system]:self._systemOffset[system] + self.nFrequencies[system]]


    def _quadratureIndices(self, system=0):
        """The slice indices for the requested in-phase data.
        
        Parameters
        ----------
        system : int
            Requested system index.
            
        Returns
        -------
        out : numpy.slice
            The slice pertaining to the requested system.
            
        """

        assert system < self.nSystems, ValueError("system must be < nSystems {}".format(self.nSystems))

        return np.s_[self._systemOffset[system] + self.nFrequencies[system]: self._systemOffset[system+1]]


    @property
    def data(self):
        return self._data


    def frequencies(self, system=0):
        """ Return the frequencies in an StatArray """
        return StatArray(self.system[system].frequencies, name='Frequency', units='Hz')

    
    def inphase(self, system=0):
        return self._data[self._inphaseIndices(system)]

    
    def inphaseStd(self, system=0):
        return self._std[self._inphaseIndices(system)]

    # @property
    # def nChannels(self):
    #     return np.sum(2*self.nFrequencies)

    @property
    def nFrequencies(self):
        return np.int32(0.5*self.nChannelsPerSystem)

    # @property
    # def nTotalFrequencies(self):
    #     return np.int32(np.sum(self.nFrequencies))

    @property
    def predictedData(self):
        return self._predictedData

    
    def predictedInphase(self, system=0):
        return self._predictedData[self._inphaseIndices(system)]

    
    def predictedQuadrature(self, system=0):
        return self._predictedData[self._quadratureIndices(system)]

    
    def quadrature(self, system=0):
        return self._data[self._quadratureIndices(system)]

    
    def quadratureStd(self, system=0):
        return self._std[self._quadratureIndices(system)]

    @property
    def std(self):
        return self._std


    def deepcopy(self):
        return self.__deepcopy__()

    
    def __deepcopy__(self):
        """ Define a deepcopy routine """
        tmp = FdemDataPoint(self.x, self.y, self.z, self.elevation, self._data, self._std, self._predictedData, self.system, self.lineNumber, self.fiducial)
        # StatArray of Relative Errors
        tmp.relErr = self.relErr.deepcopy()
        # StatArray of Additive Errors
        tmp.addErr = self.addErr.deepcopy()
        # StatArray of calibration parameters
        # The four columns are Bias,Variance,InphaseBias,QuadratureBias.
        tmp.calibration = self.calibration.deepcopy()
        # Initialize the sensitivity matrix
        tmp.J = deepcopy(self.J)

        return tmp

    
    def getMeasurementType(self, channel, system=0):
        """Returns the measurement type of the channel

        Parameters
        ----------
        channel : int
            Channel number
        system : int, optional
            System number

        Returns
        -------
        out : str
            Either "In-Phase " or "Quadrature "
        
        """
        return 'In-Phase' if channel < self.nFrequencies[system] else 'Quadrature'


    def getFrequency(self, channel, system=0):
        """Return the measurement frequency of the channel

        Parameters
        ----------
        channel : int
            Channel number
        system : int, optional
            System number

        Returns
        -------
        out : float
            The measurement frequency of the channel

        """
        return self.system[system].frequencies[channel%self.nFrequencies[system]]


    def hdfName(self):
        """ Reproducibility procedure """
        return('FdemDataPoint()')


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
        self.elevation.createHdf(grp, 'e', nRepeats=nRepeats, fillvalue=fillvalue)
        self._data.createHdf(grp, 'd', nRepeats=nRepeats, fillvalue=fillvalue)
        self._std.createHdf(grp, 's', nRepeats=nRepeats, fillvalue=fillvalue)
        self._predictedData.createHdf(grp, 'p', nRepeats=nRepeats, fillvalue=fillvalue)
        self.relErr.createHdf(grp, 'relErr', nRepeats=nRepeats, fillvalue=fillvalue)
        self.addErr.createHdf(grp, 'addErr', nRepeats=nRepeats, fillvalue=fillvalue)
        self.calibration.createHdf(grp, 'calibration', nRepeats=nRepeats, fillvalue=fillvalue)
        self.system[0].toHdf(grp, 'sys')


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
        self.elevation.writeHdf(grp, 'e',  index=index)

        self._data.writeHdf(grp, 'd',  index=index)
        self._std.writeHdf(grp, 's',  index=index)
        self._predictedData.writeHdf(grp, 'p',  index=index)
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

        item = grp.get('sys')
        obj = eval(safeEval(item.attrs.get('repr')))
        system = obj.fromHdf(item)

        _aPoint = FdemDataPoint(x, y, z, e, system=system)

        slic = None
        if not index is None:
            slic=np.s_[index,:]
        item = grp.get('d')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint._data = obj.fromHdf(item, index=slic)

        item = grp.get('s')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint._std = obj.fromHdf(item, index=slic)

        item = grp.get('p')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint._predictedData = obj.fromHdf(item, index=slic)

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
            tmp = cf.mergeComplex(self._predictedData)
        else:
            tmp = cf.mergeComplex(self._data)

        # Get the calibration factors for each frequency
        i1 = 0
        i2 = self.nFrequencies
        G = self.calibration[i1:i2]
        i1 += self.nFrequencies
        i2 += self.nFrequencies
        Phi = self.calibration[i1:i2]
        i1 += self.nFrequencies
        i2 += self.nFrequencies
        Bi = self.calibration[i1:i2]
        i1 += self.nFrequencies
        i2 += self.nFrequencies
        Bq = self.calibration[i1:i2]

        # Calibrate the data
        tmp[:] = G * np.exp(1j * Phi) * tmp + Bi + (1j * Bq)

        # Split the complex numbers back out
        if (Predicted):
            self._predictedData[:] = cf.splitComplex(tmp)
        else:
            self._data[:] = cf.splitComplex(tmp)


    def plot(self, title='Frequency Domain EM Data', system=0,  **kwargs):
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
        cp.ylabel('Frequency domain data (ppm)')
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

        plt.errorbar(self.frequencies(system), self.inphase(system), yerr=self.inphaseStd(system),
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

        plt.errorbar(self.frequencies(system), self.quadrature(system), yerr=self.quadratureStd(system),
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


    def plotPredicted(self, title='Frequency Domain EM Data', system=0, **kwargs):

        ax = plt.gca()
        cp.pretty(ax)

        noLabels = kwargs.pop('nolabels', False)

        if (not noLabels):
            cp.xlabel('Frequency (Hz)')
            cp.ylabel('Data (ppm)')
            cp.title(title)

        c = kwargs.pop('color',cp.wellSeparated[3])
        lw = kwargs.pop('linewidth',2)
        a = kwargs.pop('alpha',0.7)

        xscale = kwargs.pop('xscale','log')
        yscale = kwargs.pop('yscale','log')

        plt.semilogx(self.frequencies(system), self.predictedInphase(system), color=c, linewidth=lw, alpha=a, **kwargs)
        plt.semilogx(self.frequencies(system), self.predictedQuadrature(system), color=c, linewidth=lw, alpha=a, **kwargs)

        plt.xscale(xscale)
        plt.yscale(yscale)

        return ax


    # def scaleJ(self, Jin, power=1.0):
    #     """ Scales a matrix by the errors in the given data
    #     Useful if a sensitivity matrix is generated using one data point, but must be scaled by the errors in another """
    #     J1 = np.zeros(Jin.shape)
    #     J1[:, :] = Jin * (np.repeat(self.s[self.iActive, np.newaxis] ** -power, np.size(J1, 1), 1))
    #     return J1


    def updateSensitivity(self, J, mod, option, scale=False):
        """ Compute an updated sensitivity matrix based on the one already containined in the FdemDataPoint object  """
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
        for i, s in enumerate(self.system):
            tmp = fdem1dfwd(s, mod, -self.z[0])
            self._predictedData[:self.nFrequencies[i]] = tmp.real
            self._predictedData[self.nFrequencies[i]:] = tmp.imag


    def _sensitivity1D(self, mod, scale=False):
        """ Compute the sensitivty matrix for a 1D layered earth model """
        # Re-arrange the sensitivity matrix to Real:Imaginary vertical
        # concatenation
        J = np.zeros([self.nChannels, mod.nCells[0]])

        for j, s in enumerate(self.system):
            Jtmp = fdem1dsen(s, mod, -self.z[0])
            J[:self.nFrequencies[j], :] = Jtmp.real
            J[self.nFrequencies[j]:, :] = Jtmp.imag

        # Scale the sensitivity matrix rows by the data weights if required
        if scale:
            J *= (np.repeat(self._std[:, np.newaxis]**-1.0, np.size(J, 1), 1))

        J = J[self.iActive, :]
        return J

    
    def Isend(self, dest, world, systems=None):
        tmp = np.empty(7, dtype=np.float64)
        tmp[:] = np.asarray([self.x, self.y, self.z, self.elevation, self.nSystems, self.lineNumber, self.fiducial])
        myMPI.Isend(tmp, dest=dest, world=world)

        if systems is None:
            for i in range(self.nSystems):
                self.system[i].Isend(dest=dest, world=world)
        self._data.Isend(dest, world)
        self._std.Isend(dest, world)
        self._predictedData.Isend(dest, world)


    def Irecv(self, source, world, systems=None):

        tmp = myMPI.Irecv(source=source, world=world)

        if systems is None:
            nSystems = np.int32(tmp[4])

            systems = []
            fs = FdemSystem()
            for i in range(nSystems):
                systems.append(fs.Irecv(source=source, world=world))

        s = StatArray(0)
        d = s.Irecv(source, world)
        s = s.Irecv(source, world)
        p = s.Irecv(source, world)

        return FdemDataPoint(tmp[0], tmp[1], tmp[2], tmp[3], data=d, std=s, predictedData=p, system=systems, lineNumber=tmp[5], fiducial=tmp[6])
       

