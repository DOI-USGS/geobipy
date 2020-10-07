from copy import deepcopy
from ....classes.core import StatArray
from ...model.Model import Model
from .EmDataPoint import EmDataPoint
from ...forwardmodelling.Electromagnetic.TD.tdem1d import (tdem1dfwd, tdem1dsen)
from ...system.EmLoop import EmLoop
from ...system.SquareLoop import SquareLoop
from ...system.CircularLoop import CircularLoop
from ....base.logging import myLogger
from ...system.TdemSystem import TdemSystem
from ...system.filters.butterworth import butterworth
from ...system.Waveform import Waveform
from ...statistics.Histogram1D import Histogram1D
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
                td = TdemSystem().read(sys)
                systems.append(td)
            elif isinstance(sys, TdemSystem):
                systems.append(sys)

            # Number of time gates
            nTimes[j] = systems[j].nTimes

        nChannels = np.sum(nTimes)

        if not data is None:
            if isinstance(data, list):
                assert len(data) == nSystems, ValueError("Must have {} arrays of data values".format(nSystems))
                data = np.hstack(data)
            assert data.size == nChannels, ValueError("Size of data must equal total number of time channels {}".format(nChannels))
            # Mask invalid data values less than 0.0 to NaN
            i = np.where(~np.isnan(data))[0]
            i1 = np.where(data[i] <= 0.0)[0]
            data[i[i1]] = np.nan
        if not std is None:
            if isinstance(std, list):
                assert len(std) == nSystems, ValueError("Must have {} arrays of std values".format(nSystems))
                std = np.hstack(std)
            assert std.size == nChannels, ValueError("Size of std must equal total number of time channels {}".format(nChannels))
        if not predictedData is None:
            if isinstance(predictedData, list):
                assert len(predictedData) == nSystems, ValueError("Must have {} arrays of predictedData values".format(nSystems))
                predictedData = np.hstack(predictedData)
            assert predictedData.size == nChannels, ValueError("Size of predictedData must equal total number of time channels {}".format(nChannels))

        super().__init__(nChannelsPerSystem=nTimes, x=x, y=y, z=z, elevation=elevation, data=data, std=std, predictedData=predictedData, dataUnits=r'$\frac{V}{Am^{4}}$', lineNumber=lineNumber, fiducial=fiducial)

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
                self._channelNames[k] = 'Time {:.3e} s'.format(self.system[i].times[iTime])
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
        out._relErr = self.relErr.deepcopy()
        # StatArray of Additive Errors
        out._addErr = self.addErr.deepcopy()
        out.errorPosterior = self.errorPosterior
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
        return len(self.system) == 2


    def read(self, dataFileName):
        """Read in a time domain data point from a file.

        Parameters
        ----------
        dataFileName : str or list of str
            File names of the data point.  Multiple can be given for multiple moments at the same location.

        Returns
        -------
        out : geobipy.TdemDataPoint
            Time domain data point

        """

        self._read_aarhus(dataFileName)


    def _read_aarhus(self, dataFileName):

        if isinstance(dataFileName, str):
            dataFileName = [dataFileName]

        system = []
        data = []
        std = []


        for fName in dataFileName:
            with open(fName, 'r') as f:
                # Header line
                dtype, x, y, z, elevation, fiducial, lineNumber, current = self.__aarhus_header(f)
                # Source type
                source, polarization = self.__aarhus_source(f)
                # Offset
                loopOffset = self.__aarhus_positions(f)
                # Loop Dimensions
                transmitterLoop, receiverLoop = self.__aarhus_loop_dimensions(f, source)
                # Data transforms
                transform = self.__aarhus_data_transforms(f)
                # Waveform
                time, amplitude = self.__aarhus_waveform(f)
                waveform = Waveform(time, amplitude, current)
                # Frontgate
                nPreFilters, frontGate, damping = self.__aarhus_frontgate(f)
                # Filter
                onTimeFilters = self.__aarhus_filters(f, nPreFilters)

                if frontGate:
                    # frontGate time
                    frontGateTime = np.float64(f.readline().strip())
                    offTimeFilters = self.__aarhus_filters(f, 1)

                # Data and standard deviation
                times, d, s = self.__aarhus_data(f)
                data.append(d)
                std.append(s*d)

                system.append(TdemSystem(offTimes=times,
                                         transmitterLoop=transmitterLoop,
                                         receiverLoop=receiverLoop,
                                         loopOffset=loopOffset,
                                         waveform=waveform,
                                         offTimeFilters=offTimeFilters))

        TdemDataPoint.__init__(self, x, y, 0.0, elevation, data, std, system=system, lineNumber=lineNumber, fiducial=fiducial)


    def __aarhus_header(self, f):
        line = f.readline().strip().split(';')
        dtype = x = y = z = elevation = current = None
        fiducial = lineNumber = 0.0
        for item in line:
            item = item.split("=")
            tag = item[0].lower()
            value = item[-1]

            if tag == "datatypestring":
                dtype = value
            elif tag == "xutm":
                x = np.float(value)
            elif tag == "yutm":
                y = np.float(value)
            elif tag == "elevation":
                elevation = np.float(value)
            elif tag == "stationnumber":
                fiducial = np.float(value)
            elif tag == "linenumber":
                lineNumber = np.float(value)
            elif tag == "current":
                current = np.float(value)

        assert not np.any([x, y, elevation, current] is None), ValueError("Aarhus file header line must contain 'XUTM', 'YUTM', 'Elevation', 'current'")

        return dtype, x, y, z, elevation, fiducial, lineNumber, current


    def __aarhus_source(self, f):
        line = f.readline().strip().split()
        source = np.int32(line[0])
        polarization = np.int32(line[1])

        assert source == 7, ValueError("Have only incorporated source == 7 so far.")
        assert polarization == 3, ValueError("Have only incorporated polarization == 3 so far.")

        return source, polarization


    def __aarhus_positions(self, f):
        line = f.readline().strip().split()
        tx, ty, tz, rx, ry, rz = [np.float(x) for x in line]
        return np.asarray([rx - tx, ry - ty, rz - tz]) # loopOffset


    def __aarhus_loop_dimensions(self, f, source):

        if source <= 6:
            return
        if source in [10, 11]:
            return

        line = f.readline().strip().split()
        if source == 7:
            dx, dy = [np.float(x) for x in line]
            assert dx == dy, ValueError("Only handling square loops at the moment")
            transmitter = SquareLoop(sideLength = dx)
            receiver = CircularLoop() # Dummy.
            return transmitter, receiver


    def __aarhus_data_transforms(self, f):
        line = f.readline().strip().split()
        a, b, c = [np.int32(x) for x in line]
        assert a == 3, ValueError("Can only handle data transform 3.  dB/dT")

        return a


    def __aarhus_waveform(self, f):
        line = f.readline().strip().split()
        typ, nWaveforms = [np.int32(x) for x in line]

        assert typ == 3, ValueError("Can only handle user defined waveforms, option 3")

        time = np.empty(0)
        amplitude = np.empty(0)
        for i in range(nWaveforms):
            line = f.readline().strip().split()
            tmp = np.asarray([np.float(x) for x in line[1:]])
            time = np.append(time, np.hstack([tmp[:2], tmp[5::4]]))
            amplitude = np.append(amplitude, np.hstack([tmp[2:4], tmp[6::5]]))

        return time, amplitude


    def __aarhus_frontgate(self, f):
        line = f.readline().strip().split()
        nFilters = np.int(line[0])
        frontGate = np.bool(np.int(line[1]))
        damping = np.float64(line[2])

        return nFilters, frontGate, damping


    def __aarhus_filters(self, f, nFilters):

        filters = []

        for i in range(nFilters):
            # Low Pass Filter
            line = f.readline().strip().split()
            nLowPass = np.int(line[0])
            for j in range(nLowPass):
                order = np.int(np.float(line[(2*j)+1]))
                frequency = np.float64(line[(2*j)+2])
                b = butterworth(order, frequency, btype='low', analog=True)
                filters.append(b)

            # High Pass Filter
            line = f.readline().strip().split()
            nHighPass = np.int(line[0])
            for j in range(nHighPass):
                order = np.int(np.floate(line[(2*j)+1]))
                frequency = np.float64(line[(2*j)+2])
                filters.append(butterworth(order, frequency, btype='high', analog=True))

        return filters


    def __aarhus_data(self, f):

        time = []
        data = []
        std = []
        while True:
            line = f.readline().strip().replace('%', '').split()
            if not line:
                break
            time.append(np.float64(line[0]))
            tmp = np.float64(line[1])
            data.append(np.nan if tmp == 999 else tmp)
            std.append(np.float64(line[2]))

        return np.asarray(time), np.asarray(data), np.asarray(std)


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

        if not self.errorPosterior is None:
            self.relErr.setPosterior([self.errorPosterior[i].marginalize(axis=1) for i in range(self.nSystems)])
            self.addErr.setPosterior([self.errorPosterior[i].marginalize(axis=0) for i in range(self.nSystems)])

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

        if not self.errorPosterior is None:
            self.relErr.setPosterior([self.errorPosterior[i].marginalize(axis=1) for i in range(self.nSystems)])
            self.addErr.setPosterior([self.errorPosterior[i].marginalize(axis=0) for i in range(self.nSystems)])

        self.relErr.writeHdf(grp, 'relErr',  withPosterior=withPosterior, index=index)
        self.addErr.writeHdf(grp, 'addErr',  withPosterior=withPosterior, index=index)
        self.T.writeHdf(grp, 'T', index=index)
        self.R.writeHdf(grp, 'R', index=index)
        self.loopOffset.writeHdf(grp, 'loop_offset', index=index)
        #writeNumpy(self.active, grp, 'iActive')

#    def toHdf(self, parent, myName):
#        """ Write the TdemDataPoint to an HDF object
#        h5obj: :An HDF File or Group Object.
#        """
#        self.writeHdf(parent, myName, index=np.s_[0])

    def fromHdf(self, grp, index=None, **kwargs):
        """ Reads the object from a HDF group """

        assert ('systemFilepath' in kwargs), ValueError("missing 1 required argument 'systemFilepath', the path to directory containing system files")

        systemFilepath = kwargs.pop('systemFilepath', None)
        assert (not systemFilepath is None), ValueError("missing 1 required argument 'systemFilepath', the path to directory containing system files")
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
            # filename = join(systemFilepath, str(np.asarray(grp.get('System{}'.format(i))), 'utf-8'))
            td = TdemSystem().read(systemFilepath[i])
            systems.append(td)

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
        _aPoint._relErr = obj.fromHdf(item, index=slic)
        item = grp.get('addErr')
        obj = eval(cf.safeEval(item.attrs.get('repr')))
        _aPoint._addErr = obj.fromHdf(item, index=slic)
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

        # _aPoint.active = _aPoint.getActiveData()

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
            iJ0 += self.system[j].nTimes


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

        for i in range(self.nSystems):
            iAct = self.iplotActive[i]
            dD = self.deltaD[self._systemIndices(i)]
            np.abs(dD[iAct]).plot(x=self.times(i)[iAct], **kwargs)

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

        probability += errProbability
        if height:  # Elevation
            P_height = (self.z.probability(log=True))
            probability += P_height

        if rErr:  # Relative Errors
            P_relative = self.relErr.probability(log=True)
            errProbability += P_relative

        if aErr:  # Additive Errors
            P_additive = self.addErr.probability(log=True)
            errProbability += P_additive

        if calibration:  # Calibration parameters
            P_calibration = self.calibration.probability(log=True)
            probability += P_calibration

        probability = np.float64(probability)

        if verbose:
            return probability, np.asarray([P_relative, P_additive, P_height, P_calibration])
        return probability


    def setPosteriors(self, log=10):
        return super().setPosteriors(log=10)


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
        assert (additiveErr.size == self.nSystems), TypeError("additiveErr must be a list of size equal to the number of systems {}".format(self.nSystems))

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
        if self._predictedData.hasPrior:
            self._predictedData.prior.variance[np.diag_indices(self.active.size)] = self._std[self.active]**2.0


    def updateSensitivity(self, mod):
        """ Compute an updated sensitivity matrix using a new model based on an existing matrix """

        J1 = np.zeros([np.size(self.active), mod.nCells[0]])

        perturbedLayer = mod.action[1]

        if(mod.action[0] == 'none'):  # Do Nothing!
            J1[:, :] = self.J[:, :]

        elif (mod.action[0] == 'birth'):  # Created a layer
            J1[:, :perturbedLayer] = self.J[:, :perturbedLayer]
            J1[:, perturbedLayer + 2:] = self.J[:, perturbedLayer + 1:]
            tmp = self.sensitivity(mod, ix=[perturbedLayer, perturbedLayer + 1], modelChanged=True)
            J1[:, perturbedLayer:perturbedLayer + 2] = tmp

        elif(mod.action[0] == 'death'):  # Deleted a layer
            J1[:, :perturbedLayer] = self.J[:, :perturbedLayer]
            J1[:, perturbedLayer + 1:] = self.J[:, perturbedLayer + 2:]
            tmp = self.sensitivity(mod, ix=[perturbedLayer], modelChanged=True)
            J1[:, perturbedLayer] = tmp[:, 0]

        elif(mod.action[0] == 'perturb'):  # Perturbed a layer
            J1[:, :perturbedLayer] = self.J[:, :perturbedLayer]
            J1[:, perturbedLayer + 1:] = self.J[:, perturbedLayer + 1:]
            tmp = self.sensitivity(mod, ix=[perturbedLayer], modelChanged=True)
            J1[:, perturbedLayer] = tmp[:, 0]

        self.J = J1


    def forward(self, mod):
        """ Forward model the data from the given model """

        assert isinstance(mod, Model), TypeError("Invalid model class for forward modeling [1D]")

        tdem1dfwd(self, mod)


    def sensitivity(self, model, ix=None, modelChanged=True):
        """ Compute the sensitivty matrix for the given model """

        assert isinstance(model, Model), TypeError("Invalid model class for sensitivity matrix [1D]")
        return StatArray.StatArray(tdem1dsen(self, model, ix, modelChanged), 'Sensitivity', '$\\frac{V}{SAm^{3}}$')


    def _empymodForward(self, mod):

        print('stuff')

    # def _simPEGForward(self, mod):

    #     from SimPEG import Maps
    #     from simpegEM1D import (EM1DSurveyTD, EM1D, set_mesh_1d)

    #     mesh1D = set_mesh_1d(mod.depth)
    #     expmap = Maps.ExpMap(mesh1D)
    #     prob = EM1D(mesh1D, sigmaMap = expmap, chi = mod.chim)

    #     if (self.dualMoment()):

    #         print(self.system[0].loopRadius(), self.system[0].peakCurrent())

    #         simPEG_survey = EM1DSurveyTD(
    #             rx_location=np.array([0., 0., 0.]),
    #             src_location=np.array([0., 0., 0.]),
    #             topo=np.r_[0., 0., 0.],
    #             depth=-mod.depth,
    #             rx_type='dBzdt',
    #             wave_type='general',
    #             src_type='CircularLoop',
    #             a=self.system[0].loopRadius(),
    #             I=self.system[0].peakCurrent(),
    #             time=self.system[0].windows.centre,
    #             time_input_currents=self.system[0].waveform.transmitterTime,
    #             input_currents=self.system[0].waveform.transmitterCurrent,
    #             n_pulse=2,
    #             base_frequency=self.system[0].baseFrequency(),
    #             use_lowpass_filter=True,
    #             high_cut_frequency=450000,
    #             moment_type='dual',
    #             time_dual_moment=self.system[1].windows.centre,
    #             time_input_currents_dual_moment=self.system[1].waveform.transmitterTime,
    #             input_currents_dual_moment=self.system[1].waveform.transmitterCurrent,
    #             base_frequency_dual_moment=self.system[1].baseFrequency(),
    #         )
    #     else:

    #         simPEG_survey = EM1DSurveyTD(
    #             rx_location=np.array([0., 0., 0.]),
    #             src_location=np.array([0., 0., 0.]),
    #             topo=np.r_[0., 0., 0.],
    #             depth=-mod.depth,
    #             rx_type='dBzdt',
    #             wave_type='general',
    #             src_type='CircularLoop',
    #             a=self.system[0].loopRadius(),
    #             I=self.system[0].peakCurrent(),
    #             time=self.system[0].windows.centre,
    #             time_input_currents=self.system[0].waveform.transmitterTime,
    #             input_currents=self.system[0].waveform.transmitterCurrent,
    #             n_pulse=1,
    #             base_frequency=self.system[0].baseFrequency(),
    #             use_lowpass_filter=True,
    #             high_cut_frequency=7e4,
    #             moment_type='single',
    #         )

    #     prob.pair(simPEG_survey)

    #     self._predictedData[:] = -simPEG_survey.dpred(mod.par)


    def Isend(self, dest, world, systems=None):
        tmp = np.asarray([self.x, self.y, self.z, self.elevation, self.nSystems, self.lineNumber, self.fiducial, *self.loopOffset], dtype=np.float64)
        myMPI.Isend(tmp, dest=dest, ndim=1, shape=(10, ), dtype=np.float64, world=world)

        if systems is None:
            for i in range(self.nSystems):
                world.send(self.system[i].fileName, dest=dest)

        self._data.Isend(dest, world)
        self._std.Isend(dest, world)
        self._predictedData.Isend(dest, world)
        self.T.Isend(dest, world)
        self.R.Isend(dest, world)



    def Irecv(self, source, world, systems=None):

        tmp = myMPI.Irecv(source=source, ndim=1, shape=(10, ), dtype=np.float64, world=world)

        if systems is None:
            nSystems = np.int32(tmp[4])

            systems = []
            for i in range(nSystems):
                sys = world.recv(source=source)
                systems.append(sys)

        s = StatArray.StatArray(0)
        d = s.Irecv(source, world)
        s = s.Irecv(source, world)
        p = s.Irecv(source, world)
        c = CircularLoop()
        T = c.Irecv(source, world)
        R = c.Irecv(source, world)
        loopOffset  = tmp[-3:]
        return TdemDataPoint(tmp[0], tmp[1], tmp[2], tmp[3], data=d, std=s, predictedData=p, system=systems, T=T, R=R, loopOffset=loopOffset, lineNumber=tmp[5], fiducial=tmp[6])
