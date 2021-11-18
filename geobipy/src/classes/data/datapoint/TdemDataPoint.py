from copy import deepcopy
from abc import abstractmethod
from ....classes.core import StatArray
from ...model.Model1D import Model1D
from .EmDataPoint import EmDataPoint
from ...forwardmodelling.Electromagnetic.TD.tdem1d import (
    tdem1dfwd, tdem1dsen)
from ...system.EmLoop import EmLoop
from ...system.SquareLoop import SquareLoop
from ...system.CircularLoop import CircularLoop
from ....base.logging import myLogger
from ...system.TdemSystem import TdemSystem
from ...system.TdemSystem_GAAEM import TdemSystem_GAAEM
from ...system.filters.butterworth import butterworth
from ...system.Waveform import Waveform
from ...statistics.Histogram1D import Histogram1D
from ...statistics.Distribution import Distribution
import matplotlib.pyplot as plt
import numpy as np

#from ....base import Error as Err
from ....base import fileIO as fIO
from ....base import utilities as cf
from ....base import plotting as cp
from ....base import MPI as myMPI
from os.path import split as psplt
from os.path import join


class TdemDataPoint(EmDataPoint):
    """ Initialize a Time domain EMData Point


    TdemDataPoint(x, y, z, elevation, data, std, system, transmitter_loop, receiver_loop, lineNumber, fiducial)

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
    transmitter_loop : EmLoop, optional
        Transmitter loop class
    receiver_loop : EmLoop, optional
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

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=0.0,
                 data=None, std=None, predictedData=None,
                 system=None,
                 transmitter_loop=None, receiver_loop=None,
                 lineNumber=0.0, fiducial=0.0):

        self.system = system

        super().__init__(x=x, y=y, z=z, elevation=elevation,
                         components=self.components,
                         channels_per_system=self.nTimes,
                         data=data, std=std, predictedData=predictedData,
                         lineNumber=lineNumber, fiducial=fiducial)

        self.transmitter = transmitter_loop
        self.receiver = receiver_loop

        self.channelNames = None

    @EmDataPoint.addErr.setter
    def addErr(self, values):
        if values is None:
            values = self.nSystems
        else:
            assert np.size(values) == self.nSystems, ValueError("additiveError must be a list of size equal to the number of systems {}".format(self.nSystems))
            assert (np.all(np.asarray(values) > 0.0)), ValueError("additiveErr must be > 0.0. Make sure the values are in linear space")
            # assert (isinstance(relativeErr[i], float) or isinstance(relativeErr[i], np.ndarray)), TypeError(
            #     "relativeErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.nTimes[i]))

        self._addErr = StatArray.StatArray(values, '$\epsilon_{Additive}$', self.units)

    # @property
    # def components(self):
    #     return self.system[0].components

    @EmDataPoint.channelNames.setter
    def channelNames(self, values):
        if values is None:
            if self.system is None:
                self._channelNames = ['None']
                return


            self._channelNames = []

            for component in self.components:
                for i in range(self.nSystems):
                    for t in self.off_time(i):
                        self._channelNames.append('Time {:.3e} s {}'.format(t, component))
        else:
            assert all((isinstance(x, str) for x in values))
            assert len(values) == self.nChannels, Exception("Length of channelNames must equal total number of channels {}".format(self.nChannels))
            self._channelNames = values

    @property
    def channels(self):
        return np.asarray([self.off_time(i) for i in range(self.nSystems)])

    @property
    def loopOffset(self):
        diff = self.receiver - self.transmitter
        return np.r_[diff.x, diff.y, diff.z]

    @property
    def receiver(self):
        return self._receiver

    @receiver.setter
    def receiver(self, value):
        if not value is None:
            assert isinstance(value, EmLoop), TypeError(
                "receiver must be of type EmLoop")
            self._receiver = value

    @EmDataPoint.system.setter
    def system(self, value):

        if value is None:
            self._system = None
            self.components = None
            self.channels_per_system = None
            return

        if isinstance(value, (str, TdemSystem)):
            value = [value]
        assert all((isinstance(sys, (str, TdemSystem_GAAEM)) for sys in value)), TypeError(
            "System must be list with items of type TdemSystem")

        self._system = []
        for j, sys in enumerate(value):
            if isinstance(sys, str):
                self._system.append(TdemSystem.read(sys))
            else:
                self._system.append(sys)

        self.components = self.system[0].components
        self.channels_per_system = self.n_components * np.r_[[x.nTimes for x in self.system]]

    @property
    def transmitter(self):
        return self._transmitter

    @transmitter.setter
    def transmitter(self, value):
        if not value is None:
            assert isinstance(value, EmLoop), TypeError("transmitter must be of type EmLoop")
            self._transmitter = value

    @property
    def n_components(self):
        return np.size(self.components)

    @property
    def nTimes(self):
        return (self.channels_per_system / self.n_components).astype(np.int32)

    @property
    def nWindows(self):
        return self.nChannels

    @EmDataPoint.units.setter
    def units(self, value):
        if value is None:
            value = r"$\frac{V}{m^{2}}$"
        else:
            assert isinstance(value, str), TypeError(
                'units must have type str')
        self._units = value

    def off_time(self, system=0):
        """ Return the window times in an StatArray """
        return self.system[system].off_time

    @property
    def _ravel_index(self):
        return np.cumsum(np.hstack([0, np.repeat(self.nTimes, self.n_components)]))

    def _component_indices(self, component=0, system=0):
        i = np.ravel_multi_index((component, system), (self.n_components, self.nSystems))
        return np.s_[self._ravel_index[i]:self._ravel_index[i+1]]

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out.system = self._system
        out._transmitter = deepcopy(self.transmitter)
        out._receiver = deepcopy(self.receiver)

        return out

    @EmDataPoint.std.getter
    def std(self):
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

        assert np.all(self.relErr > 0.0), ValueError('relErr must be > 0.0')

        t0 = 0.5 * np.log(1e-3)  # Assign fixed t0 at 1ms
        # For each system assign error levels using the user inputs
        for i in range(self.nSystems):
            iSys = self._systemIndices(system=i)

            # Compute the relative error
            rErr = self.relErr[i] * self.data[iSys]
            aErr = np.exp(
                np.log(self.addErr[i]) - 0.5 * np.log(self.off_time(i)) + t0)

            self._std[iSys] = np.sqrt((rErr**2.0) + (aErr**2.0))

        # Update the variance of the predicted data prior
        if self.predictedData.hasPrior:
            self.predictedData.prior.variance[np.diag_indices(np.sum(self.active))] = self._std[self.active]**2.0

        return self._std

    @property
    def system_indices(self):
        tmp = np.hstack([0, np.cumsum(self.channels_per_system)])
        return [np.s_[tmp[i]:tmp[i+1]] for i in range(self.nSystems)]

    @property
    def iplotActive(self):
        """ Get the active data indices per system.  Used for plotting. """
        return [cf.findNotNans(self.data[self.system_indices[i]]) for i in range(self.nSystems)]
        # self.iplotActive = []
        # i0 = 0
        # for i in range(self.nSystems):
        #     i1 = i0 + self.nTimes[i]
        #     self.iplotActive.append(cf.findNotNans(self._data[i0:i1]))
        #     i0 = i1

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
        data = np.empty(0)
        std = np.empty(0)

        for fName in dataFileName:
            with open(fName, 'r') as f:
                # Header line
                dtype, x, y, z, elevation, fiducial, lineNumber, current = self.__aarhus_header(
                    f)
                # Source type
                source, polarization = self.__aarhus_source(f)
                # Offset
                loopOffset = self.__aarhus_positions(f)
                # Loop Dimensions
                transmitterLoop, receiverLoop = self.__aarhus_loop_dimensions(
                    f, source)
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
                data = np.hstack([data, d])
                std = np.hstack([std, s*d])

                system.append(TdemSystem(offTimes=times,
                                         transmitterLoop=transmitterLoop,
                                         receiverLoop=receiverLoop,
                                         waveform=waveform,
                                         offTimeFilters=offTimeFilters))

        TdemDataPoint.__init__(self, x, y, 0.0, elevation, data, std,
                               system=system, lineNumber=lineNumber, fiducial=fiducial)

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

        assert not np.any([x, y, elevation, current] is None), ValueError(
            "Aarhus file header line must contain 'XUTM', 'YUTM', 'Elevation', 'current'")

        return dtype, x, y, z, elevation, fiducial, lineNumber, current

    def __aarhus_source(self, f):
        line = f.readline().strip().split()
        source = np.int32(line[0])
        polarization = np.int32(line[1])

        assert source == 7, ValueError(
            "Have only incorporated source == 7 so far.")
        assert polarization == 3, ValueError(
            "Have only incorporated polarization == 3 so far.")

        return source, polarization

    def __aarhus_positions(self, f):
        line = f.readline().strip().split()
        tx, ty, tz, rx, ry, rz = [np.float(x) for x in line]
        return np.asarray([rx - tx, ry - ty, rz - tz])  # loopOffset

    def __aarhus_loop_dimensions(self, f, source):

        if source <= 6:
            return
        if source in [10, 11]:
            return

        line = f.readline().strip().split()
        if source == 7:
            dx, dy = [np.float(x) for x in line]
            assert dx == dy, ValueError(
                "Only handling square loops at the moment")
            transmitter = SquareLoop(sideLength=dx)
            receiver = CircularLoop()  # Dummy.
            return transmitter, receiver

    def __aarhus_data_transforms(self, f):
        line = f.readline().strip().split()
        a, b, c = [np.int32(x) for x in line]
        assert a == 3, ValueError("Can only handle data transform 3.  dB/dT")

        return a

    def __aarhus_waveform(self, f):
        line = f.readline().strip().split()
        typ, nWaveforms = [np.int32(x) for x in line]

        assert typ == 3, ValueError(
            "Can only handle user defined waveforms, option 3")

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
                filters.append(butterworth(
                    order, frequency, btype='high', analog=True))

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

    def createHdf(self, parent, name, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """

        grp = super().createHdf(parent, name, withPosterior, nRepeats, fillvalue)

        grp.create_dataset('nSystems', data=self.nSystems)
        for i in range(self.nSystems):
            grp.create_dataset('System{}'.format(i), data=np.string_(psplt(self.system[i].filename)[-1]))

        self.transmitter.createHdf(grp, 'T', nRepeats=nRepeats, fillvalue=fillvalue)
        self.receiver.createHdf(grp, 'R', nRepeats=nRepeats, fillvalue=fillvalue)

        return grp

    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        super().writeHdf(parent, name, withPosterior, index)

        grp = parent[name]

        self.transmitter.writeHdf(grp, 'T', index=index)
        self.receiver.writeHdf(grp, 'R', index=index)

    @classmethod
    def fromHdf(cls, grp, index=None, **kwargs):
        """ Reads the object from a HDF group """

        assert ('system_file_path' in kwargs), ValueError(
            "missing 1 required argument 'system_file_path', the path to directory containing system files")

        system_file_path = kwargs['system_file_path']

        self = super(TdemDataPoint, cls).fromHdf(grp, index)

        self.transmitter = (eval(cf.safeEval(grp['T'].attrs.get('repr')))).fromHdf(grp['T'], index=index)
        self.receiver = (eval(cf.safeEval(grp['R'].attrs.get('repr')))).fromHdf(grp['R'], index=index)

        if 'loop_offset' in grp:
            loopOffset = StatArray.StatArray.fromHdf(grp['loop_offset'], index=index)
            self.receiver.x = self.transmitter.x + loopOffset[0]
            self.receiver.y = self.transmitter.y + loopOffset[1]
            self.receiver.z = self.transmitter.z + loopOffset[2]

        nSystems = np.int(np.asarray(grp['nSystems']))

        lst = [join(system_file_path, str(np.asarray(grp['System{}'.format(i)]), 'utf-8')) for i in range(nSystems)]

        self.system = lst

        # self._channels_per_system = self.nTimes

        return self

    def plotWaveform(self, **kwargs):
        for i in range(self.nSystems):
            if (self.nSystems > 1):
                plt.subplot(2, 1, i + 1)
            plt.plot(self.system[i].waveform.time,
                     self.system[i].waveform.current, **kwargs)
            cp.xlabel('Time (s)')
            cp.ylabel('Normalized Current (A)')
            plt.margins(0.1, 0.1)

    def plot(self, title='Time Domain EM Data', with_error_bars=True, **kwargs):
        """ Plot the Inphase and Quadrature Data for an EM measurement
        """
        ax = kwargs.pop('ax', None)
        ax = plt.gca() if ax is None else plt.sca(ax)
        plt.cla()

        kwargs['marker'] = kwargs.pop('marker', 'v')
        kwargs['markersize'] = kwargs.pop('markersize', 7)
        c = kwargs.pop('color', [cp.wellSeparated[i+1]
                       for i in range(self.nSystems)])
        mfc = kwargs.pop('markerfacecolor', [
                         cp.wellSeparated[i+1] for i in range(self.nSystems)])
        assert len(c) == self.nSystems, ValueError(
            "color must be a list of length {}".format(self.nSystems))
        assert len(mfc) == self.nSystems, ValueError(
            "markerfacecolor must be a list of length {}".format(self.nSystems))
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor', 'k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth', 1.0)
        kwargs['alpha'] = kwargs.pop('alpha', 0.8)
        kwargs['linestyle'] = kwargs.pop('linestyle', 'none')
        kwargs['linewidth'] = kwargs.pop('linewidth', 2)

        xscale = kwargs.pop('xscale', 'log')
        yscale = kwargs.pop('yscale', 'log')

        logx = kwargs.pop('logX', None)
        logy = kwargs.pop('logY', None)

        for j in range(self.nSystems):
            system_times, _ = cf._log(self.off_time(j), logx)

            for k in range(self.n_components):

                icomp = self._component_indices(k, j)
                # component_times = system_times[icomp]
                d = self.data[icomp]

                if (with_error_bars):
                    s = self._std[icomp]
                    plt.errorbar(system_times, d, yerr=s,
                                 color=c[j],
                                 markerfacecolor=mfc[j],
                                 label='System: {}'.format(j+1),
                                 **kwargs)
                else:
                    plt.plot(system_times, d,
                             markerfacecolor=mfc[j],
                             label='System: {}'.format(j+1),
                             **kwargs)

        plt.xscale(xscale)
        plt.yscale(yscale)
        cp.xlabel('Time (s)')
        cp.ylabel(cf.getNameUnits(self.data))
        cp.title(title)

        if self.nSystems > 1:
            plt.legend()

        return ax

    # def plot_posteriors(self, axes=None, height_kwargs={}, data_kwargs={}, rel_error_kwargs={}, add_error_kwargs={}, **kwargs):
    #     super().plot_posteriors(axes=axes,
    #                             height_kwargs=height_kwargs,
    #                             data_kwargs=data_kwargs,
    #                             rel_error_kwargs=rel_error_kwargs,
    #                             add_error_kwargs=add_error_kwargs,
    #                             **kwargs)

    def plotPredicted(self, title='Time Domain EM Data', **kwargs):

        ax = kwargs.pop('ax', None)
        ax = plt.gca() if ax is None else plt.sca(ax)

        noLabels = kwargs.pop('nolabels', False)

        if (not noLabels):
            cp.xlabel('Time (s)')
            cp.ylabel(cf.getNameUnits(self.predictedData))
            cp.title(title)

        kwargs['color'] = kwargs.pop('color', cp.wellSeparated[3])
        kwargs['linewidth'] = kwargs.pop('linewidth', 2)
        kwargs['alpha'] = kwargs.pop('alpha', 0.7)
        xscale = kwargs.pop('xscale', 'log')
        yscale = kwargs.pop('yscale', 'log')

        logx = kwargs.pop('logX', None)
        logy = kwargs.pop('logY', None)

        for j in range(self.nSystems):
            system_times, _ = cf._log(self.off_time(j), logx)

            for k in range(self.n_components):
                iS = self._component_indices(k, j)

                if np.all(self.data <= 0.0):
                    active = (self.predictedData[iS] > 0.0)

                else:
                    active = self.active[iS]

                p = self.predictedData[iS][active]
                p.plot(x=system_times[active], **kwargs)

        plt.xscale(xscale)
        plt.yscale(yscale)

    def plotDataResidual(self, title='', **kwargs):

        ax = kwargs.pop('ax', None)
        ax = plt.gca() if ax is None else plt.sca(ax)
        cp.pretty(ax)

        dD = self.deltaD
        for j in range(self.nSystems):
            system_times, _ = cf._log(self.off_time(j), kwargs.get('logX', None))

            for k in range(self.n_components):
                iS = self._component_indices(k, j)
                active = self.active[iS]
                (dD[iS][active]).plot(x=system_times[active], **kwargs)

        plt.ylabel("|{}| ({})".format(dD.name, dD.units))

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

    def set_posteriors(self, log=10):
        super().set_posteriors(log=log)

    def set_priors(self, height_prior=None, data_prior=None, relative_error_prior=None, additive_error_prior=None, **kwargs):

        if kwargs.get('solve_additive_error', False):
            additive_error_prior = Distribution('Uniform', kwargs['minimum_additive_error'], kwargs['maximum_additive_error'], log=10, prng=kwargs['prng'])

        super().set_priors(height_prior, data_prior, relative_error_prior, additive_error_prior, **kwargs)

        # if not data_prior is None:
        #     self.predictedData.prior = data_prior

    def set_additive_error_proposal(self, proposal, **kwargs):
        if proposal is None:
            if kwargs.get('solve_additive_error', False):
                proposal = Distribution('MvLogNormal', self.addErr, kwargs['additive_error_proposal_variance'], linearSpace=True, prng=kwargs['prng'])

        self.addErr.proposal = proposal

    def forward(self, mod):
        """ Forward model the data from the given model """

        assert isinstance(mod, Model1D), TypeError(
            "Invalid model class for forward modeling [1D]")
        fm = tdem1dfwd(self, mod)

        for i in range(self.nSystems):
            iSys = self._systemIndices(i)
            comps = []
            if 'x' in self.components:
                comps.append(fm[i].SX)
            if 'y' in self.components:
                comps.append(fm[i].SY)
            if 'z' in self.components:
                comps.append(-fm[i].SZ)
            self.predictedData[iSys] = np.hstack(comps)  # Store the necessary component

    # def fm_dlogc(self, mod):
    #     """ Forward model the data from the given model """

    #     assert isinstance(mod, Model1D), TypeError(
    #         "Invalid model class for forward modeling [1D]")
    #     return ga_fm_dlogc(self, mod)

    def sensitivity(self, model, ix=None, modelChanged=True):
        """ Compute the sensitivty matrix for the given model """

        assert isinstance(model, Model1D), TypeError(
            "Invalid model class for sensitivity matrix [1D]")
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

    def Isend(self, dest, world, **kwargs):

        if not 'system' in kwargs:
            # for i in range(self.nSystems):
            system = [sys.filename for sys in self.system]
            world.isend(system, dest=dest)

        super().Isend(dest, world)

        self.transmitter.Isend(dest, world)
        self.receiver.Isend(dest, world)

    @classmethod
    def Irecv(cls, source, world, **kwargs):

        if not 'system' in kwargs:
            kwargs['system'] = world.irecv(source=source).wait()

        out = super(TdemDataPoint, cls).Irecv(source, world, **kwargs)

        out._transmitter = CircularLoop.Irecv(source, world)
        out._receiver = CircularLoop.Irecv(source, world)

        return out