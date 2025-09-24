from os.path import split as psplt
from os.path import join
from copy import deepcopy

from itertools import cycle

from numpy import append, argwhere, asarray, cumsum, diag_indices, empty, exp, float64
from numpy import hstack, int32, log, nan, ravel_multi_index, repeat, s_, size, sqrt, squeeze
from numpy import all as npall
from numpy import any as npany

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplot, gcf, gca, sca, cla, plot, margins

from ...core.DataArray import DataArray
from ...statistics.StatArray import StatArray
from ...model.Model import Model
from .EmDataPoint import EmDataPoint
from ...forwardmodelling.Electromagnetic.TD.tdem1d import (
    tdem1dfwd, tdem1dsen, ga_fm_dlogc)
from ...system.SquareLoop import SquareLoop
from ...system.CircularLoop import CircularLoop
from ...system.Loop_pair import Loop_pair
from ...system.TdemSystem import TdemSystem
from ...system.TdemSystem_GAAEM import TdemSystem_GAAEM
from ...system.filters.butterworth import butterworth
from ...system.Waveform import Waveform
from ...statistics.Distribution import Distribution

#from ....base import Error as Err
from ....base import utilities as cf
from ....base import plotting as cp

class TdemDataPoint(EmDataPoint):
    """ Initialize a Time domain EMData Point


    TdemDataPoint(x, y, z, elevation, data, std, system, transmitter_loop, receiver_loop, line_number, fiducial)

    Parameters
    ----------
    x : float64
        The easting co-ordinate of the data point
    y : float64
        The northing co-ordinate of the data point
    z : float64
        The height of the data point above ground
    elevation : float64, optional
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
    line_number : float, optional
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
    __slots__ = ('_loop_pair', '_predicted_primary_field', '_predicted_secondary_field', '_primary_field', '_secondary_field')

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=0.0,
                 primary_field=None, secondary_field=None,
                 relative_error=None, additive_error=None, std=None,
                 predicted_primary_field=None, predicted_secondary_field=None,
                 system=None,
                 transmitter=None, receiver=None,
                 line_number=0.0, fiducial=0.0, **kwargs):

        self.system = system

        super().__init__(x=x, y=y, z=z, elevation=elevation,
                         components=self.components,
                         channels_per_system=self.nTimes,
                         data=None, std=std, predicted_data=None,
                         line_number=line_number, fiducial=fiducial, **kwargs)

        self.additive_error = additive_error
        self.relative_error = relative_error

        self.loop_pair = Loop_pair(transmitter, receiver)

        self.primary_field = primary_field
        self.secondary_field = secondary_field
        self.predicted_primary_field = predicted_primary_field
        self.predicted_secondary_field = predicted_secondary_field

        self.channel_names = None

    @property
    def active_system_indices(self):
        return squeeze(argwhere([npany(self.active[i]) for i in self.component_indices]))

    @EmDataPoint.additive_error.setter
    def additive_error(self, values):
        if values is None:
            values = self.nSystems
        else:
            assert size(values) == self.nSystems, ValueError("additive_error must be a list of size equal to the number of systems {}".format(self.nSystems))
        self._additive_error = StatArray(values, r'$\epsilon_{Additive}$', self.units)

    @property
    def addressof(self):
        return super().addressof + self.loop_pair.addressof

    @property
    def address(self):
        out = super().address
        for x in [self.loop_pair]:
            out = hstack([out, x.address.flatten()])

        return out

    @EmDataPoint.channel_names.setter
    def channel_names(self, values):
        if values is None:
            if self.system is None:
                self._channel_names = ['None']
                return


            self._channel_names = []

            for component in self.components:
                for i in range(self.nSystems):
                    for t in self.off_time(i):
                        self._channel_names.append('Time {:.3e} s {}'.format(t, component))
        else:
            assert all((isinstance(x, str) for x in values))
            assert len(values) == self.nChannels, Exception("Length of channel_names must equal total number of channels {}".format(self.nChannels))
            self._channel_names = values

    @property
    def channels(self):
        out = DataArray(hstack([self.off_time(i) for i in range(self.nSystems)]), name='time', units='s')
        return out

    @property
    def channels_per_system(self):
        return self.n_components * self.nTimes

    @EmDataPoint.data.getter
    def data(self):
        self._data[:] = self.secondary_field
        return self._data

    @property
    def loop_pair(self):
        return self._loop_pair

    @loop_pair.setter
    def loop_pair(self, value):
        assert isinstance(value, Loop_pair), TypeError("loop_pair must be a Loop_pair")
        self._loop_pair = value

    @EmDataPoint.predicted_data.getter
    def predicted_data(self):
        self._predicted_data = self._predicted_secondary_field
        return self._predicted_data

    @property
    def predicted_primary_field(self):
        return self._predicted_primary_field

    @predicted_primary_field.setter
    def predicted_primary_field(self, values):

        if values is None:
            values = self.n_components
        else:
            assert size(values) == self.n_components*self.n_systems, ValueError("predicted primary field must have size {}".format(self.n_components*self.n_systems))

        self._predicted_primary_field = StatArray(values, "Predicted primary field", self.units)

    @property
    def predicted_secondary_field(self):
        return self._predicted_secondary_field

    @predicted_secondary_field.setter
    def predicted_secondary_field(self, values):
        if values is None:
            values = self.nChannels
        else:
            assert size(values) == self.nChannels, ValueError("predicted secondary field must have size {}".format(self.nChannels))

        self._predicted_secondary_field = StatArray(values, "Predicted secondary field", self.units)

    @property
    def primary_field(self):
        return self._primary_field

    @primary_field.setter
    def primary_field(self, values):

        if values is None:
            values = self.n_components
        else:
            assert size(values) == self.n_components * self.nSystems, ValueError("primary field must have size {}".format(self.n_components*self.nSystems))

        self._primary_field = DataArray(values, "Primary field", self.units)

    @property
    def receiver(self):
        return self.loop_pair.receiver

    # @receiver.setter
    # def receiver(self, value):
    #     if not value is None:
    #         assert isinstance(value, EmLoop), TypeError(
    #             "receiver must be of type EmLoop")
    #         self._receiver = value

    @property
    def secondary_field(self):
        return self._secondary_field

    @secondary_field.setter
    def secondary_field(self, values):

        if values is None:
            values = self.nChannels
        else:
            assert size(values) == self.nChannels, ValueError("Secondary field must have size {}".format(self.nChannels))

        self._secondary_field = DataArray(values, "Secondary field", self.units)

    @EmDataPoint.system.setter
    def system(self, value):

        if value is None:
            self._system = None
            self.components = None
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

    @property
    def transmitter(self):
        return self.loop_pair.transmitter

    # @transmitter.setter
    # def transmitter(self, value):
    #     if not value is None:
    #         assert isinstance(value, EmLoop), TypeError("transmitter must be of type EmLoop")
    #         self._transmitter = value

    @property
    def n_components(self):
        return size(self.components)

    @property
    def nTimes(self):
        return asarray([x.nTimes for x in self.system])

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
        return cumsum(hstack([0, repeat(self.nTimes, self.n_components)]))

    def _component_indices(self, component=0, system=0):
        i = ravel_multi_index((component, system), (self.n_components, self.nSystems))
        return s_[self._ravel_index[i]:self._ravel_index[i+1]]

    @property
    def component_indices(self):
        return [self._component_indices(comp, sys) for comp in range(self.n_components) for sys in range(self.nSystems)]

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out.system = self._system
        out._loop_pair = deepcopy(self.loop_pair, memo=memo)
        out._primary_field = deepcopy(self.primary_field, memo=memo)
        out._secondary_field = deepcopy(self.secondary_field, memo=memo)
        out._predicted_primary_field = deepcopy(self.predicted_primary_field, memo=memo)
        out._predicted_secondary_field = deepcopy(self.predicted_secondary_field, memo=memo)

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

        assert npall(self.relative_error > 0.0), ValueError('relative_error must be > 0.0')

        # For each system assign error levels using the user inputs
        for i in range(self.nSystems):
            off_times = self.off_time(i)
            for j in range(self.n_components):
                ic = self._component_indices(j, i)
                relative_error = self.relative_error[(i*self.n_components)+j] * self.secondary_field[ic]
                additive_error = exp(log(self.additive_error[i]) - 0.5 * (log(off_times) - log(1e-3)))
                variance = relative_error**2.0 + additive_error**2.0
                self._std[ic] = sqrt(variance)

            # # Compute the relative error
            # rErr = self.relative_error[i] * self.secondary_field[iSys]
            # # aErr = exp(log(self.additive_error[i]) - 0.5 * log(self.off_time(i)) + t0)
            # # self._std[iSys] = sqrt((rErr**2.0) + (aErr[i]**2.0))

            # self._std[iSys] = sqrt((rErr**2.0) + (self.additive_error[i]**2.0))


        # Update the variance of the predicted data prior
        if self.predicted_data.hasPrior:
            self.predicted_data.prior.variance = self._std[self.active]**2.0

        return self._std

    @property
    def system_indices(self):
        tmp = hstack([0, cumsum(self.channels_per_system)])
        return [s_[tmp[i]:tmp[i+1]] for i in range(self.nSystems)]

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
        data = empty(0)
        std = empty(0)

        for fName in dataFileName:
            with open(fName, 'r') as f:
                # Header line
                dtype, x, y, z, elevation, fiducial, line_number, current = self.__aarhus_header(
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
                    frontGateTime = float64(f.readline().strip())
                    offTimeFilters = self.__aarhus_filters(f, 1)

                # Data and standard deviation
                times, d, s = self.__aarhus_data(f)
                data = hstack([data, d])
                std = hstack([std, s*d])

                system.append(TdemSystem(offTimes=times,
                                         transmitterLoop=transmitterLoop,
                                         receiverLoop=receiverLoop,
                                         waveform=waveform,
                                         offTimeFilters=offTimeFilters))

        TdemDataPoint.__init__(self, x, y, 0.0, elevation, data, std,
                               system=system, line_number=line_number, fiducial=fiducial)

    def __aarhus_header(self, f):
        line = f.readline().strip().split(';')
        dtype = x = y = z = elevation = current = None
        fiducial = line_number = 0.0
        for item in line:
            item = item.split("=")
            tag = item[0].lower()
            value = item[-1]

            if tag == "datatypestring":
                dtype = value
            elif tag == "xutm":
                x = float64(value)
            elif tag == "yutm":
                y = float64(value)
            elif tag == "elevation":
                elevation = float64(value)
            elif tag == "stationnumber":
                fiducial = float64(value)
            elif tag == "line_number":
                line_number = float64(value)
            elif tag == "current":
                current = float64(value)

        assert not any([x, y, elevation, current] is None), ValueError(
            "Aarhus file header line must contain 'XUTM', 'YUTM', 'Elevation', 'current'")

        return dtype, x, y, z, elevation, fiducial, line_number, current

    def __aarhus_source(self, f):
        line = f.readline().strip().split()
        source = int32(line[0])
        polarization = int32(line[1])

        assert source == 7, ValueError(
            "Have only incorporated source == 7 so far.")
        assert polarization == 3, ValueError(
            "Have only incorporated polarization == 3 so far.")

        return source, polarization

    def __aarhus_positions(self, f):
        line = f.readline().strip().split()
        tx, ty, tz, rx, ry, rz = [float(x) for x in line]
        return asarray([rx - tx, ry - ty, rz - tz])  # loopOffset

    def __aarhus_loop_dimensions(self, f, source):

        if source <= 6:
            return
        if source in [10, 11]:
            return

        line = f.readline().strip().split()
        if source == 7:
            dx, dy = [float(x) for x in line]
            assert dx == dy, ValueError(
                "Only handling square loops at the moment")
            transmitter = SquareLoop(sideLength=dx)
            receiver = CircularLoop()  # Dummy.
            return transmitter, receiver

    def __aarhus_data_transforms(self, f):
        line = f.readline().strip().split()
        a, b, c = [int32(x) for x in line]
        assert a == 3, ValueError("Can only handle data transform 3.  dB/dT")

        return a

    def __aarhus_waveform(self, f):
        line = f.readline().strip().split()
        typ, nWaveforms = [int32(x) for x in line]

        assert typ == 3, ValueError(
            "Can only handle user defined waveforms, option 3")

        time = empty(0)
        amplitude = empty(0)
        for i in range(nWaveforms):
            line = f.readline().strip().split()
            tmp = asarray([float(x) for x in line[1:]])
            time = append(time, hstack([tmp[:2], tmp[5::4]]))
            amplitude = append(amplitude, hstack([tmp[2:4], tmp[6::5]]))

        return time, amplitude

    def __aarhus_frontgate(self, f):
        line = f.readline().strip().split()
        nFilters = int32(line[0])
        frontGate = bool(int32(line[1]))
        damping = float64(line[2])

        return nFilters, frontGate, damping

    def __aarhus_filters(self, f, nFilters):

        filters = []

        for i in range(nFilters):
            # Low Pass Filter
            line = f.readline().strip().split()
            nLowPass = int32(line[0])
            for j in range(nLowPass):
                order = int32(float(line[(2*j)+1]))
                frequency = float64(line[(2*j)+2])
                b = butterworth(order, frequency, btype='low', analog=True)
                filters.append(b)

            # High Pass Filter
            line = f.readline().strip().split()
            nHighPass = int32(line[0])
            for j in range(nHighPass):
                order = int32(float64(line[(2*j)+1]))
                frequency = float64(line[(2*j)+2])
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
            time.append(float64(line[0]))
            tmp = float64(line[1])
            data.append(nan if tmp == 999 else tmp)
            std.append(float64(line[2]))

        return asarray(time), asarray(data), asarray(std)

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """

        grp = super().createHdf(parent, name, withPosterior, add_axis, fillvalue)

        grp.create_dataset('nSystems', data=self.nSystems)
        for i in range(self.nSystems):
            self.system[i].toHdf(grp, 'System{}'.format(i))

        grp.create_dataset('components', data=self._components)
        self.loop_pair.createHdf(grp, 'loop_pair', add_axis=add_axis, fillvalue=fillvalue)

        self.primary_field.createHdf(grp, 'primary_field', add_axis=add_axis, fillvalue=fillvalue)
        self.secondary_field.createHdf(grp, 'secondary_field', add_axis=add_axis, fillvalue=fillvalue)
        self.predicted_primary_field.createHdf(grp, 'predicted_primary_field', add_axis=add_axis, fillvalue=fillvalue)
        self.predicted_secondary_field.createHdf(grp, 'predicted_secondary_field', add_axis=add_axis, fillvalue=fillvalue)

        if add_axis is not None:
            grp.attrs['repr'] = 'TdemData'

        return grp

    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        super().writeHdf(parent, name, withPosterior, index)

        grp = parent[name]

        self.loop_pair.writeHdf(grp, 'loop_pair', index=index)

        self.primary_field.writeHdf(grp, 'primary_field', index=index)
        self.secondary_field.writeHdf(grp, 'secondary_field', index=index)

        self.predicted_primary_field.writeHdf(grp, 'predicted_primary_field', index=index)
        self.predicted_secondary_field.writeHdf(grp, 'predicted_secondary_field', index=index)

    @classmethod
    def fromHdf(cls, grp, **kwargs):
        """ Reads the object from a HDF group """

        nSystems = int32(asarray(grp['nSystems']))

        systems = TdemDataPoint.read_systems_from_h5(grp, **kwargs)

        self = super(TdemDataPoint, cls).fromHdf(grp, system=systems, **kwargs)

        self.loop_pair = Loop_pair.fromHdf(grp['loop_pair'], **kwargs)

        self._primary_field = StatArray.fromHdf(grp['primary_field'], **kwargs)
        self._secondary_field = StatArray.fromHdf(grp['secondary_field'], **kwargs)
        self._predicted_primary_field = StatArray.fromHdf(grp['predicted_primary_field'], **kwargs)
        self._predicted_secondary_field = StatArray.fromHdf(grp['predicted_secondary_field'], **kwargs)

        return self

    @staticmethod
    def read_systems_from_h5(grp, **kwargs):
        nSystems = int32(asarray(grp.get('nSystems')))
        if 'system_filename' in kwargs:
            system_filename = kwargs['system_filename']
            if not isinstance(system_filename, list): system_filename = [system_filename]

        systems = [None]*nSystems
        for i in range(nSystems):
            if 'system_filename' in kwargs:
                systems[i] = TdemSystem(system_filename=system_filename[i])
            else:
                # Get the system file name. h5py has to encode strings using utf-8, so decode it!
                systems[i] = TdemSystem.fromHdf(grp['System{}'.format(i)], 'System{}.stm'.format(i))
        return systems

    def perturb(self):
        super().perturb()
        self.loop_pair.perturb()

    def _init_posterior_plots(self, gs=None):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """
        if gs is None:
            gs = figure()

        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        n_rows = 1
        if (self.relative_error.hasPosterior & self.additive_error.hasPosterior) or any([self.transmitter.hasPosterior, self.loop_pair.hasPosterior, self.receiver.hasPosterior]):
            n_rows = 2

        splt = gs.subgridspec(n_rows, 1, wspace=0.3)

        ## Top row of plot
        n_cols = 1
        width_ratios = None
        if self.relative_error.hasPosterior or self.additive_error.hasPosterior:
            n_cols = 2
            width_ratios = (1, 2)

        splt_top = splt[0].subgridspec(1, n_cols, width_ratios=width_ratios)

        ax = {}
        # Data axis
        ax['data'] = subplot(splt_top[-1])

        if self.relative_error.hasPosterior:
            # Relative error axes
            ax['relative_error'] = self.relative_error._init_posterior_plots(splt_top[0])
        else:
            if self.additive_error.hasPosterior:
                ax['additive_error'] = self.additive_error._init_posterior_plots(splt_top[0])

        ## Bottom row of plot
        n_cols = any([self.transmitter.hasPosterior, self.loop_pair.hasPosterior, self.receiver.hasPosterior])
        n_cols += (self.relative_error.hasPosterior and self.additive_error.hasPosterior)

        if n_cols > 0:
            widths = None
            if n_cols == 1 and not any([self.transmitter.hasPosterior, self.loop_pair.hasPosterior, self.receiver.hasPosterior]):
                n_cols = 2
                widths = (1, 2)
            if n_cols == 2:
                widths = (1, 2)

            splt_bottom = splt[1].subgridspec(1, n_cols, width_ratios=widths)

            i = 0
            # Additive Error axes
            if self.relative_error.hasPosterior & self.additive_error.hasPosterior:
                tmp = []
                tmp = self.additive_error._init_posterior_plots(splt_bottom[i])
                if tmp is not None:
                    for j in range(self.nSystems):
                        others = s_[(j * self.n_components):(j * self.n_components)+self.n_components]
                        tmp[1].get_shared_y_axes().joined(tmp[1], *tmp[others])
                ax['additive_error'] = tmp
                i += 1

            if any([self.transmitter.hasPosterior, self.loop_pair.hasPosterior, self.receiver.hasPosterior]):
                # Loop pair
                ax['loop_pair'] = self.loop_pair._init_posterior_plots(splt_bottom[i])

        return ax


    def plot_posteriors(self, axes=None, **kwargs):

        if axes is None:
            axes = kwargs.pop('fig', gcf())

        if not isinstance(axes, dict):
            axes = self._init_posterior_plots(axes)

        data_kwargs = kwargs.pop('data_kwargs', {})
        rel_error_kwargs = kwargs.pop('rel_error_kwargs', {})
        add_error_kwargs = kwargs.pop('add_error_kwargs', {})

        overlay = kwargs.pop('overlay', None)

        ax = axes['data']; ax.cla()
        if self.predicted_data.hasPosterior:
            self.predicted_data.plot_posteriors(ax = ax, colorbar=False, **data_kwargs)
        self.plot(ax=ax, **data_kwargs)

        c = cp.wellSeparated[0] if overlay is None else cp.wellSeparated[3]
        self.plot_predicted(color=c, ax=ax, **data_kwargs)

        if self.relative_error.hasPosterior:
            self.relative_error.plot_posteriors(ax=axes['relative_error'], **rel_error_kwargs)

        if self.additive_error.hasPosterior:
            add_error_kwargs['colorbar'] = False
            self.additive_error.plot_posteriors(ax=axes['additive_error'], **add_error_kwargs)

        if any([x.hasPosterior for x in [self.transmitter, self.loop_pair, self.receiver]]):
            self.loop_pair.plot_posteriors(axes = axes['loop_pair'], **kwargs)

        if overlay is not None:
            self.overlay_on_posteriors(overlay, axes, **kwargs)

    def overlay_on_posteriors(self, overlay, axes, rel_error_kwargs={}, add_error_kwargs={}, **kwargs):

        assert isinstance(overlay, TdemDataPoint), TypeError("overlay must have type TdemDataPoint")

        if self.relative_error.hasPosterior:
            self.relative_error.overlay_on_posteriors(overlay.relative_error, ax=axes['relative_error'], **rel_error_kwargs, **kwargs)

        if self.additive_error.hasPosterior:
            add_error_kwargs['colorbar'] = False
            self.additive_error.overlay_on_posteriors(overlay.additive_error, ax=axes['additive_error'], **add_error_kwargs, **kwargs)

        if self.loop_pair.hasPosterior:
            self.loop_pair.overlay_on_posteriors(overlay=overlay.loop_pair, axes = axes['loop_pair'], **kwargs)

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
        ax = plt.gca() if ax is None else ax

        markers = tuple(kwargs.pop('marker', ('o', 'x', 'v')))
        kwargs['markersize'] = kwargs.pop('markersize', 3)
        c = kwargs.pop('color', [cp.wellSeparated[i+1] for i in range(self.nSystems)])
        mfc = kwargs.pop('markerfacecolor', [cp.wellSeparated[i+1] for i in range(self.nSystems)])
        assert len(c) == self.nSystems, ValueError("color must be a list of length {}".format(self.nSystems))
        assert len(mfc) == self.nSystems, ValueError("markerfacecolor must be a list of length {}".format(self.nSystems))
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor', 'k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth', 1.0)
        kwargs['alpha'] = kwargs.pop('alpha', 0.8)
        kwargs['linestyle'] = kwargs.pop('linestyle', 'none')
        kwargs['linewidth'] = kwargs.pop('linewidth', 1)

        xscale = kwargs.pop('xscale', 'log')
        yscale = kwargs.pop('yscale', 'log')

        kwargs.pop('logX', None)
        kwargs.pop('logY', None)

        marker = cycle(markers)

        for j in range(self.nSystems):
            system_times = self.off_time(j)

            for k in range(self.n_components):

                # kwargs['marker'] = markers[self._components[k]]
                kwargs['marker'] = next(marker)

                icomp = self._component_indices(k, j)
                d = self.data[icomp]

                if (with_error_bars):
                    s = self.std[icomp]
                    ax.errorbar(system_times, d, yerr=s,
                                 color=c[j],
                                 markerfacecolor=mfc[j],
                                 label='System: {}{}'.format(j+1, self.components[k]),
                                 **kwargs)
                else:
                    ax.plot(system_times, d,
                             markerfacecolor=mfc[j],
                             label='System: {}{}'.format(j+1, self.components[k]),
                             **kwargs)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(cf.getNameUnits(self.data))
        ax.set_title(title)

        if self.nSystems > 1 or self.n_components > 1:
            ax.legend()

        return ax

    def plot_predicted(self, title='Time Domain EM Data', **kwargs):

        ax = kwargs.get('ax', None)
        ax = plt.gca() if ax is None else ax

        labels = kwargs.pop('labels', True)

        if (labels):
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(cf.getNameUnits(self.predicted_data))
            ax.set_title(title)

        kwargs['color'] = kwargs.pop('color', cp.wellSeparated[3])
        kwargs['linewidth'] = kwargs.pop('linewidth', 1)
        kwargs['alpha'] = kwargs.pop('alpha', 0.7)
        xscale = kwargs.pop('xscale', 'log')
        yscale = kwargs.pop('yscale', 'log')

        kwargs.pop('logX', None)
        kwargs.pop('logY', None)

        for j in range(self.nSystems):
            system_times = self.off_time(j)

            for k in range(self.n_components):
                iS = self._component_indices(k, j)

                if npall(self.data <= 0.0):
                    active = (self.predicted_data[iS] > 0.0)

                else:
                    active = self.active[iS]

                p = self.predicted_data[iS][active]
                p.plot(x=system_times[active], **kwargs)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

    def plotDataResidual(self, title='', **kwargs):

        ax = kwargs.get('ax', None)
        ax = plt.gca() if ax is None else plt.sca(ax)
        cp.pretty(ax)

        dD = self.deltaD
        for j in range(self.nSystems):
            system_times, _ = cf._log(self.off_time(j), kwargs.get('logX', None))

            for k in range(self.n_components):
                iS = self._component_indices(k, j)
                active = self.active[iS]
                (abs(dD[iS][active])).plot(x=system_times[active], **kwargs)

        ax.set_ylabel("|{}| ({})".format(dD.name, dD.units))

        ax.set_title(title)

    @property
    def probability(self):
        return super().probability + self.loop_pair.probability

    def set_posteriors(self, log=10):
        super().set_posteriors(log=log)
        self.loop_pair.set_posteriors()

    def reset_posteriors(self):
        super().reset_posteriors()
        self.loop_pair.reset_posteriors()

    def set_priors(self, relative_error_prior=None, additive_error_prior=None, data_prior=None, **kwargs):

        if additive_error_prior is None:
            if kwargs.get('solve_additive_error', False):
                additive_error_prior = Distribution('Uniform',
                                                    kwargs['minimum_additive_error'],
                                                    kwargs['maximum_additive_error'],
                                                    log=True,
                                                    prng=kwargs.get('prng'))

        if data_prior is None:
            data_prior = Distribution('MvNormal', self.data[self.active], self.std[self.active]**2.0, prng=kwargs.get('prng'))

        super().set_priors(relative_error_prior, additive_error_prior, data_prior, **kwargs)

        self.loop_pair.set_priors(**kwargs)

    def set_proposals(self, relative_error_proposal=None, additive_error_proposal=None, **kwargs):

        super().set_proposals(relative_error_proposal, additive_error_proposal, **kwargs)

        self.loop_pair.set_proposals(**kwargs)

    def set_additive_error_proposal(self, proposal, **kwargs):
        if proposal is None:
            if kwargs.get('solve_additive_error', False):
                proposal = Distribution('MvLogNormal', self.additive_error, kwargs['additive_error_proposal_variance'], linearSpace=True, prng=kwargs.get('prng'))

        self.additive_error.proposal = proposal

    @property
    def summary(self):
        msg = super().summary + self.loop_pair.summary
        return msg

    def update_posteriors(self):
        super().update_posteriors()
        self.loop_pair.update_posteriors()

    def forward(self, model):
        """ Forward model the data from the given model """

        assert isinstance(model, Model), TypeError(
            "Invalid model class {} for forward modeling [1D]".format(type(model)))
        fm = tdem1dfwd(self, model)

        for i in range(self.nSystems):
            iSys = self._systemIndices(i)
            primary = []
            secondary = []
            if 'x' in self.components:
                primary.append(fm[i].PX)
                secondary.append(fm[i].SX)
            if 'y' in self.components:
                primary.append(fm[i].PY)
                secondary.append(fm[i].SY)
            if 'z' in self.components:
                primary.append(-fm[i].PZ)
                secondary.append(-fm[i].SZ)

            self.predicted_secondary_field[iSys] = hstack(secondary)  # Store the necessary component

            s = s_[i * self.n_components: (i * self.n_components) + self.n_components]

            self.predicted_primary_field[s] = hstack(primary)

    def sensitivity(self, model, ix=None, model_changed=False):
        """ Compute the sensitivty matrix for the given model """
        assert isinstance(model, Model), TypeError("Invalid model class for sensitivity matrix [1D]")
        self._sensitivity_matrix = DataArray(tdem1dsen(self, model, ix, model_changed), 'Sensitivity', r'$\frac{V}{SAm^{3}}$')
        return self.sensitivity_matrix

    def fm_dlogc(self, model):
        values, J = ga_fm_dlogc(self, model)
        self._sensitivity_matrix = DataArray(J, 'Sensitivity', r'$\frac{V}{SAm^{3}}$')

    def _empymodForward(self, mod):

        stuff = None

    # def _simPEGForward(self, mod):

    #     from SimPEG import Maps
    #     from simpegEM1D import (EM1DSurveyTD, EM1D, set_mesh_1d)

    #     mesh1D = set_mesh_1d(mod.depth)
    #     expmap = Maps.ExpMap(mesh1D)
    #     prob = EM1D(mesh1D, sigmaMap = expmap, chi = mod.chim)

    #     if (self.dualMoment()):

    #         simPEG_survey = EM1DSurveyTD(
    #             rx_location=array([0., 0., 0.]),
    #             src_location=array([0., 0., 0.]),
    #             topo=r_[0., 0., 0.],
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
    #             rx_location=array([0., 0., 0.]),
    #             src_location=array([0., 0., 0.]),
    #             topo=r_[0., 0., 0.],
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

    #     self._predicted_data[:] = -simPEG_survey.dpred(mod.par)

    def Isend(self, dest, world, **kwargs):

        if not 'system' in kwargs:
            # for i in range(self.nSystems):
            system = [sys.filename for sys in self.system]
            world.isend(system, dest=dest).wait()

        super().Isend(dest, world)

        self.loop_pair.Isend(dest, world)

        self.primary_field.Isend(dest, world)
        self.secondary_field.Isend(dest, world)
        self.predicted_primary_field.Isend(dest, world)
        self.predicted_secondary_field.Isend(dest, world)

    @classmethod
    def Irecv(cls, source, world, **kwargs):

        if not 'system' in kwargs:
            kwargs['system'] = world.irecv(source=source).wait()

        out = super(TdemDataPoint, cls).Irecv(source, world, **kwargs)

        out._loop_pair = Loop_pair.Irecv(source, world)

        out._primary_field = DataArray.Irecv(source, world)
        out._secondary_field = DataArray.Irecv(source, world)
        out._predicted_primary_field = StatArray.Irecv(source, world)
        out._predicted_secondary_field = StatArray.Irecv(source, world)

        return out
