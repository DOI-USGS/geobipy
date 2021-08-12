from copy import deepcopy

from ....classes.core import StatArray
from ...model.Model1D import Model1D
from .TdemDataPoint import TdemDataPoint
from ...forwardmodelling.Electromagnetic.TD.tdem1d import (tdem1dfwd, tdem1dsen)
from ...system.EmLoop import EmLoop
from ...system.SquareLoop import SquareLoop
from ...system.CircularLoop import CircularLoop
from ....base.logging import myLogger
from ...system.TdemSystem import TdemSystem
from ...system.filters.butterworth import butterworth
from ...system.Waveform import Waveform
from ...statistics.Histogram1D import Histogram1D
from ...statistics.Histogram2D import Histogram2D
import matplotlib.pyplot as plt
import numpy as np

#from ....base import Error as Err
from ....base import fileIO as fIO
from ....base import utilities as cf
from ....base import plotting as cp
from ....base import MPI as myMPI
from os.path import split as psplt
from os.path import join


class Tempest_datapoint(TdemDataPoint):
    """ Initialize a Tempest Time domain data point


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
                 primary_field=None, secondary_field=None,
                 std=None,
                 predicted_primary_field=None, predicted_secondary_field=None,
                 system=None,
                 transmitter_loop=None, receiver_loop=None, loopOffset=[0.0, 0.0, 0.0],
                 lineNumber=0.0, fiducial=0.0):
        """Initializer. """

        if system is None:
            return

        super().__init__(x=x, y=y, z=z, elevation=elevation,
                         data=None, std=None, predictedData=None,
                         system=system, transmitter_loop=transmitter_loop,
                         receiver_loop=receiver_loop, loopOffset=loopOffset,
                         lineNumber=lineNumber, fiducial=fiducial)

        self.primary_field = primary_field
        self.secondary_field = secondary_field

        self.predicted_primary_field = predicted_primary_field
        self.predicted_secondary_field = predicted_secondary_field

        # self.data[self._component_indices(0, 0)] += self.x_primary_field
        # self.data[self._component_indices(1, 0)] += self.z_primary_field

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out._primary_field = deepcopy(self.primary_field)

        return out

    @property
    def channels(self):
        return np.squeeze(np.asarray([np.tile(self.times(i), 2) for i in range(self.nSystems)]))

    @property
    def data(self):
        for i in range(self.n_components):
            ic = self._component_indices(i, 0)
            self._data[ic] = self.primary_field[i] + self.secondary_field[ic]
        return self._data

    @data.setter
    def data(self, values):
        if not '_data' in self.__dict__:
            self._data = StatArray.StatArray(self.nChannels, "Tempest data", self.units)

        if not values is None:
            assert values.size == self.nChannels, ValueError("Size of data must equal total number of time channels * components {}".format(self.nChannels))
            # Mask invalid data values less than 0.0 to NaN
            self._data[:] = values

    @property
    def predicted_primary_field(self):
        return self._predicted_primary_field

    @predicted_primary_field.setter
    def predicted_primary_field(self, values):
        if not '_predicted_primary_field' in self.__dict__:
            self._predicted_primary_field = np.squeeze(StatArray.StatArray(self.n_components, "Predicted primary field", self.units))

        if not values is None:
            self._predicted_primary_field[:] = values

    @property
    def primary_field(self):
        return self._primary_field

    @primary_field.setter
    def primary_field(self, values):
        if not '_primary_field' in self.__dict__:
            self._primary_field = StatArray.StatArray(self.n_components, "Primary field", self.units)

        if not values is None:
            self._primary_field[:] = values

    @TdemDataPoint.predictedData.getter
    def predictedData(self):
        for i in range(self.n_components):
            ic = self._component_indices(i, 0)
            self._predictedData[ic] = self.predicted_primary_field[i] + self.predicted_secondary_field[ic]

        return self._predictedData

    # @predictedData.setter
    # def predictedData(self, values):
    #     if not '_predictedData' in self.__dict__:
    #         self._predictedData = StatArray.StatArray(self.nChannels, "Predicted Data", self.units)

    #     if not values is None:
    #         assert values.size == self.nChannels, ValueError("Size of predictedData must equal total number of time channels * components {}".format(self.nChannels))
    #         # Mask invalid data values less than 0.0 to NaN
    #         self._predictedData[:] = values

    @TdemDataPoint.units.setter
    def units(self, value):

        if value is None:
            self._units = r"fT"
        else:
            assert isinstance(value, str), TypeError('units must have type str')
            self._units = value

    def createHdf(self, parent, name, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """

        grp = super().createHdf(parent, name, withPosterior, nRepeats, fillvalue)
        self.primary_field.createHdf(grp, 'primary_field', nRepeats=nRepeats, fillvalue=fillvalue)

        return grp


        # grp.create_dataset('nSystems', data=self.nSystems)
        # for i in range(self.nSystems):
        #     grp.create_dataset('System{}'.format(i), data=np.string_(psplt(self.system[i].fileName)[-1]))

        # self.transmitter.createHdf(grp, 'T', nRepeats=nRepeats, fillvalue=fillvalue)
        # self.receiver.createHdf(grp, 'R', nRepeats=nRepeats, fillvalue=fillvalue)
        # self.loopOffset.createHdf(grp, 'loop_offset', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        super().writeHdf(parent, name, withPosterior, index)

        grp = parent[name]

        self.primary_field.writeHdf(grp, 'primary_field', index=index)

    def fromHdf(self, grp, index=None, **kwargs):
        """ Reads the object from a HDF group """

        super().fromHdf(grp, index, **kwargs)

        self._primary_field = StatArray.StatArray().fromHdf(grp['primary_field'], index=index)

        return self

    def plotWaveform(self,**kwargs):
        for i in range(self.nSystems):
            if (self.nSystems > 1):
                plt.subplot(2, 1, i + 1)
            plt.plot(self.system[i].waveform.time, self.system[i].waveform.current, **kwargs)
            cp.xlabel('Time (s)')
            cp.ylabel('Normalized Current (A)')
            plt.margins(0.1, 0.1)


    def plot(self, **kwargs):
        kwargs['xscale'] = 'linear'
        kwargs['yscale'] = 'linear'
        kwargs['logX'] = 10
        return super().plot(**kwargs)

    def plotPredicted(self, **kwargs):
        kwargs['xscale'] = 'linear'
        kwargs['yscale'] = 'linear'
        kwargs['logX'] = 10
        return super().plotPredicted(**kwargs)

    def setPosteriors(self):
        return super().setPosteriors(log=None)

    # def set_predicted_data_posterior(self):
    #     if self.predictedData.hasPrior:
    #         times = np.log10(self.times(0))

    #         xbuf = 0.05*(times[-1] - times[0])
    #         xbins = np.logspace(times[0]-xbuf, times[-1]+xbuf, 200)
    #         ybins = np.linspace(0.8*np.nanmin(self.data), 1.2*np.nanmax(self.data), 200)
    #         # rto = 0.5 * (ybins[0] + ybins[-1])
    #         # ybins -= rto

    #         H = Histogram2D(xBins=xbins, xlog=10, yBins=ybins)

    #         self.predictedData.setPosterior(H)

    #         # H = Histogram2D(xBins = StatArray.StatArray(self.z.prior.bins(), name=self.z.name, units=self.z.units), relativeTo=self.z)
    #         # self.z.setPosterior(H)


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
        #assert (isinstance(relativeErr, list)), TypeError("relativeErr must be a list of size equal to the number of systems {}".format(self.nSystems))
        assert (relativeErr.size == self.nSystems), TypeError("relativeErr must be a list of size equal to the number of systems {}".format(self.nSystems))

        additiveErr = np.atleast_1d(additiveErr)
        #assert (isinstance(additiveErr, list)), TypeError("additiveErr must be a list of size equal to the number of systems {}".format(self.nSystems))
        check = (self.nSystems, self.nChannels)
        assert (additiveErr.size in check), TypeError("additiveErr must have size of one of these {}".format(check))

        # For each system assign error levels using the user inputs
        for i in range(self.nSystems):
            assert (isinstance(relativeErr[i], float) or isinstance(relativeErr[i], np.ndarray)), TypeError("relativeErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.nTimes[i]))
            assert (np.all(relativeErr[i] > 0.0)), ValueError("relativeErr for system {} cannot contain values <= 0.0.".format(i+1))

            assert (isinstance(additiveErr[i], float) or isinstance(additiveErr[i], np.ndarray)), TypeError("additiveErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.nTimes[i]))
            assert (np.all(additiveErr[i] > 0.0)), ValueError("additiveErr for system {} should contain values > 0.0. Make sure the values are in linear space".format(i+1))
            iSys = self._systemIndices(system=i)

            # Compute the relative error
            data = self.data[iSys]

            rErr = relativeErr[i] * data
            if additiveErr.size == self.nSystems:
                aErr = np.full_like(rErr, fill_value=additiveErr[i])
            else:
                aErr = additiveErr[iSys]

            self._std[iSys] = np.sqrt((rErr**2.0) + (aErr**2.0))

        # Update the variance of the predicted data prior
        if self._predictedData.hasPrior:
            self._predictedData.prior.variance[np.diag_indices(self.active.size)] = self._std[self.active]**2.0

    def updatePosteriors(self):
        super().updatePosteriors()
        if self.predictedData.hasPosterior:
            for i in range(self.n_components):
                j = self._component_indices(i, 0)
                self.predictedData.posterior.update_line(x=self.channels[j], y=self.predictedData[j])


    def forward(self, mod):
        """ Forward model the data from the given model """

        assert isinstance(mod, Model1D), TypeError("Invalid model class for forward modeling [1D]")
        tdem1dfwd(self, mod)

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
        self.transmitter.Isend(dest, world)
        self.receiver.Isend(dest, world)

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
        transmitter = c.Irecv(source, world)
        receiver = c.Irecv(source, world)
        loopOffset  = tmp[-3:]
        return TdemDataPoint(tmp[0], tmp[1], tmp[2], tmp[3], data=d, std=s, predictedData=p, system=systems, transmitter_loop=transmitter, receiver_loop=receiver, loopOffset=loopOffset, lineNumber=tmp[5], fiducial=tmp[6])
