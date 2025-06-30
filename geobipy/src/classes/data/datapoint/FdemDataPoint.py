""" @FdemDataPoint_Class
Module describing a frequency domain EMData Point that contains a single measurement.
"""
from copy import copy, deepcopy

from numpy import asarray, exp
from numpy import int32, isinf, log10, logspace, s_, squeeze, tile
from numpy import all as npall

from ...core.DataArray import DataArray
from ...statistics.StatArray import StatArray
from ...forwardmodelling.Electromagnetic.FD.fdem1d import fdem1dfwd, fdem1dsen
from .EmDataPoint import EmDataPoint
from ...model.Model import Model
from...mesh.RectilinearMesh2D import RectilinearMesh2D
from ...statistics.Histogram import Histogram
from ...statistics.Distribution import Distribution
from ...system.FdemSystem import FdemSystem
import matplotlib.pyplot as plt

#from ....base import Error as Err
from ....base import utilities as cf
from ....base import MPI as myMPI
from ....base import plotting as cp

class FdemDataPoint(EmDataPoint):
    """Class defines a Frequency domain electromagnetic data point.

    Contains an easting, northing, height, elevation, observed and predicted data, and uncertainty estimates for the data.

    FdemDataPoint(x, y, z, elevation, data, std, system, line_number, fiducial)

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
    line_number : float, optional
        The line number associated with the datapoint
    fiducial : float, optional
        The fiducial associated with the datapoint

    """

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=0.0, data=None, std=None, predicted_data=None, system=None, line_number=0.0, fiducial=0.0):
        """Define initializer. """

        # self._system = None
        # if (system is None):
        #     return super().__init__(x=x, y=y, z=z, elevation=elevation)

        self.system = system

        super().__init__(x=x, y=y, z=z, elevation=elevation,
                         components=self.components,
                         channels_per_system=2*self.nFrequencies,
                         data=data, std=std, predicted_data=predicted_data,
                         line_number=line_number, fiducial=fiducial)

        self._data.name = 'Frequency domain data'

        # StatArray of calibration parameters
        # The four columns are Bias,Variance,InphaseBias,QuadratureBias.
        # self.calibration = DataArray([self.nChannels * 2], 'Calibration Parameters')

        self.channel_names = None

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out._system = self._system
        # out.calibration = deepcopy(self.calibration)
        return out

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if value is None:
            value = "ppm"
        else:
            assert isinstance(value, str), TypeError("units must have type str")
        self._units = value

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, value):

        if value is None:
            self._system = None
            self.components = None
            self.channels_per_system = None
            return

        if isinstance(value, (str, FdemSystem)):
            value = [value]

        assert all((isinstance(sys, (str, FdemSystem)) for sys in value)), TypeError("System must have items of type str or geobipy.FdemSystem")

        systems = []
        for j, sys in enumerate(value):
            if (isinstance(sys, str)):
                systems.append(FdemSystem().read(sys))
            elif (isinstance(sys, FdemSystem)):
                systems.append(sys)

        self._system = systems

        self.components = 'z'
        self.channels_per_system = 2 * self.system[0].nFrequencies

    @EmDataPoint.channel_names.setter
    def channel_names(self, values):
        if values is None:
            if self.system is None:
                self._channel_names = ['None']
                return
            self._channel_names = []
            for i in range(self.nSystems):
                # Set the channel names
                if not self.system[i] is None:
                    for iFrequency in range(2*self.nFrequencies[i]):
                        self._channel_names.append('{} {} (Hz)'.format(self.getMeasurementType(iFrequency, i), self.getFrequency(iFrequency, i)))
        else:
            assert all((isinstance(x, str) for x in values))
            assert len(values) == self.nChannels, Exception("Length of channel_names must equal total number of channels {}".format(self.nChannels))
            self._channel_names = values

    @property
    def nFrequencies(self):
        return (self.channels_per_system / 2).astype(int32)

    @property
    def channels(self):
        return squeeze(asarray([tile(self.frequencies(i), 2) for i in range(self.nSystems)]))


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

        return s_[self.systemOffset[system]:self.systemOffset[system] + self.nFrequencies[system]]


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

        return s_[self.systemOffset[system] + self.nFrequencies[system]: 2*self.nFrequencies[system]]


    def frequencies(self, system=0):
        """ Return the frequencies in an StatArray """
        return DataArray(self.system[system].frequencies, name='Frequency', units='Hz')


    def inphase(self, system=0):
        return self.data[self._inphaseIndices(system)]


    def inphaseStd(self, system=0):
        return self.std[self._inphaseIndices(system)]

    # @property
    # def nFrequencies(self):
    #     return int32(0.5*self.nChannelsPerSystem)

    def predictedInphase(self, system=0):
        return self.predicted_data[self._inphaseIndices(system)]

    def predictedQuadrature(self, system=0):
        return self.predicted_data[self._quadratureIndices(system)]

    def quadrature(self, system=0):
        return self.data[self._quadratureIndices(system)]

    def quadratureStd(self, system=0):
        return self.std[self._quadratureIndices(system)]

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

    def set_priors(self, relative_error_prior=None, additive_error_prior=None, data_prior=None, **kwargs):

        super().set_priors(relative_error_prior, additive_error_prior, data_prior, **kwargs)


    def set_predicted_data_posterior(self):
        if self.predicted_data.hasPrior:
            freqs = log10(self.frequencies())
            xbuf = 0.05*(freqs[-1] - freqs[0])
            xbins = DataArray(logspace(freqs[0]-xbuf, freqs[-1]+xbuf, 200), freqs.name, freqs.units)

            data = log10(self.data[self.active])
            a = data.min()
            b = data.max()
            buf = 0.5*(b - a)
            ybins = DataArray(logspace(a-buf, b+buf, 200), data.name, data.units)

            mesh = RectilinearMesh2D(x_edges=xbins, x_log=10, y_edges=ybins, y_log=10)
            self.predicted_data.posterior = Histogram(mesh=mesh)


    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        grp = super().createHdf(parent, name, withPosterior, add_axis, fillvalue)
        # self.calibration.createHdf(grp, 'calibration', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        self.system[0].toHdf(grp, 'sys')

        if add_axis is not None:
            grp.attrs['repr'] = 'FdemData'

        return grp

    @classmethod
    def fromHdf(cls, grp, index=None, **kwargs):
        """ Reads the object from a HDF group """

        if not 'Point' in grp.attrs['repr']:
            assert index is not None, ValueError("Data saved as a dataset, specify an index")
            # from ..dataset.FdemData import FdemData
            # return FdemData.fromHdf(grp, **kwargs)

        system = FdemSystem.fromHdf(grp['sys'])
        out = super(FdemDataPoint, cls).fromHdf(grp, index, system=system)

        return out

    def calibrate(self, Predicted=True):
        """ Apply calibration factors to the data point """
        # Make complex numbers from the data
        if (Predicted):
            tmp = cf.mergeComplex(self._predicted_data)
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
        tmp[:] = G * exp(1j * Phi) * tmp + Bi + (1j * Bq)

        # Split the complex numbers back out
        if (Predicted):
            self._predicted_data[:] = cf.splitComplex(tmp)
        else:
            self._data[:] = cf.splitComplex(tmp)


    def plot(self, title='Frequency Domain EM Data', system=0,  with_error_bars=True, **kwargs):
        """ Plot the Inphase and Quadrature Data

        Parameters
        ----------
        title : str
            Title of the plot
        system : int
            If multiple system are present, select which one
        with_error_bars : bool
            Plot vertical lines representing 1 standard deviation

        See Also
        --------
        matplotlib.pyplot.errorbar : For more keyword arguements

        Returns
        -------
        out : matplotlib.pyplot.ax
            Figure axis

        """
        ax = kwargs.pop('ax', plt.gca())
        cp.pretty(ax)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Frequency domain data (ppm)')
        ax.set_title(title)

        inColor = kwargs.pop('incolor', cp.wellSeparated[0])
        quadColor = kwargs.pop('quadcolor', cp.wellSeparated[1])
        im = kwargs.pop('inmarker', 'v')
        qm = kwargs.pop('quadmarker', 'o')
        kwargs['markersize'] = kwargs.pop('markersize', 7)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor', 'k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth', 1.0)
        kwargs['alpha'] = kwargs.pop('alpha', 0.8)
        kwargs['linestyle'] = kwargs.pop('linestyle', 'none')
        kwargs['linewidth'] = kwargs.pop('linewidth', 2)

        xscale = kwargs.pop('xscale','log')
        yscale = kwargs.pop('yscale','log')

        f = self.frequencies(system)

        if with_error_bars:
            ax.errorbar(f, self.inphase(system), yerr=self.inphaseStd(system),
                marker=im, color=inColor, markerfacecolor=inColor, label='In-Phase', **kwargs)

            ax.errorbar(f, self.quadrature(system), yerr=self.quadratureStd(system),
                marker=qm, color=quadColor, markerfacecolor=quadColor, label='Quadrature', **kwargs)
        else:
            ax.plot(f, log10(self.inphase(system)),
                marker=im, color=inColor, markerfacecolor=inColor, label='In-Phase', **kwargs)

            ax.plot(f, log10(self.quadrature(system)),
                marker=qm, color=quadColor, markerfacecolor=quadColor, label='Quadrature', **kwargs)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.legend(fontsize=8)

        return ax


    def plot_predicted(self, title='Frequency Domain EM Data', system=0, **kwargs):
        """ Plot the predicted Inphase and Quadrature Data

        Parameters
        ----------
        title : str
            Title of the plot
        system : int
            If multiple system are present, select which one

        See Also
        --------
        matplotlib.pyplot.semilogx : For more keyword arguements

        Returns
        -------
        out : matplotlib.pyplot.ax
            Figure axis

        """
        ax = kwargs.pop('ax', plt.gca())
        cp.pretty(ax)

        labels = kwargs.pop('labels', True)

        if (labels):
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Data (ppm)')
            ax.set_title(title)

        c = kwargs.pop('color', cp.wellSeparated[3])
        lw = kwargs.pop('linewidth', 2)
        a = kwargs.pop('alpha', 0.7)

        xscale = kwargs.pop('xscale','log')
        yscale = kwargs.pop('yscale','log')

        ax.semilogx(self.frequencies(system), self.predictedInphase(system), color=c, linewidth=lw, alpha=a, **kwargs)
        ax.semilogx(self.frequencies(system), self.predictedQuadrature(system), color=c, linewidth=lw, alpha=a, **kwargs)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        return ax

    def update_posteriors(self):
        super().update_posteriors()

    #     if self.predicted_data.hasPosterior:
    #         x = self.frequencies()
    #         self.predicted_data.posterior.update_with_line(x, self.predictedInphase())
    #         self.predicted_data.posterior.update_with_line(x, self.predictedQuadrature())


    def updateSensitivity(self, model):
        """ Compute an updated sensitivity matrix based on the one already containined in the FdemDataPoint object  """
        self.J = self.sensitivity(model)




    def forward(self, mod):
        """ Forward model the data from the given model """
        assert isinstance(mod, Model), TypeError("Invalid model class for forward modeling [1D]")
        self._forward1D(mod)


    def sensitivity(self, mod, **kwargs):
        """ Compute the sensitivty matrix for the given model """
        assert isinstance(mod, Model), TypeError("Invalid model class for sensitivity matrix [1D]")
        return DataArray(self._sensitivity1D(mod), 'Sensitivity', r'$\frac{ppm.m}{S}$')

    def fm_dlogc(self, mod):
        self.forward(mod)
        self.sensitivity(mod)

    def _forward1D(self, mod):
        """ Forward model the data from a 1D layered earth model """
        assert isinf(mod.mesh.edges[-1]), ValueError('mod.edges must have last entry be infinity for forward modelling.')
        for i, s in enumerate(self.system):
            tmp = fdem1dfwd(s, mod, self.z[0])
            self._predicted_data[:self.nFrequencies[i]] = tmp.real
            self._predicted_data[self.nFrequencies[i]:] = tmp.imag


    def _sensitivity1D(self, mod):
        """ Compute the sensitivty matrix for a 1D layered earth model """
        # Re-arrange the sensitivity matrix to Real:Imaginary vertical
        # concatenation
        self._sensitivity_matrix = DataArray((self.nChannels, mod.mesh.nCells.item()), 'Sensitivity', r'$\frac{ppm.m}{S}$')

        for j, s in enumerate(self.system):
            Jtmp = fdem1dsen(s, mod, self.z.item())
            self._sensitivity_matrix[:self.nFrequencies[j], :] = Jtmp.real
            self._sensitivity_matrix[self.nFrequencies[j]:, :] = Jtmp.imag

        return self.sensitivity_matrix


    def Isend(self, dest, world, **kwargs):

        if not 'system' in kwargs:
            myMPI.Isend(self.nSystems, dest=dest, world=world)
            for i in range(self.nSystems):
                self.system[i].Isend(dest=dest, world=world)

        super().Isend(dest, world)

        self.data.Isend(dest, world)
        self.predicted_data.Isend(dest, world)

    @classmethod
    def Irecv(cls, source, world, **kwargs):

        if not 'system' in kwargs:
            nSystems = myMPI.Irecv(source=source, world=world)
            kwargs['system'] = [FdemSystem.Irecv(source=source, world=world) for i in range(nSystems)]

        out = super(FdemDataPoint, cls).Irecv(source, world, **kwargs)

        out._data = StatArray.Irecv(source, world)
        out._predicted_data = StatArray.Irecv(source, world)

        return out
