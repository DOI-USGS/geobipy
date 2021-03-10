from cached_property import cached_property
from copy import deepcopy
from ...pointcloud.Point import Point
from ....classes.core import StatArray
from ....base import customFunctions as cf
from ....base import customPlots as cP
from ....base import MPI as myMPI
from ...statistics.Histogram2D import Histogram2D
import numpy as np
import matplotlib.pyplot as plt

class DataPoint(Point):
    """Class defines a data point.

    Contains an easting, northing, height, elevation, observed and predicted data, and uncertainty estimates for the data.

    DataPoint(x, y, z, elevation, nChannels, data, std, units)

    Parameters
    ----------
    nChannelsPerSystem : int or array_like
        Number of data channels in the data
        * If int, a single acquisition system is assumed.
        * If array_like, each entry is the number of channels for each system.
    x : float
        Easting co-ordinate of the data point
    y : float
        Northing co-ordinate of the data point
    z : float
        Height above ground of the data point
    elevation : float, optional
        Elevation from sea level of the data point
    data : geobipy.StatArray or array_like, optional
        Data values to assign the data of length sum(nChannelsPerSystem).
        * If None, initialized with zeros.
    std : geobipy.StatArray or array_like, optional
        Estimated uncertainty standard deviation of the data of length sum(nChannelsPerSystem).
        * If None, initialized with ones if data is None, else 0.1*data values.
    predictedData : geobipy.StatArray or array_like, optional
        Predicted data values to assign the data of length sum(nChannelsPerSystem).
        * If None, initialized with zeros.
    units : str, optional
        Units of the data.  Default is "ppm".
    channelNames : list of str, optional
        Names of each channel of length sum(nChannelsPerSystem)

    """

    def __init__(self, nChannelsPerSystem=1, x=0.0, y=0.0, z=0.0, elevation=None, data=None, std=None, predictedData=None, units=None, channelNames=None, lineNumber=0.0, fiducial=0.0):
        """ Initialize the Data class """

        super().__init__(x, y, z)

        # Number of Channels
        self._nChannelsPerSystem = np.atleast_1d(np.asarray(nChannelsPerSystem))

        self.elevation = elevation

        self.units = units

        # StatArray of data
        self.data = data

        self.std = std

        self.predictedData = predictedData

        self.lineNumber = lineNumber

        self.fiducial = fiducial

        self.channelNames = channelNames

        self.errorPosterior = None


    def __deepcopy__(self, memo={}):

        out = super().__deepcopy__(memo)

        out._nChannelsPerSystem = deepcopy(self._nChannelsPerSystem, memo)

        out._elevation = deepcopy(self.elevation, memo)
        out._units = deepcopy(self.units, memo)
        out._data = deepcopy(self.data, memo)
        out._std = deepcopy(self.std, memo)
        out._predictedData = deepcopy(self.predictedData, memo)
        out._lineNumber = deepcopy(self.lineNumber, memo)
        out._fiducial = deepcopy(self.fiducial, memo)
        out._channelNames = deepcopy(self.channelNames, memo)
        out._relErr = deepcopy(self.relErr, memo)
        out._addErr = deepcopy(self.addErr, memo)
        out._errorPosterior = deepcopy(self.errorPosterior, memo)

        return out

    @property
    def additive_error(self):
        return self._addErr

    @property
    def addErr(self):
        return self._addErr

    @addErr.setter
    def addErr(self, values):
        if values is None:
            self._addErr = StatArray.StatArray(self.nSystems, '$\epsilon_{Additive}$', self.units)
        else:
            assert np.size(values) == self.nSystems, ValueError("additiveError must have length {}".format(self.nSystems))
            assert np.asarray(values).dtype.kind == 'f', ValueError("additive_error must be floats")
            self._addErr = StatArray.StatArray(values, '$\epsilon_{Additive}$', self.units)

    @property
    def channelNames(self):
        return self._channelNames

    @channelNames.setter
    def channelNames(self, values):
        if values is None:
            self._channelNames = ['Channel {}'.format(i) for i in range(self.nChannels)]
        else:
            assert len(values) == self.nChannels, Exception("Length of channelNames must equal total number of channels {}".format(self.nChannels))
            self._channelNames = values


    @property
    def systemOffset(self):
        return np.concatenate([[0], np.cumsum(self.nChannelsPerSystem)])

    @property
    def fiducial(self):
        return self._fiducial


    @fiducial.setter
    def fiducial(self, value):
        self._fiducial = StatArray.StatArray(value, 'fiducial')


    @property
    def lineNumber(self):
        return self._lineNumber


    @lineNumber.setter
    def lineNumber(self, value):
        self._lineNumber = StatArray.StatArray(value, 'Line number')

    @property
    def nChannelsPerSystem(self):
        return self._nChannelsPerSystem

    @property
    def nSystems(self):
        return np.size(self.nChannelsPerSystem)

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if values is None:
            self._units = ""
        else:
            assert isinstance(value, str), TypeError('units must have type str')
            self._units = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, values):

        self._data = StatArray.StatArray(self.nChannels, "Data", self.units)

        if not values is None:
            assert np.size(values) == self.nChannels, ValueError("data must have size {}".format(self.nChannels))
            self._data[:] = values

    @property
    def deltaD(self):
        """Get the difference between the predicted and observed data,

        .. math::
            \delta \mathbf{d} = \mathbf{d}^{pre} - \mathbf{d}^{obs}.

        Returns
        -------
        out : StatArray
            The residual between the active observed and predicted data
            with size equal to the number of active channels.

        """
        return StatArray.StatArray(self._predictedData - self._data, name="$\\mathbf{Fm} - \\mathbf{d}_{obs}$", units=self.units)

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, value):
        if value is None:
            self._elevation = StatArray.StatArray(1, "Elevation", "m")
        else:
            self._elevation = StatArray.StatArray(value, "Elevation", "m")


    @property
    def nActiveChannels(self):
        return self.active.size

    @property
    def nChannels(self):
        return np.sum(self.nChannelsPerSystem)


    @property
    def predictedData(self):
        """The predicted data. """
        return self._predictedData


    @predictedData.setter
    def predictedData(self, values):
        if values is None:
            self._predictedData = StatArray.StatArray(self.nChannels, "Predicted Data", self.units)
        else:
            assert np.size(values) == self.nChannels, ValueError("predictedData must have size {}".format(self.nChannels))
            self._predictedData = StatArray.StatArray(values, "Predicted Data", self.units)


    @property
    def relative_error(self):
        return self._relErr

    @property
    def relErr(self):
        return self._relErr

    @relErr.setter
    def relErr(self, values):
        if values is None:
            self._relErr = StatArray.StatArray(self.nSystems, '$\epsilon_{Relative}x10^{2}$', '%')
        else:
            assert np.size(values) == self.nSystems, ValueError("relativeError must have length {}".format(self.nSystems))
            assert np.asarray(values).dtype.kind == 'f', ValueError("relative_error must be floats")
            self._relErr = StatArray.StatArray(self.nSystems, '$\epsilon_{Relative}x10^{2}$', '%') + values


    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, values):

        self._std = StatArray.StatArray(np.ones(self.nChannels), "Standard Deviation", self.units)

        if not values is None:
            assert np.size(values) == self.nChannels, ValueError("std must have size {}".format(self.nChannels))
            assert np.all(values[self.active] > 0.0), ValueError("Cannot assign standard deviations that are <= 0.0.")
            self._std[:] = values


    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):

        if value is None:
            self._units = ""
        else:
            assert isinstance(value, str), TypeError("units must have type str")
            self._units = value


    def generate_noise(self, additive_error, relative_error):

        std = np.sqrt(additive_error**2.0 + (relative_error * self.predictedData)**2.0)
        return np.random.randn(self.nChannels) * std



    def weightingMatrix(self, power=1.0):
        """Return a diagonal data weighting matrix of the reciprocated data standard deviations."""
        return np.diag(self.std[self.active]**-power)


    def _systemIndices(self, system=0):
        """The slice indices for the requested system.

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
        return np.s_[self.systemOffset[system]:self.systemOffset[system+1]]


    @property
    def active(self):
        """Gets the indices to the observed data values that are not NaN

        Returns
        -------
        out : array of ints
            Indices into the observed data that are not NaN

        """
        return cf.findNotNans(self.data)


    def likelihood(self, log):
        """Compute the likelihood of the current predicted data given the observed data and assigned errors

        Returns
        -------
        out : np.float64
            Likelihood of the data point

        """
        return self.predictedData.probability(i=self.active, log=log)


    def dataMisfit(self, squared=False):
        """Compute the :math:`L_{2}` norm squared misfit between the observed and predicted data

        .. math::
            \| \mathbf{W}_{d} (\mathbf{d}^{obs}-\mathbf{d}^{pre})\|_{2}^{2},

        where :math:`\mathbf{W}_{d}` are the reciprocal data errors.

        Parameters
        ----------
        squared : bool
            Return the squared misfit.

        Returns
        -------
        out : np.float64
            The misfit value.

        """
        assert not any(self._std[self.active] == 0.0), ValueError('Cannot compute the misfit when the data standard deviations are zero.')
        tmp2 = self._std[self.active]**-1.0
        PhiD = np.float64(np.sum((cf.Ax(tmp2, self.deltaD[self.active]))**2.0, dtype=np.float64))
        return PhiD if squared else np.sqrt(PhiD)


    def scaleJ(self, Jin, power=1.0):
        """ Scales a matrix by the errors in the given data

        Useful if a sensitivity matrix is generated using one data point, but must be scaled by the errors in another.

        Parameters
        ----------
        Jin : array_like
            2D array representing a matrix
        power : float
            Power to raise the error level too. Default is -1.0

        Returns
        -------
        Jout : array_like
            Matrix scaled by the data errors

        Raises
        ------
        ValueError
            If the number of rows in Jin do not match the number of active channels in the datapoint

        """
        assert Jin.shape[0] == self.nActiveChannels, ValueError("Number of rows of Jin must match the number of active channels in the datapoint {}".format(self.nActiveChannels))

        Jout = np.zeros(Jin.shape)
        Jout[:, :] = Jin * (np.repeat(self._std[self.active, np.newaxis]**-power, Jout.shape[1], 1))
        return Jout

    @property
    def summary(self):
        """ Print a summary of the EMdataPoint """
        msg = ('Data Point: \n'
               'Channel Names {} \n'
               'x: {} \n'
               'y: {} \n'
               'z: {} \n'
               'elevation: {} \n'
               'Number of active channels: {} \n'
               '{} {} {}').format(self._channelNames, self.x, self.y, self.z, self.elevation, self.nActiveChannels, self._data[self.active].summary, self._predictedData[self.active].summary, self._std[self.active].summary)
        return msg


    def updateErrors(self, relativeErr, additiveErr):
        """Updates the data errors.

        Updates the standard deviation of the data errors using the following model

        .. math::
            \sqrt{(\mathbf{\epsilon}_{rel} \mathbf{d}^{obs})^{2} + \mathbf{\epsilon}^{2}_{add}},

        where :math:`\mathbf{\epsilon}_{rel}` is the relative error, a percentage fraction and :math:`\mathbf{\epsilon}_{add}` is the additive error.

        If the predicted data have been assigned a multivariate normal distribution, the variance of that distribution is also updated as the squared standard deviations.

        Parameters
        ----------
        relativeErr : float or array_like
            A fraction percentage that is multiplied by the observed data.
        additiveErr : float or array_like
            An absolute value of additive error.

        Raises
        ------
        ValueError
            If relativeError is <= 0.0
        ValueError
            If additiveError is <= 0.0

        """
        relativeErr = np.atleast_1d(relativeErr)
        additiveErr = np.atleast_1d(additiveErr)
        assert all(relativeErr > 0.0), ValueError("relativeErr must be > 0.0")
        assert all(additiveErr > 0.0), ValueError("additiveErr must be > 0.0")

        tmp = (relativeErr * self.data)**2.0 + additiveErr**2.0
        self._std[:] = np.sqrt(tmp)

        if self._predictedData.hasPrior:
            self._predictedData.prior.variance[np.diag_indices(self.nActiveChannels)] = tmp[self.active]


    def Isend(self, dest, world):
        myMPI.Isend(self.nChannelsPerSystem, dest=dest, world=world)
        tmp = np.hstack([self.x, self.y, self.z, self.elevation])
        myMPI.Isend(tmp, dest=dest, world=world)
        self._data.Isend(dest, world)
        self._std.Isend(dest, world)
        self._predictedData.Isend(dest, world)
        world.isend(self._channelNames, dest=dest)



    def Irecv(self, source, world):
        ncps = myMPI.Irecv(source=source, world=world)
        tmp = myMPI.Irecv(source=source, world=world)
        x = StatArray.StatArray(0)
        d = x.Irecv(source, world)
        s = x.Irecv(source, world)
        p = x.Irecv(source, world)
        cn = world.irecv(source = 0).wait()

        return DataPoint(ncps, tmp[0], tmp[1], tmp[2], tmp[3], d, s, p, channelNames=cn)

