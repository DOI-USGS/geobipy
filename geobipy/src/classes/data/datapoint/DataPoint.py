from cached_property import cached_property
from ...pointcloud.Point import Point
from ....classes.core import StatArray
from ....base import customFunctions as cf
from ....base import customPlots as cP
from ....base import MPI as myMPI
import numpy as np
import matplotlib.pyplot as plt

class DataPoint(Point):
    """Class defines a data point.

    Contains an easting, northing, height, elevation, observed and predicted data, and uncertainty estimates for the data.

    DataPoint(x, y, z, elevation, nChannels, data, std, dataUnits)

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
    dataUnits : str, optional
        Units of the data.  Default is "ppm".
    channelNames : list of str, optional
        Names of each channel of length sum(nChannelsPerSystem)

    """

    def __init__(self, nChannelsPerSystem=1, x=0.0, y=0.0, z=0.0, elevation=None, data=None, std=None, predictedData=None, dataUnits=None, channelNames=None):
        """ Initialize the Data class """

        Point.__init__(self, x, y, z)

        self.nSystems = np.size(nChannelsPerSystem)

        # Number of Channels
        self.nChannelsPerSystem = np.atleast_1d(np.asarray(nChannelsPerSystem))
        self._systemOffset = np.concatenate([[0], np.cumsum(self.nChannelsPerSystem)])

        # StatArray of data
        if not elevation is None:
            # assert np.size(elevation) == 1, ValueError("elevation must be single float")
            self._elevation = StatArray.StatArray(elevation, "Elevation", "m")
        else:
            self._elevation = StatArray.StatArray(1, "Elevation", "m")

        # StatArray of data
        if not data is None:
            assert np.size(data) == self.nChannels, ValueError("data must have size {}".format(self.nChannels))
            self._data = StatArray.StatArray(data, "Data", dataUnits)
        else:
            self._data = StatArray.StatArray(self.nChannels, "Data", dataUnits)

        # # Index to non NaN values
        # self.active = self.getActiveData()

        # StatArray of Standard Deviations
        if not std is None:
            assert np.size(std) == self.nChannels, ValueError("std must have size {}".format(self.nChannels))
            assert np.all(std[self.active] > 0.0), ValueError("Cannot assign standard deviations that are <= 0.0.")
            self._std = StatArray.StatArray(std, "Standard Deviation", dataUnits)
        else:
            self._std = StatArray.StatArray(np.ones(self.nChannels), "Standard Deviation", dataUnits)


        # Create predicted data
        if not predictedData is None:
            assert np.size(predictedData) == self.nChannels, ValueError("predictedData must have size {}".format(self.nChannels))
            self._predictedData = StatArray.StatArray(predictedData, "Predicted Data", dataUnits)
        else:
            self._predictedData = StatArray.StatArray(self.nChannels, "Predicted Data", dataUnits)

        # Assign the channel names
        if channelNames is None:
            self._channelNames = ['Channel {}'.format(i) for i in range(self.nChannels)]
        else:
            assert len(channelNames) == self.nChannels, Exception("Length of channelNames must equal total number of channels {}".format(self.nChannels))
            self._channelNames = channelNames


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, values):
        assert np.size(values) == self.nChannelsPerSystem, ValueError("data must have size {}".format(self.nChannelsPerSystem))
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
        return StatArray.StatArray(self._predictedData - self._data, name="$\\mathbf{Fm} - \\mathbf{d}_{obs}$", units=self._data.units)

    @property
    def elevation(self):
        return self._elevation

    @property
    def nActiveChannels(self):
        return self.active.size

    @property
    def nChannels(self):
        return np.sum(self.nChannelsPerSystem)

    @property
    def predictedData(self):
        return self._predictedData

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, values):
        assert np.size(values) == self.nChannels, ValueError("std must have size {}".format(self.nChannels))
        assert np.all(values[self.active] > 0.0), ValueError("Cannot assign standard deviations that are <= 0.0.")
        self._std[:] = values


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
        return np.s_[self._systemOffset[system]:self._systemOffset[system+1]]


    @cached_property
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
        return self._predictedData.probability(i=self.active, log=log)


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


    def summary(self, out=False):
        """ Print a summary of the EMdataPoint """
        msg = ('Data Point: \n'
               'Channel Names {} \n'
               'x: {} \n'
               'y: {} \n'
               'z: {} \n'
               'elevation: {} \n'
               'Number of active channels: {} \n'
               '{} {} {}').format(self._channelNames, self.x, self.y, self.z, self.elevation, self.nActiveChannels, self._data[self.active].summary(True), self._predictedData[self.active].summary(True), self._std[self.active].summary(True))
        if (out):
            return msg
        print(msg)


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

