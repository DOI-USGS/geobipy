from numpy import argmin, asarray, atleast_1d, cumsum
from numpy import hstack, inf, int32, isnan, log10, logspace, nan
from numpy import r_, size, sum, zeros
from numpy import all as npall

from .DataPoint import DataPoint
from ...core.DataArray import DataArray
from ...statistics.StatArray import StatArray
from ...mesh.RectilinearMesh1D import RectilinearMesh1D
from ...statistics.Histogram import Histogram
from ...model.Model import Model
# from ...statistics.Histogram2D import Histogram2D
from ...statistics.Distribution import Distribution
from ....base import utilities as cf
from ....base import plotting as cP
from copy import deepcopy

import matplotlib.pyplot as plt


class EmDataPoint(DataPoint):
    """Abstract EmDataPoint Class

    This is an abstract base class for TdemDataPoint and FdemDataPoint classes

    See Also
    ----------
    geobipy.src.classes.data.datapoint.FdemDataPoint
    geobipy.src.classes.data.datapoint.TdemDataPoint

    """
    __slots__ = ('_channels_per_system', '_system')

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=None,
                       components=None, channels_per_system=None,
                       data=None, std=None, predicted_data=None,
                       channel_names=None,
                       line_number=0.0, fiducial=0.0, **kwargs):

        super().__init__(x = x, y = y, z = z, elevation = elevation,
                         data = data, std = std, predicted_data = predicted_data,
                         channel_names=channel_names, line_number=line_number, fiducial=fiducial, **kwargs)

    @property
    def active(self):
        """Gets the indices to the observed data values that are not NaN

        Returns
        -------
        out : array of ints
            Indices into the observed data that are not NaN

        """
        d = self.data.copy()
        d[d <= 0.0] = nan
        return ~isnan(d)

    @property
    def channels_per_system(self):
        return self._channels_per_system

    @channels_per_system.setter
    def channels_per_system(self, values):
        if values is None:
            values = zeros(1, dtype=int32)
        else:
            values = atleast_1d(asarray(values, dtype=int32)).copy()

        self._channels_per_system = values

    @property
    def n_data_channels(self):
        return self.nChannels

    @property
    def components(self):
        m = ('x', 'y', 'z')
        return [m[x] for x in self._components]

    @components.setter
    def components(self, values):

        m = {'x': 0,
             'y': 1,
             'z': 2}

        if values is None:
            values = ['z']
        else:

            if isinstance(values, str):
                values = [values]

            assert all([isinstance(x, str) for x in values]), TypeError('components must be list of str')

        self._components = asarray([m[x] for x in values], dtype=int32)

    @DataPoint.data.setter
    def data(self, values):
        if values is None:
            values = self.n_data_channels
        else:
            assert size(values) == self.n_data_channels, ValueError(f"data must have size {self.n_data_channels} not {size(values)}")
        self._data = DataArray(values, "Data", self.units)

    @property
    def n_components(self):
        return size(self.components)

    @property
    def nChannels(self):
        return sum(self.channels_per_system)

    @property
    def nSystems(self):
        return size(self.channels_per_system)

    @DataPoint.predicted_data.setter
    def predicted_data(self, values):
        if values is None:
            values = self.n_data_channels
        else:
            assert size(values) == self.n_data_channels, ValueError("Size of predicted_data must equal total number of time channels {}".format(self.n_data_channels))
        self._predicted_data = StatArray(values, "Predicted Data", self.units)

    @DataPoint.std.setter
    def std(self, values):
        if values is None:
            values = self.n_data_channels
        else:
            assert size(values) == self.n_data_channels, ValueError(f"data must have size {self.n_data_channels} not {size(values)}")
        self._std = DataArray(values, "Data", self.units)

    @property
    def system(self):
        return self._system

    @property
    def systemOffset(self):
        return hstack([0, cumsum(self.channels_per_system)])

    @property
    def new_model(self):
        mesh = RectilinearMesh1D(edges=DataArray(asarray([0.0, inf]), 'Depth', 'm'))
        conductivity = DataArray(mesh.nCells.item(), 'Conductivity', r'$\frac{S}{m}$')
        magnetic_susceptibility = DataArray(mesh.nCells.item(), "Magnetic Susceptibility", r"$\kappa$")
        magnetic_permeability = DataArray(mesh.nCells.item(), "Magnetic Permeability", "$\frac{H}{m}$")

        out = Model(mesh=mesh, values=conductivity)
        # out.setattr('magnetic_susceptibility', magnetic_susceptibility)
        # out.setattr('magnetic_permeability', magnetic_permeability)

        return out

    def find_best_halfspace(self, minConductivity=1e-10, maxConductivity=1e2, nSamples=1000):
        """Computes the best value of a half space that fits the data.

        Carries out a brute force search of the halfspace conductivity that best fits the data.
        The profile of data misfit vs halfspace conductivity is not quadratic, so a bisection will not work.

        Parameters
        ----------
        minConductivity : float, optional
            The minimum conductivity to search over
        maxConductivity : float, optional
            The maximum conductivity to search over
        nSamples : int, optional
            The number of values between the min and max

        Returns
        -------
        out : float64
            The best fitting log10 conductivity for the half space

        """
        assert maxConductivity > minConductivity, ValueError("Maximum conductivity must be greater than the minimum")
        minConductivity = log10(minConductivity)
        maxConductivity = log10(maxConductivity)

        c = logspace(minConductivity, maxConductivity, nSamples)

        PhiD = zeros(nSamples)

        model = self.new_model

        for i in range(nSamples):
            model.values[0] = c[i]
            self.forward(model)
            PhiD[i] = self.data_misfit()

        i = argmin(PhiD)
        model.values[0] = c[i]
        return model


        # # Generate new calibration errors
        #     self.calibration[:] = self.calibration.proposal.rng(1)
        #     # Update the mean of the proposed calibration errors
        #     self.calibration.proposal.mean[:] = self.calibration

        #     self.calibrate()

    def plot_halfspace_responses(self, minConductivity=-4.0, maxConductivity=2.0, nSamples=100, **kwargs):
        """Plots the reponses of different half space models.

        Parameters
        ----------
        minConductivity : float, optional
            The minimum log10 conductivity to search over
        maxConductivity : float, optional
            The maximum log10 conductivity to search over
        nInc : int, optional
            The number of increments between the min and max

        """

        # tmp = deepcopy(self)
        c = DataArray(logspace(min, max, nSamples), 'Conductivity', '$S/m$')
        PhiD = DataArray(size(c), 'Normalized Data Misfit', '')

        model = self.new_model

        for i in range(size(c)):
            model.values[0] = c[i]
            self.forward(model)
            PhiD[i] = self.data_misfit()

        plt.loglog(c, PhiD, **kwargs)
        plt.xlabel(c.getNameUnits())
        plt.ylabel('Data misfit')