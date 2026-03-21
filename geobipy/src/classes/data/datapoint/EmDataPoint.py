import numpy as np
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
    __slots__ = ('_channels_per_system', '_system', '_total_field')

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=None,
                       components=None, channels_per_system=None,
                       data=None, std=None, predicted_data=None,
                       channel_names=None,
                       line_number=0.0, fiducial=0.0, total_field=False, **kwargs):

        self.channels_per_system = channels_per_system
        self.total_field = total_field

        super().__init__(x = x, y = y, z = z, elevation = elevation,
                         components=components,
                         data = data, std = std, predicted_data = predicted_data,
                         channel_names=channel_names, line_number=line_number, fiducial=fiducial, **kwargs)

    def __deepcopy__(self, memo={}):

        out = super().__deepcopy__(memo)

        out._channels_per_system = deepcopy(self.channels_per_system, memo)
        out._total_field = deepcopy(self._total_field, memo)
        out.system = self._system

        return out

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
    def total_field(self) -> bool:
        return self._total_field

    @total_field.setter
    def total_field(self, value: bool):
        self._total_field = value

    @property
    def n_channels(self):
        return sum(self.n_components * self.channels_per_system)

    @property
    def data_channels_per_system(self):
        out = self.channels_per_system.copy()
        if not self.amplitude_data:
            out *= self.n_components
        return out

    @property
    def n_data_channels(self):
        return sum(self.data_channels_per_system)

    @property
    def n_systems(self):
        return size(self.channels_per_system)

    @DataPoint.data.getter
    def data(self):
        if self.amplitude_data:
            self._data[...] = 0.0
            for j in range(self.n_systems):
                isys = self._indices(0, j)
                for i in range(self.n_components):
                    ic = self._indices(i, j)
                    # Compute Sum(Pc + Sc) for c in x, y, z
                    if self.total_field:
                        tmp = self.primary_field[i] + self.secondary_field[ic]
                    else:
                        tmp = self.secondary_field[ic]
                    self._data[isys] += tmp**2.0
            self._data[...] = np.sqrt(self._data)
        else:
            for j in range(self.n_systems):
                for i in range(self.n_components):
                    ic = self._indices(i, j)
                    self._data[ic] = self.secondary_field[ic]
                    if self.total_field:
                        self._data[ic] += self.primary_field[i]

        return self._data

    @DataPoint.predicted_data.setter
    def predicted_data(self, values):
        if values is None:
            values = self.n_data_channels
        else:
            assert size(values) == self.n_data_channels, ValueError("Size of predicted_data must equal total number of time channels {}".format(self.n_data_channels))
        self._predicted_data = StatArray(values, "Predicted Data", self.units)

    @DataPoint.predicted_data.getter
    def predicted_data(self):
        if self.amplitude_data:
            self._predicted_data[...] = 0.0
            for j in range(self.n_systems):
                isys = self._indices(0, j)
                for i in range(self.n_components):
                    ic = self._indices(i, j)
                    # Compute Sum(Pc + Sc) for c in x, y, z
                    if self.total_field:
                        tmp = self.predicted_primary_field[i] + self.predicted_secondary_field[ic]
                    else:
                        tmp = self.predicted_secondary_field[ic]
                    self._predicted_data[isys] += tmp**2.0
            self._predicted_data[...] = np.sqrt(self._predicted_data)
        else:
            for j in range(self.n_systems):
                for i in range(self.n_components):
                    ic = self._indices(i, j)
                    self._predicted_data[ic] = self.predicted_secondary_field[ic]
                    if self.total_field:
                        self._predicted_data[ic] += self.predicted_primary_field[i]
        return self._predicted_data

    @DataPoint.std.getter
    def std(self):
        assert np.min(self.relative_error) > 0.0, ValueError("relative_error must be > 0.0")
        for i in range(self.n_systems):
            j = self.system_indices[i]
            self._std[j] = np.sqrt((self.relative_error[i] * self.data[j])**2 + (self.additive_error[i]**2))

        return self._std

    @property
    def system(self):
        return self._system

    @property
    def empty_halfspace(self):
        mesh = RectilinearMesh1D(edges=DataArray(asarray([0.0, inf]), 'Depth', 'm'))
        conductivity = DataArray(mesh.nCells.item(), 'Conductivity', r'$\frac{S}{m}$')
        magnetic_susceptibility = DataArray(mesh.nCells.item(), "Magnetic Susceptibility", r"$\kappa$")
        magnetic_permeability = DataArray(mesh.nCells.item(), "Magnetic Permeability", "$\frac{H}{m}$")

        out = Model(mesh=mesh, values=conductivity)

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

        model = self.empty_halfspace

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
        c = DataArray(logspace(minConductivity, maxConductivity, nSamples), 'Conductivity', '$S/m$')
        PhiD = DataArray(size(c), 'Normalized Data Misfit', '')

        model = self.empty_halfspace

        for i in range(size(c)):
            model.values[0] = c[i]
            self.forward(model)
            PhiD[i] = self.data_misfit()

        plt.loglog(c, PhiD, **kwargs)
        plt.xlabel(c.getNameUnits())
        plt.ylabel('Data misfit')


    def Isend(self, dest, world, **kwargs):

        world.isend(self.total_field, dest=dest).wait()

        super().Isend(dest, world)

    @classmethod
    def Irecv(cls, source, world, **kwargs):

        kwargs['total_field'] = world.irecv(source=source).wait()

        out = super(EmDataPoint, cls).Irecv(source, world, **kwargs)

        return out