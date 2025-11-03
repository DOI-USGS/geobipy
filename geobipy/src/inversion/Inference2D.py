import os
from pathlib import Path
from numpy import float64, int32, s_, integer
from numpy import arange, argmax, argsort, argwhere, asarray, divide, empty, full, isnan, linspace, logspace
from numpy import log10, maximum, minimum, mean, min, max, nan, nanmin, nanmax, ones, repeat, sum, sqrt
from numpy import searchsorted, size, shape, sort, squeeze, unique, where, zeros
from numpy import all as npall
from numpy.random import Generator

import h5py
from copy import deepcopy
from cached_property import cached_property
from datetime import timedelta
from ..classes.core.myObject import myObject
from ..classes.statistics.StatArray import StatArray
from ..classes.statistics import get_prng
from ..classes.statistics.Distribution import Distribution
from ..classes.statistics.mixPearson import mixPearson
from ..classes.statistics.Histogram import Histogram
# from ..classes.statistics.Histogram2D import Histogram2D
# from ..classes.statistics.Hitmap2D import Hitmap2D
from ..classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from ..classes.mesh.RectilinearMesh2D import RectilinearMesh2D
from ..classes.data.datapoint.DataPoint import DataPoint
from ..classes.data.dataset.Data import Data
from ..classes.data.dataset.FdemData import FdemData
from ..classes.data.dataset.TdemData import TdemData
from ..classes.data.dataset.TempestData import TempestData
from ..classes.model.Model import Model
from ..base.HDF import hdfRead
from ..base import plotting as cP
from ..base import utilities as cF
from ..base import fileIO as fIO
from ..base.MPI import loadBalance1D_shrinkingArrays
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import (split, join)
from .Inference1D import Inference1D
import progressbar


try:
    from pyvtk import VtkData, UnstructuredGrid, CellData, Scalars
except:
    pass

from numba import (jit, float64)
_numba_settings = {'nopython': True, 'nogil': False, 'fastmath': True, 'cache': True}

class Inference2D(myObject):
    """ Class to define results from EMinv1D_MCMC for a line of data """
    def __init__(self, data, prng, world=None):
        """ Initialize the lineResults """

        self.world = world

        self.data = data
        self.prng = prng

        # if (hdf5_file_path is None): return
        # # assert not system_file_path is None, Exception("Please also specify the path to the system file")
        # self.fName = hdf5_file_path
        # self.directory = split(hdf5_file_path)[0]
        # # self.line_number = float64(os.path.splitext(split(hdf5_file_path)[1])[0])
        # self.hdf_file = None
        # if (hdf5_file is None): # Open the file
        #     self.open(mode, world)
        # else:
        #     self.hdf_file = hdf5_file
        # self._indices = None

    @cached_property
    def acceptance(self):
        return StatArray.fromHdf(self.hdf_file['acceptance_rate'])

    @cached_property
    def additiveError(self):
        """ Get the Additive error of the best data points """
        return StatArray.fromHdf(self.hdf_file['data/additive_error'])

    @property
    def additiveErrorPosteriors(self):
        return self.data.additive_error.posterior

    @cached_property
    def best_halfspace(self, log=None):
        a = log10(asarray(self.hdf_file['model/values/posterior/x/x/data'][:, 0]))
        b = log10(asarray(self.hdf_file['model/values/posterior/x/x/data'][:, -1]))
        return 0.5 * (b + a)

    @cached_property
    def burned_in(self):
        if 'burned_in' in self.hdf_file:
            key = 'burned_in'
        elif 'burnedin' in self.hdf_file:
            key = 'burnedin'
        return StatArray(asarray(self.hdf_file[key]))

    @property
    def data(self):
        """ Get the best data """
        return self._data

    @data.setter
    def data(self, value):
        assert isinstance(value, (Data, DataPoint)), TypeError("data must have type geobipy.Data, instead has type {}".format(type(value)))
        assert value.nPoints > 0, ValueError("Data has no value. nPoints is 0.")
        self._data = value


    @property
    def doi(self):
        if 'doi' in self.hdf_file.keys():
            return StatArray.fromHdf(self.hdf_file['doi'])
        else:
            return self.compute_doi()

    @property
    def easting(self):
        return StatArray.fromHdf(self.hdf_file['data/x'])

    @property
    def northing(self):
        return StatArray.fromHdf(self.hdf_file['data/y'])

    @property
    def depth(self):
        return self.mesh.z.centres

    @cached_property
    def elevation(self):
        """ Get the elevation of the data points """
        return StatArray.fromHdf(self.hdf_file['data/elevation'])

    @property
    def entropy(self):
        return self.parameter_posterior().entropy(axis=1)

    @cached_property
    def fiducials(self):
        """ Get the id numbers of the data points in the line results file """
        return StatArray.fromHdf(self.hdf_file['data/fiducial'])

    @cached_property
    def halfspace(self):
        return StatArray.fromHdf(self.hdf_file['halfspace'])

    @property
    def hdf_file(self):
        return self._hdf_file

    @hdf_file.setter
    def hdf_file(self, value):
        assert isinstance(value, (h5py.File, h5py.Group)), TypeError("hdf_file must have type h5py.File")
        self._hdf_file = value

    @cached_property
    def height(self):
        """Get the height of the observations. """
        return StatArray.fromHdf(self.hdf_file['data/z'])

    @cached_property
    def heightPosterior(self):
        zPosterior = self.getAttribute('height posterior')
        zPosterior.bins.name = 'Relative ' + zPosterior.bins.name

        return zPosterior

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, values):
        assert isinstance(values, slice), TypeError("indices must be a slice")
        self._indices = values

    @property
    def interfacePosterior(self):
        out = Histogram.fromHdf(self.hdf_file['model/mesh/y/edges/posterior'])

        if out.mesh.y.name == "Depth":
            out.mesh.y.edges = StatArray(-out.mesh.y.edges, name='elevation', units=out.mesh.y.units)
        out.mesh.y.relative_to = self.data.elevation

        return out

    @cached_property
    def labels(self):
        return self.getAttribute('labels')

    @cached_property
    def line_number(self):
        return self.data.line_number[0]

    @property
    def longest_coordinate(self):
        # if 0.8 < self.data.x.range / self.data.y.range < 1.2:
        #     return sqrt(self.data.x**2.0 + self.data.y**2.0)
        return self.data.x if self.data.x.range > self.data.y.range else self.data.y

    @cached_property
    def mesh(self):
        """Get the 2D topo fitting rectilinear mesh. """
        # if self._mesh is None:
        mesh = hdfRead.read_item(self.hdf_file['/model/mesh/y/edges/posterior/mesh'], skip_posterior=True)

        # Change positive depth to negative height
        mesh.y.edges = StatArray(-mesh.y.edges, name='Height', units=self.y.units)
        mesh.y.relative_to = self.elevation
        mesh.x.centres = self.longest_coordinate

        return mesh

    @property
    def minParameter(self):
        """ Get the mean model of the parameters """
        return min(asarray(self.hdf_file["model/values/posterior/x/x/data"][:, 0]))

    @cached_property
    def model(self):
        out = Model.fromHdf(self.hdf_file['/model'], skip_posterior=False)

        out.mesh.y_edges = StatArray(-out.mesh.y_edges, name='elevation', units=self.mean_parameters().mesh.y.units)
        out.mesh.relative_to = self.elevation
        out.mesh.x.centres = self.longest_coordinate
        return out

    @cached_property
    def nLayers(self):
        """ Get the number of layers in the best model for each data point """
        return StatArray.fromHdf(self.hdf_file['model/nCells'])

    @property
    def nPoints(self):
        return self.fiducials.size

    @cached_property
    def nSystems(self):
        """ Get the number of systems """
        return self.getAttribute('# of systems')

    @property
    def parallel_access(self):
        return not self.world is None

    @property
    def parameterName(self):
        return self.hdf_file['/model/values/posterior/mesh/y/edges'].attrs['name']

    @property
    def parameterUnits(self):
        return self.hdf_file['/model/values/posterior/mesh/y/edges'].attrs['units']

    @property
    def prng(self):
        return self._prng

    @prng.setter
    def prng(self, value):
        assert isinstance(value, Generator), TypeError(("prng must have type np.random.Generator.\n"
                                                        "You can generate one using\n"
                                                        "from numpy.random import Generator\n"
                                                        "from numpy.random import PCG64DXSM\n"
                                                        "Generator(bit_generator)\n\n"
                                                        "Where bit_generator is one of the several generators from either numpy or randomgen"))

        self._prng = value

    @cached_property
    def relativeError(self):
        """ Get the Relative error of the best data points """
        return StatArray.fromHdf(self.hdf_file['data/relative_error'])

    @property
    def relativeErrorPosteriors(self):
        """ Get the Relative error of the best data points """
        return self.data.relative_error.posterior

    @cached_property
    def totalError(self):
        """ Get the total error of the best data points """
        return self.getAttribute('Total Error')

    @cached_property
    def x(self):
        """ Get the X co-ordinates (Easting) """
        return StatArray.fromHdf(self.hdf_file['data/x'])

    @cached_property
    def y(self):
        """ Get the Y co-ordinates (Easting) """
        return StatArray.fromHdf(self.hdf_file['data/y'])

    @property
    def probability_of_highest_marginal(self):

        out = StatArray(self.mesh.shape, "Probability")

        hm = self.highestMarginal
        classes = unique(hm)

        mp = self.marginal_probability()

        for i, c in enumerate(classes):
            iWhere = where(hm == c)
            out[iWhere[0], iWhere[1]] = mp[i, iWhere[0], iWhere[1]]

        return out

    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, communicator):
        self._world = communicator

    def __del__(self):
        if "_hdf_file" in self.__dict__:
            self.hdf_file.close()

    def __deepcopy__(self, memo={}):
        return None

    def open(self, filename, mode = "r", **kwargs):
        """ Check whether the file is open """
        if self.parallel_access:
            kwargs['driver'] = 'mpio'
            kwargs['comm'] = self.world

        self.hdf_file = h5py.File(filename, mode, **kwargs)
        self.mode = mode

    @property
    def writable(self):
        return self.mode in ('w', 'r+', 'a')

    def close(self):
        """ Check whether the file is open """
        try:
            self.hdf_file.close()
        except:
            pass # Already closed

    def changeUnits(self, units='m'):
        """Change the units of the Coordinates

        Parameters
        ----------
        units : str
            The distance units to change to

        """
        if (units == 'km' and self.x.units != 'km'):
            self._x = self.x / 1000.0
            self._y = self.y / 1000.0

            self._x.units = 'km'
            self._y.units = 'km'


    # def crossplotErrors(self, system=0, **kwargs):
    #     """ Create a crossplot of the relative errors against additive errors for the most probable data point, for each data point along the line """
    #     kwargs['marker'] = kwargs.pop('marker','o')
    #     kwargs['markersize'] = kwargs.pop('markersize',5)
    #     kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
    #     kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
    #     kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
    #     kwargs['linestyle'] = kwargs.pop('linestyle','none')
    #     kwargs['linewidth'] = kwargs.pop('linewidth',0.0)

    #     if (self.nSystems > 1):
    #         r = range(self.nSystems)
    #         for i in r:
    #             fc = cP.wellSeparated[i+2]
    #             cP.plot(x=self.relativeError[:,i], y=self.additiveError[:,i], c=fc,
    #                 alpha = 0.7,label='System ' + str(i + 1), **kwargs)

    #         plt.legend()

    #     else:
    #         fc = cP.wellSeparated[2]
    #         cP.plot(x=self.relativeError, y=self.additiveError, c=fc,
    #                 alpha = 0.7,label='System ' + str(1), **kwargs)

    #     cP.xlabel(self.relativeError.getNameUnits())
    #     cP.ylabel(self.additiveError.getNameUnits())

    def uncache(self, variable):

        if isinstance(variable, str):
            variable = [variable]

        for var in variable:
            if var in self.__dict__:
                del self.__dict__[var]

    def compute_additive_error_opacity(self, percent=95.0, log=None):

        self.additive_error_opacity = self.compute_posterior_opacity(self.additiveErrorPosteriors, percent, log)

    def compute_relative_error_opacity(self, percent=95.0, log=None):

        self.relative_error_opacity = self.compute_posterior_opacity(self.relativeErrorPosteriors, percent, log)

    # def compute_posterior_opacity(self, posterior, percent=95.0, log=None):
    #     opacity = StatArray(zeros(self.nPoints))

    #     for i in range(self.nPoints):
    #         h = Histogram1D(edges = self.additiveErrorPosteriors._edges + self.additiveErrorPosteriors.relative_to[i])
    #         h._counts[:] = self.additiveErrorPosteriors.counts[i, :]
    #         opacity[i] = h.credibleRange(percent, log)

    #     opacity = opacity.normalize()
    #     return 1.0 - opacity

    # @cached_property
    # def bestData(self):
    #     """ Get the best data """
    #     dtype = self.hdf_file['data'].attrs['repr']
    #     if "FdemDataPoint" in dtype:
    #         bestData = FdemData.fromHdf(self.hdf_file[attr[0]])
    #     elif "TdemDataPoint" in dtype:
    #         bestData = TdemData.fromHdf(self.hdf_file[attr[0]])
    #     return bestData

    def percentile(self, percent, slic=None):
        # Read in the opacity if present
        key = "percentile_{}".format(percent)
        if key in self.hdf_file.keys():
            try:
                return StatArray.fromHdf(self.hdf_file[key], index=slic)
            except:
                pass

        h = self.parameter_posterior()
        ci = h.percentile(percent=percent, axis=1)

        if self.mode == 'r+':
            if key in self.hdf_file.keys():
                ci.writeHdf(self.hdf_file, key)
            else:
                ci.toHdf(self.hdf_file, key)
        return ci

    def credible_interval(self, percent=90.0):
        percent = 0.5 * minimum(percent, 100.0 - percent)
        return self.percentile(percent), self.percentile(100.0-percent)

    def compute_mean_parameter(self, log=None, track=True):

        posterior = self.parameter_posterior()
        mean = posterior.mean(axis=1)

        if self.mode == 'r+':
            key = 'mean_parameter'
            if key in self.hdf_file.keys():
                mean.writeHdf(self.hdf_file, key)
            else:
                mean.toHdf(self.hdf_file, key)
            self.hdf_file[key].attrs['name'] = mean.values.name
            self.hdf_file[key].attrs['units'] = mean.values.units

        return mean

    def compute_median_parameter(self, log=None, track=True):

        posterior = self.parameter_posterior()
        median = posterior.median(axis=1)

        if self.mode == 'r+':
            key = 'median_parameter'
            if key in self.hdf_file.keys():
                median.writeHdf(self.hdf_file, key)
            else:
                median.toHdf(self.hdf_file, key)
            self.hdf_file[key].attrs['name'] = median.values.name
            self.hdf_file[key].attrs['units'] = median.values.units

        return median

    def compute_mode_parameter(self, log=None, track=True):

        posterior = self.parameter_posterior()

        mode = posterior.mode(axis=1)

        if self.mode == 'r+':
            key = 'mode_parameter'
            if key in self.hdf_file.keys():
                mode.writeHdf(self.hdf_file, key)
            else:
                mode.toHdf(self.hdf_file, key)
            self.hdf_file[key].attrs['name'] = mode.values.name
            self.hdf_file[key].attrs['units'] = mode.values.units

        return mode


    def compute_doi(self, percent=67.0, smooth=None, track=True):
        """ Get the DOI of the line depending on a percentage credible interval cutoff for each data point """

        self.uncache('doi')
        assert 0.0 < percent < 100.0, ValueError("Must have 0.0 < percent < 100.0")

        nz = self.mesh.y.nCells

        p = 0.01 * percent

        r = range(self.nPoints)
        if track:
            print('Computing Depth of Investigation', flush=True)
            r = progressbar.progressbar(r)

        @jit(**_numba_settings)
        def loop(axis, values, p):
            shp = shape(values)
            out = empty(shp[0])
            for i in range(shp[0]):
                tmp = values[i, :]
                j = shp[1] - 1
                while tmp[j] < p and j >= 1:
                    j -= 1
                out[i] = axis[i, j]
            return out

        doi = loop(self.mesh.y_centres, self.opacity().values, p)
        doi = StatArray(doi, 'Depth of investigation', 'm')

        if smooth is not None:
            doi = doi.smooth(smooth)

        if self.mode == 'r+':
            if 'doi' in self.hdf_file.keys():
                doi.writeHdf(self.hdf_file, 'doi')
            else:
                doi.toHdf(self.hdf_file, 'doi')

        return doi

    # def extract1DModel(self, values, index=None, fiducial=None):
    #     """ Obtain the results for the given iD number """

    #     assert not (index is None and fiducial is None), Exception("Please specify either an integer index or a fiducial.")
    #     assert index is None or fiducial is None, Exception("Only specify either an integer index or a fiducial.")

    #     if not fiducial is None:
    #         assert fiducial in self.fiducials, ValueError("This fiducial {} is not available from this HDF5 file. The min max fids are {} to {}.".format(fiducial, self.fiducials.min(), self.fiducials.max()))
    #         # Get the point index
    #         i = self.fiducials.searchsorted(fiducial)
    #     else:
    #         i = index
    #         fiducial = self.fiducials[index]

    #     depth = self.mesh.z.edges[:-1]
    #     parameter = values[:, i]

    #     return Model1D(self.mesh.z.nCells, depth=depth, parameters=parameter, hasHalfspace=False)

    def fiducialIndex(self, fiducial):

        if size(fiducial) == 1:
            return where(self.fiducials == fiducial)[0]

        fiducial = asarray(fiducial)
        idx = searchsorted(self.fiducials, fiducial)

        # Take care of out of bounds cases
        idx[idx==self.nPoints] = 0

        return idx[fiducial == self.fiducials[idx]]

    def find_posteriors(self, grp=None, out=None):
        from h5py import Group, File
        if grp is None:
            grp = self.hdf_file
        if out is None:
            out = []
        if isinstance(grp, (Group, File)):
            if 'posterior' in grp.keys():
                out.append(grp['posterior'].name)
            if 'nPosteriors' in grp.keys():
                npost = int32(grp['nPosteriors'])
                if npost > 1:
                    for i in range(npost):
                        out.append(grp.name + '/posterior{}'.format(i))
            for k in grp.keys():
                out = self.find_posteriors(grp[k], out)
        else:
            return out

        return out

    def _get(self, variable, reciprocateParameter=False, slic=None, **kwargs):

        variable = variable.lower()
        assert variable in ['mean', 'best', 'interfaces', 'opacity', 'highest_marginal', 'marginal_probability'], ValueError("variable must be ['mean', 'best', 'interfaces', 'opacity', 'highest_marginal', 'marginal_probability']")

        if variable == 'mean':

            if reciprocateParameter:
                vals = divide(1.0, self.meanParameters(slic))
                vals.name = 'Resistivity'
                vals.units = '$Omega m$'
                return vals
            else:
                return self.meanParameters(slic)

        elif variable == 'best':
            if reciprocateParameter:
                vals = 1.0 / self.bestParameters(slic)
                vals.name = 'Resistivity'
                vals.units = '$Omega m$'
                return vals
            else:
                return self.bestParameters(slic)

        if variable == 'interfaces':
            return self.interface_probability(slic)

        if variable == 'opacity':
            return self.opacity(slic)

        if variable == 'highest_marginal':
            return self.highest_marginal(slic)

        if variable == 'marginal_probability':
            assert "index" in kwargs, ValueError('Please specify keyword "index" when requesting marginal_probability')
            assert not kwargs['index'] is None, ValueError('Please specify keyword "index" when requesting marginal_probability')
            return self.marginal_probability((slic[0], slic[1], kwargs["index"]))


    # def fit_gaussian_mixture(self, intervals, **kwargs):

    #     distributions = []

    #     hm = deepcopy(self.hitmap(0))
    #     counts = asarray(self.hdf_file['model/values/posterior/arr/data'])

    #     # Bar = progressbar.ProgressBar()
    #     for i in range(self.nPoints):

    #         try:
    #             dpDistributions = hm.fitMajorPeaks(intervals, **kwargs)
    #             distributions.append(dpDistributions)
    #         except:
    #             pass

    #         hm._counts = counts[i, :, :]

    #     return distributions


    # def fitMajorPeaks(self, intervals, **kwargs):
    #     """Fit distributions to the major peaks in each hitmap along the line.

    #     Parameters
    #     ----------
    #     intervals : array_like, optional
    #         Accumulate the histogram between these invervals before finding peaks

    #     """
    #     distributions = []

    #     hm = deepcopy(self.hitmap(0))
    #     counts = asarray(self.hdf_file['model/values/posterior/arr/data'])

    #     # Bar = progressbar.ProgressBar()
    #     for i in range(self.nPoints):

    #         try:
    #             dpDistributions = hm.fitMajorPeaks(intervals, **kwargs)
    #             distributions.append(dpDistributions)
    #         except:
    #             pass

    #         hm._counts = counts[i, :, :]

    #     return distributions

    def fit_estimated_pdf(self, intervals=None, external_files=True, **kwargs):
        """Uses Mixture modelling to fit disrtibutions to the hitmaps for the specified intervals.

        This mpi version fits all hitmaps individually throughout the data set.
        This provides detailed fits, but requires a lot of compute, hence the mpi enabled version.

        Parameters
        ----------
        intervals : array_like
            Depth intervals between which the marginal histogram is computed before fitting.

        See Also
        --------
        geobipy.Histogram1D.fit_mixture
            For details on the fitting arguments.

        """

        max_distributions = kwargs.get('max_distributions', 3)
        kwargs['track'] = False

        if intervals is None:
            intervals = self.mesh.y.edges

        nIntervals = size(intervals) - 1

        if external_files:
            hdfFile = h5py.File("Line_{}_fits.h5".format(self.line_number), 'w')
        else:
            hdfFile = self.hdf_file

        a = zeros(max_distributions)
        mixture = mixPearson(a, a, a, a)
        mixture.createHdf(hdfFile, 'fits', nRepeats=(self.nPoints, nIntervals))


        nUpdate = 1
        counter = 0

        nI = intervals.size - 1

        for i in range(1):

            hm = self.hitmap(i)

            mixtures = hm.fit_estimated_pdf(**kwargs)

            for j, m in enumerate(mixtures):
                if not m is None:
                    m.writeHdf(hdfFile, 'fits', index=(i, j))

            # counter += 1
            # if counter == nUpdate:
            #     print('rank {}, line/fiducial {}/{}, iteration {}/{},  time/dp {} h:m:s'.format(self.world.rank, self.line, self.fiducials[i], i-i0+1, chunk, str(timedelta(seconds=MPI.Wtime()-t0)/nUpdate)), flush=True)
            #     t0 = MPI.Wtime()
            #     counter = 0

        # print('rank {} finished in {} h:m:s'.format(self.world.rank, str(timedelta(seconds=MPI.Wtime()-tBase))), flush=True)

        if external_files:
            hdfFile.close()

    # def fit_interface_posterior(self, **kwargs):

    #     fit_interfaces = zeros(self.interfacePosterior.shape)
    #     for i in progressbar.progressbar(range(self.nPoints)):
    #         h1 = self.interfacePosterior[:, i]
    #         vest = h1.estimateVariance(100000, log=10)
    #         fit, f, p = h1.fit_estimated_pdf(mixture_type='pearson', smooth=vest, mask=0.5, epsilon=1e-1, mu=1e-5, method='lbfgsb', max_distributions=self.nLayers[i]-1)
    #         fit_interfaces[:, i] = fit.probability(h1.binCentres, log=False).sum(axis=1)
    #     return fit_interface


    def fit_estimated_pdf_mpi(self, intervals=None, external_files=True, **kwargs):
        """Uses Mixture modelling to fit disrtibutions to the hitmaps for the specified intervals.

        This mpi version fits all hitmaps individually throughout the data set.
        This provides detailed fits, but requires a lot of compute, hence the mpi enabled version.

        Parameters
        ----------
        intervals : array_like
            Depth intervals between which the marginal histogram is computed before fitting.

        See Also
        --------
        geobipy.Histogram1D.fit_mixture
            For details on the fitting arguments.

        """

        from mpi4py import MPI

        max_distributions = kwargs.get('max_distributions', 3)
        kwargs['track'] = False

        if intervals is None:
            intervals = self.hitmap(0).yBins

        nIntervals = size(intervals) - 1

        if external_files:
            hdfFile = h5py.File("Line_{}_fits.h5".format(self.line_number), 'w', driver='mpio', comm=self.world)
        else:
            hdfFile = self.hdf_file

        a = zeros(max_distributions)
        mixture = mixPearson(a, a, a, a)
        try:
            mixture.createHdf(hdfFile, 'fits', nRepeats=(self.nPoints, nIntervals))
        except:
            pass

        # Distribute the points amongst cores.
        starts, chunks = loadBalance1D_shrinkingArrays(self.nPoints, self.world.size)
        chunk = chunks[self.world.rank]
        i0 = starts[self.world.rank]
        i1 = i0 + chunk

        tBase = MPI.Wtime()
        t0 = tBase

        nUpdate = 1
        counter = 0

        nI = intervals.size - 1

        buffer = empty((nI, max_distributions))

        for i in range(i0, i1):

            hm = self.hitmap(i)

            mixtures = None
            if not npall(hm.counts == 0):
                mixtures = hm.fit_estimated_pdf(iPoint=i, rank=self.world.rank, **kwargs)

            if not mixtures is None:
                for j, m in enumerate(mixtures):
                    if not m is None:
                        m.writeHdf(hdfFile, 'fits', index=(i, j))

            counter += 1
            if self.world.rank == 0:
                if counter == nUpdate:
                    print('rank {}, line/fiducial {}/{}, iteration {}/{},  time/dp {} h:m:s'.format(self.world.rank, self.line_number, self.fiducials[i], i-i0+1, chunk, str(timedelta(seconds=MPI.Wtime()-t0)/nUpdate)), flush=True)
                    t0 = MPI.Wtime()
                    counter = 0

        print('rank {} finished in {} h:m:s'.format(self.world.rank, str(timedelta(seconds=MPI.Wtime()-tBase))), flush=True)

        if external_files:
            hdfFile.close()

    def _z_slice(self, depth=None):

        if depth is None:
            return s_[:]

        if isinstance(depth, (integer, int, slice)):
            return depth

        if size(depth) > 1:
            assert size(depth) == 2, ValueError("depth must be a scalar or size 2 array.")
            depth.sort()
            assert npall(depth < self.mesh.z.edges[-1]), 'Depths must be lees than max depth {}'.format(self.mesh.z.edges[-1])

            cell1 = self.mesh.z.cellIndex(depth[0])
            cell2 = self.mesh.z.cellIndex(depth[1])
            out = s_[cell1:cell2+1]
        else:
            assert depth < self.mesh.z.edges[-1], 'Depth must be lees than max depth {}'.format(self.mesh.z.edges[-1])

            out = self.mesh.z.cellIndex(depth)

        return out

    def depth_slice(self, depth, variable, stat=mean, **kwargs):
        """ Obtain a slice at depth from values

        Parameters
        ----------
        depth : float or array_like or int or slice
            If float: The depth at which to obtain the slice
            If array_like: length 2 array of an interval over which to average.
            If int: the index along depth
            If slice: A slice along depth to return
        values : array_like
            Values of shape self.mesh.shape from which to obtain the slice.

        Returns
        -------
        out : geobipy.StatArray
            The slice at depth.

        """

        z_slice = self._z_slice(depth=depth)

        out = self._get(variable, slic=(s_[:], z_slice), **kwargs)

        if size(depth) > 1:
            out = stat(out, axis = 0)

        return out


    def elevationSlice(self, elevation, values):
        """ Obtain a slice at an elevation from values

        Parameters
        ----------
        elevation : float or array_like
            If float: The depth at which to obtain the slice
            If arraylike: length 2 array of an interval over which to average.
        values : array_like
            Values of shape self.mesh.shape from which to obtain the slice.

        Returns
        -------
        out : geobipy.StatArray
            The slice at depth.

        """

        assert npall(shape(values) == self.mesh.shape), ValueError("values must have shape {}".fomat(self.mesh.shape))

        out = full(self.nPoints, fill_value=nan)

        if size(elevation) > 1:

            for i in range(self.nPoints):
                tmp = self.elevation[i] - elevation
                if tmp[1] < self.mesh.z.edges[-1] and tmp[0] > self.mesh.z.edges[0]:
                    cell1 = self.mesh.z.cellIndex(tmp[1], clip=True)
                    cell2 = self.mesh.z.cellIndex(tmp[0], clip=True)

                    out[i] = mean(values[cell1:cell2+1, i])

        else:

            for i in range(self.nPoints):
                tmp = self.elevation[i] - elevation
                if tmp > self.mesh.z.edges[0] and tmp < self.mesh.z.edges[-1]:
                    cell1 = self.mesh.z.cellIndex(tmp, clip=True)

                    out[i] = values[cell1, i]

        return out


    # def identifyPeaks(self, depths, nBins = 250, width=4, limits=None):
    #     """Identifies peaks in the parameter posterior for each depth in depths.

    #     Parameters
    #     ----------
    #     depths : array_like
    #         Depth intervals to identify peaks between.

    #     Returns
    #     -------

    #     """

    #     from scipy.signal import find_peaks

    #     assert size(depths) > 2, ValueError("Depths must have size > 1.")

    #     tmp = self.lineHitmap.intervalStatistic(axis=0, intervals = depths, statistic='sum')

    #     depth = zeros(0)
    #     parameter = zeros(0)

    #     # # Bar = progressbar.ProgressBar()
    #     # # for i in Bar(range(self.nPoints)):
    #     for i in range(tmp.y.nCells):
    #         peaks, _ = find_peaks(tmp.counts[i, :],  width=width)
    #         values = tmp.x.centres[peaks]
    #         if not limits is None:
    #             values = values[(values > limits[0]) & (values < limits[1])]
    #         parameter = hstack([parameter, values])
    #         depth = hstack([depth, full(values.size, fill_value=0.5*(depths[i]+depths[i+1]))])

    #     return asarray([depth, parameter]).T

    def interface_probability(self, slic=None):
        """ Get the layer interfaces from the layer depth histograms """
        return self.interfacePosterior.pdf

    # def compute_interface_probability(self):
    #     maxCount = self.interfacePosterior.counts.max()
    #     if size(self.interfacePosterior.counts, 0) != (self.mesh.z.nCells):
    #         values = vstack([self.interfacePosterior.counts, self.interfacePosterior.counts[-1, :]])
    #         out = StatArray(values / float64(maxCount), "interfaces", "")
    #     else:
    #         out = StatArray(self.interfacePosterior.counts / float64(maxCount), "interfaces", "")

        # if 'p(interface)' in self.hdf_file.keys():
        #     out.writeHdf(self.hdf_file, 'p(interface)')
        # else:
        #     out.toHdf(self.hdf_file, 'p(interface)')

        # return out

    # @property
    # def maxParameter(self):
    #     """ Get the mean model of the parameters """
    #     return max(asarray(self.hdf_file["model/values/posterior/mesh/x/edges/data"][:, -1]))

    def mean_parameters(self, slic=None):
        if not 'mean_parameter' in self.hdf_file:
            self._mean_parameter = self.compute_mean_parameter(log=10)
        else:
            # g = self.hdf_file['mean_parameter']
            # print(g.attrs['repr'])
            self._mean_parameter = Model.fromHdf(self.hdf_file['mean_parameter'])

        return self._mean_parameter



    def change_mesh_axis(self, axis):
        if self._mesh is None:
            self.mesh
        self._mesh.x.centres = self.x_axis(axis)



    def opacity(self, slic=None):
        """ Get the model parameter opacity using the credible intervals """
        if "opacity" in self.hdf_file.keys():
            if not slic is None:
                slic = slic[::-1]
            return Model.fromHdf(self.hdf_file['opacity'], index=slic)
        else:
            return self.compute_opacity()

    def compute_opacity(self, percent=90.0, log=10, multiplier=0.5):

        self.uncache('opacity')

        opacity = self.parameter_posterior().opacity(percent, log, axis=1)

        if self.mode == 'r+':
            if 'opacity' in self.hdf_file.keys():
                opacity.writeHdf(self.hdf_file, 'opacity')
            else:
                opacity.toHdf(self.hdf_file, 'opacity')

        return opacity

    def compute_probability(self, distribution, log=None, log_probability=False, axis=0, save=False, **kwargs):

        p = self.parameter_posterior().compute_probability(distribution, log, log_probability, axis, **kwargs)

        if save and self.writable:
            if 'probabilities' in self.hdf_file.keys():
                p.writeHdf(self.hdf_file, 'probabilities')
            else:
                p.toHdf(self.hdf_file, 'probabilities')

        return p

    # def percentageParameter(self, value, depth=None, depth2=None, progress=False):

    #     # Get the depth grid
    #     if (not depth is None):
    #         assert depth <= self.mesh.z.edges[-1], 'Depth is greater than max depth '+str(self.mesh.z.edges[-1])
    #         j = self.mesh.z.cellIndex(depth)
    #         k = j+1
    #         if (not depth2 is None):
    #             assert depth2 <= self.mesh.z.edges[-1], 'Depth2 is greater than max depth '+str(self.mesh.z.edges[-1])
    #             assert depth <= depth2, 'Depth2 must be >= depth'
    #             k = self.mesh.z.cellIndex(depth2)

    #     percentage = StatArray(empty(self.nPoints), name="Probability of {} > {:0.2f}".format(self.parameterName, value), units = self.parameterUnits)

    #     if depth:
    #         counts = self.hdf_file['model/values/posterior/arr/data'][:, j:k, :]
    #         # return StatArray(sum(counts[:, :, pj:]) / sum(counts) * 100.0, name="Probability of {} > {:0.2f}".format(self.meanParameters.name, value), units = self.meanParameters.units)
    #     else:
    #         counts = self.hdf_file['model/values/posterior/arr/data']

    #     parameters = RectilinearMesh1D.fromHdf(self.hdf_file['model/values/posterior/x'])

    #     Bar = progressbar.ProgressBar()
    #     print('Computing P(X > value)', flush=True)
    #     for i in Bar(range(self.nPoints)):
    #         p = RectilinearMesh1D(edges=parameters.edges[i, :])
    #         pj = p.cellIndex(value)

    #         cTmp = counts[i, :, :]

    #         percentage[i] = sum(cTmp[:, pj:]) / cTmp.sum()

    #     return percentage

    def read(self, key, **kwargs):
        return hdfRead.read_item(self.hdf_file, key, **kwargs)

    def parameter_posterior(self, index=None, fiducial=None, **kwargs):

        if fiducial is not None:
            assert fiducial in self.fiducials, ValueError("This fiducial {} is not available from this HDF5 file. The min max fids are {} to {}.".format(fiducial, self.fiducials.min(), self.fiducials.max()))
            # Get the point index
            index = self.fiducials.searchsorted(fiducial)

        out = Histogram.fromHdf(self.hdf_file['/model/values/posterior'], index=index)

        out.x.centres = self.data.axis(kwargs.pop('x', 'x'))

        if out.mesh.z.name == "Depth":
            out.mesh.z.edges = StatArray(-out.mesh.z.edges, name='elevation', units=out.mesh.z.units)

        out.mesh.z.relative_to = self.data.elevation
        out.mesh.y.relative_to = self.halfspace

        return out

    def ncells_posterior(self, index=None, fiducial=None):

        if fiducial is not None:
            assert fiducial in self.fiducials, ValueError("This fiducial {} is not available from this HDF5 file. The min max fids are {} to {}.".format(fiducial, self.fiducials.min(), self.fiducials.max()))
            # Get the point index
            index = self.fiducials.searchsorted(fiducial)

        return Histogram.fromHdf(self.hdf_file['/model/mesh/nCells/posterior'], index=index)


    def inference_1d(self, index=None, fiducial=None, reciprocateParameter=False):
        """ Obtain the results for the given iD number """

        assert not (index is None and fiducial is None), Exception("Please specify either an integer index or a fiducial.")
        assert index is None or fiducial is None, Exception("Only specify either an integer index or a fiducial.")

        if not fiducial is None:
            assert fiducial in self.fiducials, ValueError("This fiducial {} is not available from this HDF5 file. The min max fids are {} to {}.".format(fiducial, self.fiducials.min(), self.fiducials.max()))
            # Get the point index
            index = self.fiducials.searchsorted(fiducial)

        R = Inference1D.fromHdf(self.hdf_file, index=index, prng=self.prng)

        return R


    # def axis(self, axis):
    #     if axis == 'index':
    #         ax = StatArray(arange(self.nPoints, dtype=float64), 'Index')
    #     elif axis == 'fiducial':
    #         ax = self.fiducial
    #     elif axis == 'x':
    #         ax = self.x
    #     elif axis == 'y':
    #         ax = self.y
    #     elif axis == 'z':
    #         ax = self.mesh.y
    #     elif axis == 'distance':
    #         ax = StatArray(sqrt((self.data.x - self.data.x[0])**2.0 + (self.data.y - self.data.y[0])**2.0), 'Distance', 'm')
    #     return ax

    # def x_axis(self, axis, centres=False):

    #     if axis == 'index':
    #         ax = StatArray(arange(self.nPoints, dtype=float64), 'Index')
    #     elif axis == 'fiducial':
    #         ax = self.fiducial
    #     elif axis == 'distance':
    #         ax = StatArray(sqrt((self.data.x - self.data.x[0])**2.0 + (self.data.y - self.data.y[0])**2.0), 'Distance', 'm')
    #     elif axis == 'x':
    #         ax = self.x
    #     elif axis == 'y':
    #         ax = self.y
    #     return ax

    # def pcolorDataResidual(self, abs=False, **kwargs):
    #     """ Plot a channel of data as points """

    #     # xAxis = kwargs.pop('xAxis', 'x')

    #     xtmp = self.axis(xAxis, centres=False)

    #     values = self.bestData.deltaD.T

    #     if abs:
    #         values = values.abs()

    #     cP.pcolor(values, x=xtmp, y=StatArray(arange(self.bestData.predictedData.shape[1]), name='Channel'), **kwargs)


    # def pcolorObservedData(self, **kwargs):
    #     """ Plot a channel of data as points """

    #     cP.pcolor(self.bestData.data.T, x=self.mesh.x, y=StatArray(arange(self.bestData.predictedData.shape[1]), name='Channel'), **kwargs)


    # def pcolorPredictedData(self, **kwargs):
    #     """ Plot a channel of data as points """

    #     cP.pcolor(self.bestData.predictedData.T, x=self.mesh.x, y=StatArray(arange(self.bestData.predictedData.shape[1]), name='Channel'), **kwargs)


    # def plot_predictedData(self, channel=None, **kwargs):
    #     """ Plot a channel of the best predicted data as points """

    #     if channel is None:
    #         channel = s_[:]

    #     cP.plot(self.mesh.x, self.bestData.predictedData[:, channel], **kwargs)

    def plot_burned_in(self, low=0.0, high=1.0, **kwargs):

        x = self.data.axis(kwargs.pop('x', 'x'))
        cmap = plt.get_cmap(kwargs.pop('cmap', 'cividis'))

        ax = kwargs.pop('ax', plt.gca())

        if kwargs.pop('underlay', False):
            kwargs['alpha'] = kwargs.get('alpha', 0.5)
            ylim = list(ax.get_ylim())
        else:
            ylim = [low, high]

        ax.fill_between(x, ylim[1], ylim[0], step='mid', color=cmap(1.0), label="True", **kwargs)
        ax.fill_between(x, (ylim[1]-ylim[0])*(1-self.burned_in)+ylim[0], ylim[0], step='mid', color=cmap(0.0), label="False", **kwargs)
        ax.legend(title="Burned in", prop={'size': 6}, fontsize=6)

        return ax

    def plot_channel_saturation(self, **kwargs):

        kwargs['x'] = kwargs.get('x', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color', 'k')
        kwargs['linewidth'] = kwargs.pop('linewidth', 0.5)

        self.data.plot(values=self.data.channel_saturation, **kwargs)

    def plot_data_elevation(self, **kwargs):
        """ Adds the data elevations to a plot """

        kwargs['x'] = kwargs.pop('x', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color', 'k')
        kwargs['linewidth'] = kwargs.pop('linewidth', 0.5)

        self.data.plot_data_elevation(**kwargs)

    def plotDataResidual(self, channel=None, abs=False, **kwargs):
        """ Plot a channel of the observed data as points """

        if channel is None:
            channel = s_[:]

        values = self.bestData.deltaD[:, channel]

        if abs:
            values = values.abs()

        self.mesh.plot_line(values, axis=1, **kwargs)

    def plot_doi(self, **kwargs):

        kwargs['x'] = kwargs.pop('x', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color', 'k')
        kwargs['linewidth'] = kwargs.pop('linewidth', 0.5)

        self.data.plot(values=self.doi, **kwargs)

    def plot_elevation(self, **kwargs):
        kwargs['x'] = kwargs.pop('x', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color','k')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.5)

        self.data.plot(values=self.data.elevation, **kwargs)

    def plot_k_layers(self, **kwargs):
        """ Plot the number of layers in the best model for each data point """
        post = self.model.nCells.posterior
        post.mesh.x.centres = self.data.axis(kwargs.pop('x', 'x'))
        ax, _, _ = post.plot(overlay=self.model.nCells, axis=1)
        ax.set_title('P(# of Layers)')


    def plot_additive_error(self, **kwargs):
        """ Plot the additive errors of the data """
        xAxis = kwargs.pop('x', 'x')
        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','-')
        kwargs['linewidth'] = kwargs.pop('linewidth',1.0)


        if (self.nSystems > 1):
            r = range(self.nSystems)
            for i in r:
                fc = cP.wellSeparated[i+2]
                self.data.plot(values=self.additiveError[:, i],
                c=fc,
                alpha = 0.7,label='System ' + str(i + 1),
                **kwargs)
            plt.legend()
        else:
            fc = cP.wellSeparated[2]
            self.data.plot(values=self.additiveError,
                    c=fc,
                    alpha = 0.7,label='System ' + str(1), **kwargs)

    def plot_additive_error_posterior(self, system=0, **kwargs):
        """ Plot the distributions of additive errors as an image for all data points in the line """

        if self.nSystems > 1:
            post = self.additiveErrorPosteriors[system]
        else:
            post = self.additiveErrorPosteriors

        post.pcolor(**kwargs)

        cP.title('Additive error posterior distributions\nsystem {}'.format(system))

    def plot_confidence(self, **kwargs):
        """ Plot the opacity """
        kwargs['cmap'] = kwargs.get('cmap', 'plasma')

        ax, pm, cb = self.plot_x_section(self.opacity(), ticks=[0.0, 0.5, 1.0], **kwargs)

        if cb is not None:
            labels = ['Less', '', 'More']
            cb.ax.set_yticklabels(labels)
            cb.set_label("Confidence")

        return ax, pm, cb


    def plot_entropy(self, **kwargs):
        kwargs['cmap'] = kwargs.get('cmap', 'hot')
        return self.plot_x_section(self.entropy, **kwargs)

    # def plotError2DJointProbabilityDistribution(self, index, system=0, **kwargs):
    #     """ For a given index, obtains the posterior distributions of relative and additive error and creates the 2D joint probability distribution """

    #     # Read in the histogram of relative error for the data point
    #     rel = self.getAttribute('Relative error histogram', index=index)
    #     # Read in the histogram of additive error for the data point
    #     add = self.getAttribute('Additive error histogram', index=index)

    #     joint = Histogram2D()
    #     joint.create2DjointProbabilityDistribution(rel[system],add[system])

    #     joint.pcolor(**kwargs)

    def plot_interfaces(self, cut=0.0, **kwargs):
        """ Plot a cross section of the layer depth histograms. Truncation is optional. """
        kwargs['cmap'] = kwargs.get('cmap', 'gray_r')
        return self.plot_x_section(self.interface_probability(), **kwargs)

    def plot_relative_error_posterior(self, system=0, **kwargs):
        """ Plot the distributions of relative errors as an image for all data points in the line """

        if self.nSystems > 1:
            post = self.relativeErrorPosteriors[system]
        else:
            post = self.relativeErrorPosteriors

        kwargs['trim'] = kwargs.get('trim', 0.0)
        post.mesh.x.centres = self.data.axis(kwargs.pop('x', 'x'))

        post.pcolor(**kwargs)

        cP.title('Relative error posterior distributions\nsystem {}'.format(system))

    def plot_relative_error(self, **kwargs):
        """ Plot the relative errors of the data """

        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','-')
        kwargs['linewidth'] = kwargs.pop('linewidth',1.0)

        if (self.nSystems > 1):
            r = range(self.nSystems)
            for i in r:
                kwargs['c'] = cP.wellSeparated[i+2]
                self.data.plot(values=self.relativeError[:, i],
                alpha = 0.7, label='System {}'.format(i + 1), **kwargs)
            plt.legend()
        else:
            kwargs['c'] = cP.wellSeparated[2]
            self.data.plot(values=self.relativeError[:, i],
                alpha = 0.7, label='System {}'.format(1), **kwargs)

    def scatter2D(self, **kwargs):
        return self.data.scatter2D(**kwargs)

    def plot_total_error(self, channel, **kwargs):
        """ Plot the relative errors of the data """
        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','-')
        kwargs['linewidth'] = kwargs.pop('linewidth',1.0)


        fc = cP.wellSeparated[2]
        self.data.plot(values=self.totalError[:, channel], alpha = 0.7, label='Channel ' + str(channel), **kwargs)

    def plotTotalErrorDistributions(self, channel=0, nBins=100, **kwargs):
        """ Plot the distributions of relative errors as an image for all data points in the line """
        self.setAlonglineAxis(self.plotAgainst)

        H = Histogram(values=log10(self.totalError[:,channel]),nBins=nBins)

        H.plot(**kwargs)

    def histogram(self, nBins, depth=None, reciprocateParameter = False, bestModel = False, **kwargs):
        """ Compute a histogram of the model, optionally show the histogram for given depth ranges instead """

        z_slice = self._z_slice(depth)

        # if (depth1 is None):
        #     depth1 = maximum(self.mesh.z.edges[0], 0.0)
        # if (depth2 is None):
        #     depth2 = self.mesh.z.edges[-1]

        # maxDepth = self.mesh.z.edges[-1]

        # # Ensure order in depth values
        # if (depth1 > depth2):
        #     tmp = depth2
        #     depth2 = depth1
        #     depth1 = tmp

        # # Don't need to check for depth being shallower than self.mesh.y.edges[0] since the sortedsearch will return 0
        # assert depth1 <= maxDepth, ValueError('Depth1 is greater than max depth {}'.format(maxDepth))
        # assert depth2 <= maxDepth, ValueError('Depth2 is greater than max depth {}'.format(maxDepth))

        # cell1 = self.mesh.z.cellIndex(depth1, clip=True)
        # cell2 = self.mesh.z.cellIndex(depth2, clip=True)

        if (bestModel):
            vals = self.bestParameters(z_slice)
            title = 'Best model values depth = {}'.format(depth)
        else:
            vals = self.mean_parameters(z_slice)
            title = 'Mean model values depth = {}'.format(depth)

        log = kwargs.pop('log', None)

        f = linspace
        vals2 = vals
        if (log):
            vals2, logLabel = cF._log(vals,log)
            # name = logLabel + name
            f = logspace

        mesh = RectilinearMesh1D(edges = StatArray(f(nanmin(vals2), nanmax(vals2), nBins+1)), log=log)

        h = Histogram(mesh=mesh)
        h.update(vals)
        h.plot(**kwargs)
        cP.title(title)

    def parameterHistogram(self, nBins, depth = None, depth2 = None, log=None):
        """ Compute a histogram of all the parameter values, optionally show the histogram for given depth ranges instead """

        # Get the depth grid
        if (not depth is None):
            assert depth <= self.mesh.z.edges[-1], 'Depth is greater than max depth '+str(self.mesh.z.edges[-1])
            j = self.mesh.z.cellIndex(depth)
            k = j+1
            if (not depth2 is None):
                assert depth2 <= self.mesh.z.edges[-1], 'Depth2 is greater than max depth '+str(self.mesh.z.edges[-1])
                assert depth <= depth2, 'Depth2 must be >= depth'
                k = self.mesh.z.cellIndex(depth2)

        # First get the min max of the parameter hitmaps
        x0 = log10(self.minParameter)
        x1 = log10(self.maxParameter)

        if depth:
            counts = self.hdf_file['model/values/posterior/arr/data'][:, j:k, :]
            # return StatArray(sum(counts[:, :, pj:]) / sum(counts) * 100.0, name="Probability of {} > {:0.2f}".format(self.meanParameters.name, value), units = self.meanParameters.units)
        else:
            counts = self.hdf_file['model/values/posterior/arr/data']

        parameters = RectilinearMesh1D.fromHdf(self.hdf_file['model/values/posterior/x'])

        bins = StatArray(logspace(x0, x1, nBins), self.parameterName, units = self.parameterUnits)

        out = Histogram1D(edges=bins, log=log)

        # Bar = progressbar.ProgressBar()
        # for i in Bar(range(self.nPoints)):
        for i in range(self.nPoints):
            p = RectilinearMesh1D(edges=parameters.edges[i, :])
            pj = out.cellIndex(p.centres, clip=True)
            cTmp = counts[i, :, :]
            out.counts[pj] += sum(cTmp, axis=0)

        return out

    def plot_best_model(self, **kwargs):
        kwargs['mask_by_confidence'] = False
        kwargs['mask_by_doi'] = False
        return self.plot_x_section(self.model, **kwargs)


    def plot_highest_marginal(self, **kwargs):
        return self.plot_x_section(self.highest_marginal(), **kwargs)


    def plot_marginal_probabilities(self, **kwargs):

        mp = self.marginal_probability()

        classes = kwargs.pop("classes", {})
        classes['id'] = self.highest_marginal().values

        kwargs['vmin'] = kwargs.get('vmin', 1.0 / mp.shape[1])

        return self.plot_x_section(mp, axis=1, classes = classes, **kwargs)


    def mask(self, model, **kwargs):

        from pprint import pprint
        mask = None
        if kwargs.pop('mask_by_confidence', False):
            mask = self.opacity().values

        if kwargs.pop('mask_by_burned_in', False):
            if mask is not None:
                mask *= self.burned_in_mask
            else:
                mask = self.burned_in_mask

        if kwargs.pop('mask_by_doi', False):
            if mask is not None:
                mask *= self.doi_mask(model)
            else:
                mask = self.doi_mask(model)

        if kwargs.pop('mask_by_probability', False):
            if mask is not None:
                mask *= self.probability_mask()
            else:
                mask = self.probability_mask()

        return mask, kwargs

    def plot_x_section(self, model, **kwargs):
        model.mesh.x.centres = self.data.axis(kwargs.pop('x', 'x'))
        mask, kwargs = self.mask(model, **kwargs); kwargs['alpha'] = mask
        return model.pcolor(**kwargs)


    def plot_mean_model(self, **kwargs):
        return self.plot_x_section(self.mean_parameters(), **kwargs)

    def plot_median_model(self, **kwargs):
        return self.plot_x_section(self.compute_median_parameter(), **kwargs)

    def plot_mode_model(self, **kwargs):
        return self.plot_x_section(self.compute_mode_parameter(), **kwargs)

    @property
    def doi_mask(self):

        mask = ones(self.mesh.shape)
        indices = self.mesh.y.cellIndex(self.doi + self.mesh.y.relative_to)

        for i in range(self.nPoints):
            mask[i, indices[i]:] = 0.0

        return mask

    @property
    def burned_in_mask(self):
        mask = ones(self.mesh.shape)
        mask[~self.burned_in, :] = 0.0

        return mask

    @property
    def probability_mask(self):
        p = self.marginal_probability()
        hm = self.highest_marginal()

        mask = p.apply_along_axis(nanmax, axis=1).values

        n_classes = p.shape[1]

        # for i in range(n_classes):
        #     j = argwhere(hm.values == i)
        #     subset = p.values[:, i, :]
        #     vmin, vmax = nanmin(subset), nanmax(subset)
        #     mask[j[:, 0], j[:, 1]] = (mask[j[:, 0], j[:, 1]] - vmin) / (vmax - vmin)

        mask[isnan(mask)] = 0.0

        return mask


    def plot_percentile(self, percent, **kwargs):
        return self.plot_x_section(self.parameter_posterior().percentile(percent, axis=1), **kwargs)

    def plot_percentiles(self, ax, **kwargs):

        assert len(ax) == 3, ValueError("Must provide 3 axes")
        posterior = self.parameter_posterior()

        posterior.mesh.x.centres = self.data.axis(kwargs.pop('x', 'x'))

        percentile = posterior.percentile(np.r_[0.05, 0.5, 0.95], axis=1)

        mask, kwargs = self.mask(percentile, **kwargs); kwargs['alpha'] = mask

        return percentile.pcolor(**kwargs)


    def marginal_probability(self, slic=None):

        assert 'probabilities' in self.hdf_file.keys(), Exception("Marginal probabilities need computing, use Inference_2D.compute_probabilities()")

        if 'probabilities' in self.hdf_file.keys():
            marginal_probability = Model.fromHdf(self.hdf_file['probabilities'], index=slic)

        return marginal_probability

    def read_fit_distributions(self, fit_file, mask_by_doi=True, components='amvd'):

        # Get the fits for the given line
        # Define the depth intervals and plotting axis
        means = None
        amplitudes = None
        variances = None
        degrees = None
        with h5py.File(fit_file, 'r') as f:
            if 'm' in components:
                means = StatArray(asarray(f['means/data']), 'Conductivity', r'$\frac{S}{m}$')
            if 'a' in components:
                amplitudes = StatArray(asarray(f['amplitudes/data']), 'Amplitude')
            if 'v' in components:
                variances = StatArray(asarray(f['variances/data']), 'Variance')
            if 'd' in components:
                degrees = StatArray(asarray(f['degrees/data']), 'Degrees of freedom')

        intervals = self.mesh.z.centres

        if mask_by_doi:
            indices = intervals.searchsorted(self.doi)
            if 'a' in components:
                for i in range(self.nPoints):
                    amplitudes[i, indices[i]:, :] = nan
            if 'm' in components:
                for i in range(self.nPoints):
                    means[i, indices[i]:, :] = nan
            if 'v' in components:
                for i in range(self.nPoints):
                    variances[i, indices[i]:, :] = nan
            if 'd' in components:
                for i in range(self.nPoints):
                    degrees[i, indices[i]:, :] = nan

        iWhere = argsort(means, axis=-1)
        for i in range(means.shape[0]):
            for j in range(means.shape[1]):
                tmp = iWhere[i, j, :]
                if 'm' in components:
                    m = means[i, j, tmp]
                    means[i, j, :] = m

                if 'a' in components:
                    a = amplitudes[i, j, tmp]
                    amplitudes[i, j, :] = a

                if 'v' in components:
                    v = variances[i, j, tmp]
                    variances[i, j, :] = v

                if 'd' in components:
                    d = degrees[i, j, tmp]
                    degrees[i, j, :] = d

        return amplitudes, means, variances, degrees


    def read_lmfit_distributions(self, fit_file, mask_by_doi=True, components='amse'):

        with h5py.File(fit_file, 'r') as f:
            d = asarray(f['fits/params/data'])

        amplitudes = means = stds = exponents = None
        if 'a' in components:
            amplitudes = StatArray(d[:, :, 0::4], 'Amplitude')
        if 'm' in components:
            means = StatArray(d[:, :, 1::4], 'Conductivity', r'$\frac{S}{m}$')
        if 's' in components:
            stds = StatArray(d[:, :, 2::4]**2.0, 'Standard deviation')
        if 'e' in components:
            exponents = StatArray(d[:, :, 3::4], 'Exponent')

        return amplitudes, means, stds, exponents


    # def compute_marginal_probability_from_labelled_mixtures(self, fit_file, gmm, labels):

    #     amplitudes, means, variances, degrees = self.read_fit_distributions(fit_file, mask_by_doi=False)

    #     # self.marginal_probability = StatArray(zeros([self.nPoints, self.mesh.z.nCells, gmm.n_components]), 'Marginal probability')

    #     iSort = argsort(squeeze(gmm.means_))

    #     print('Computing marginal probability', flush=True)
    #     for i in progressbar.progressbar(range(self.nPoints)):
    #         hm = self.get_hitmap(i)
    #         for j in range(self.mesh.z.nCells):
    #             m = means[i, j, :]
    #             inan = ~isnan(m)
    #             m = m[inan]

    #             if size(m) > 0:
    #                 a = amplitudes[i, j, inan]

    #                 v = variances[i, j, inan]
    #                 df = degrees[i, j, inan]
    #                 l = labels[i, j, inan].astype(int)

    #                 fit_mixture = mixStudentT(m, v, df, a, labels=l)
    #                 fit_pdfs = fit_mixture.probability(log10(hm.xBinCentres), log=False)

    #                 # gmm_pdfs = zeros([gmm.n_components, self.hitmap(0).x.nCells])

    #                 # for k_gmm in range(gmm.n_components):
    #                 #     # Term 1: Get the weight of the labelled fit from the classification
    #                 #     relative_fraction = gmm.weights_[iSort[k_gmm]]

    #                 #     for k_mix in range(fit_mixture.n_mixtures):
    #                 #         # Term 2: Get the probability of each mixture given the mean of the student T.
    #                 #         pMixture = squeeze(gmm.predict_proba(m[k_mix].reshape(-1, 1)))[iSort[k_gmm]] / float(fit_mixture.n_mixtures)

    #                 #         gmm_pdfs[k_gmm, :] += relative_fraction * pMixture * fit_pdfs[:, k_mix]


    #                 a = gmm.weights_[iSort]
    #                 b = gmm.predict_proba(m.reshape(-1, 1))[:, iSort] / float(fit_mixture.n_mixtures)
    #                 gmm_pdfs = dot(fit_pdfs, a*b).T

    #                 h = hm.marginalize(index = j)
    #                 self.marginal_probability[i, j, :] = h._marginal_probability_pdfs(gmm_pdfs)
    #             else:
    #                 self.marginal_probability[i, j, :] = nan

    #     if 'marginal_probability' in self.hdf_file.keys():
    #         self.marginal_probability.writeHdf(self.hdf_file, 'marginal_probability')
    #     else:
    #         self.marginal_probability.toHdf(self.hdf_file, 'marginal_probability')


    # def compute_marginal_probability_from_fits(self, fit_file, mask_by_doi=True):

    #     amplitudes, means, variances, degrees = self.read_fit_distributions(fit_file, mask_by_doi)
    #     self.marginal_probability = StatArray(zeros([self.nPoints, self.mesh.z.nCells, means.shape[-1]]), 'Marginal probability')

    #     print('Computing marginal probability', flush=True)
    #     for i in progressbar.progressbar(range(self.nPoints)):
    #         hm = self.get_hitmap(i)
    #         mixtures = []
    #         for j in range(means.shape[1]):
    #             a = amplitudes[i, j, :]
    #             m = means[i, j, :]
    #             v = variances[i, j, :]
    #             df = degrees[i, j, :]

    #             inan = ~isnan(m)
    #             mixtures.append(mixStudentT(m[inan], v[inan], df[inan], a[inan]))

    #         mp = hm.marginal_probability(1.0, distributions=mixtures, log=10, maxDistributions=means.shape[-1])
    #         self.marginal_probability[i, :mp.shape[0], :] = mp

    #     if 'marginal_probability' in self.hdf_file.keys():
    #         self.marginal_probability.writeHdf(self.hdf_file, 'marginal_probability')
    #     else:
    #         self.marginal_probability.toHdf(self.hdf_file, 'marginal_probability')
    #     # self.marginal_probability.toHdf('line_{}_marginal_probability.h5'.format(self.line), 'marginal_probability')


    def highest_marginal(self, slic=None):
        out = self.marginal_probability(slic).apply_along_axis(argmax, axis=1)
        out.values.name = 'Highest marginal'
        return out

    def plot_inference_1d(self, fiducial, **kwargs):
        """ Plot the geobipy results for the given data point """
        R = self.inference_1d(fiducial=fiducial)
        # R.initFigure()
        R.plot_posteriors(**kwargs)

    def toVtk(self, fileName, format='binary'):
        """Write the parameter cross-section to an unstructured grid vtk file

        Parameters
        ----------
        fileName : str
            Filename to save to.
        format : str, optional
            "ascii" or "binary" format. Ascii is readable, binary is not but results in smaller files.

        """
        a = self.bestParameters
        b = self.meanParameters
        c = self.interfaces

        d = StatArray(1.0 / a, "Best Conductivity", "$\fraq{S}{m}$")
        e = StatArray(1.0 / b, "Mean Conductivity", "$\fraq{S}{m}$")

        self.mesh.toVTK(fileName, format=format, cellData=[a, b, c, d, e])


    def getAttribute(self, attribute, iDs = None, index=None, **kwargs):
        """ Gets an attribute from the line results file """
        assert (not attribute is None), "Please specify an attribute: \n"+self.possibleAttributes()

        old = False
        if (old):
            keys = self._attrTokeyOld(attribute)
        else:
            keys = self._attrTokey(attribute)

        if (iDs is None):
            iDs = ['/']

        return hdfRead.readKeyFromFile(self.hdf_file, self.fName, iDs, keys, index=index, **kwargs)


    # def _attrTokey(self, attributes):
    #     """ Takes an easy to remember user attribute and converts to the tag in the HDF file """
    #     if (isinstance(attributes, str)):
    #         attributes = [attributes]
    #     res = []
    #     nSys= None
    #     for attr in attributes:
    #         low = attr.lower()
    #         if (low == 'iteration #'):
    #             res.append('i')
    #         elif (low == '# of markov chains'):
    #             res.append('nmc')
    #         elif (low == 'burned in'):
    #             res.append('burned_in')
    #         elif (low == 'burn in #'):
    #             res.append('iburn')
    #         elif (low == 'data multiplier'):
    #             res.append('multiplier')
    #         elif (low == 'height posterior'):
    #             res.append('data/z/posterior')
    #         elif (low == 'fiducials'):
    #             res.append('data/fiducial')
    #         elif (low == 'labels'):
    #             res.append('labels')
    #         elif (low == 'layer posterior'):
    #             res.append('model/nCells/posterior')
    #         elif (low == 'layer depth posterior'):
    #             res.append('model/edges/posterior')
    #         elif (low == 'best data'):
    #             res.append('data')
    #         elif (low == 'x'):
    #             res.append('data/x')
    #         elif (low == 'y'):
    #             res.append('data/y')
    #         elif (low == 'z'):
    #             res.append('data/z')
    #         elif (low == 'elevation'):
    #             res.append('data/e')
    #         elif (low == 'observed data'):
    #             res.append('data/d')
    #         elif (low == 'predicted data'):
    #             res.append('data/p')
    #         elif (low == 'total error'):
    #             res.append('data/s')
    #         elif (low == '# of systems'):
    #             res.append('nsystems')
    #         elif (low == 'additive error'):
    #             res.append('data/additive_error')
    #         elif (low == 'relative error'):
    #             res.append('data/relative_error')
    #         elif (low == 'best model'):
    #             res.append('model')
    #         elif (low == 'meaninterp'):
    #             res.append('meaninterp')
    #         elif (low == 'bestinterp'):
    #             res.append('bestinterp')
    #         elif (low == 'opacityinterp'):
    #             res.append('opacityinterp')
    #         elif (low == '# layers'):
    #             res.append('model/nCells')
    #         elif (low == 'current data'):
    #             res.append('data')
    #         elif (low == 'hit map'):
    #             res.append('model/values/posterior')
    #         elif (low == 'hitmap/y'):
    #             res.append('model/values/posterior/y')
    #         elif (low == 'doi'):
    #             res.append('doi')
    #         elif (low == 'data misfit'):
    #             res.append('phids')
    #         elif (low == 'relative error posterior'):
    #             if (nSys is None): nSys = hdfRead.readKeyFromFile(self.hdf_file, self.fName, '/','nsystems')
    #             for i in range(nSys):
    #                 res.append('data/relative_error/posterior' +str(i))
    #         elif (low == 'additive error posterior'):
    #             if (nSys is None): nSys = hdfRead.readKeyFromFile(self.hdf_file, self.fName, '/','nsystems')
    #             for i in range(nSys):
    #                 res.append('data/additive_error/posterior' +str(i))
    #         elif (low == 'inversion time'):
    #             res.append('invtime')
    #         elif (low == 'saving time'):
    #             res.append('savetime')
    #         else:
    #             assert False, self.possibleAttributes(attr)
    #     return res


    def possibleAttributes(self, askedFor=""):
        print("====================================================\n"+
              "Incorrect attribute requested " + askedFor + "\n" +
              "====================================================\n"+
              "Possible Attribute options to read in \n" +
              "iteration # \n" +
              "# of markov chains \n" +
              "burned in\n" +
              "burn in # \n" +
              "data multiplier \n" +
              "layer posterior \n" +
              "height posterior \n" +
              "layer depth posterior \n" +
              "best data \n" +
              "fiducials" +
              "x\n" +
              "y\n" +
              "z\n" +
              "elevation\n" +
              "observed data" +
              "predicted data" +
              "total error" +
              "# of systems\n" +
              "relative error\n" +
              "best model \n" +
              "# layers \n" +
              "current data \n" +
              "hit map \n" +
              "doi \n"+
              "data misfit \n" +
              "relative error posterior\n" +
              "additive error posterior\n" +
              "inversion time\n" +
              "saving time\n"+
              "labels"+
              "marginal_probability"+
              "====================================================\n")

    def createHdf(self, parent, inference1d):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """

        parent = inference1d.createHdf(parent, add_axis=self.data.fiducial)

        # Write the sorted fiducials
        fiducials = sort(self.data.fiducial)
        fiducials.writeHdf(parent, 'data/fiducial')

        # Write the line number
        self.data.line_number.writeHdf(parent, 'data/line_number')

        return parent

    @classmethod
    def fromHdf(cls, grp, prng, mode = "r", world=None, **kwargs):
        assert mode != 'w', ValueError("Don't use mode = 'w' when reading!")
        if isinstance(grp, (Path, str)):
            tmp = {}
            if world is not None:
                tmp['driver'] = 'mpio'
                tmp['comm'] = world

            grp = h5py.File(grp, mode, **tmp)

        data = hdfRead.read_item(grp['data'], **kwargs)

        self = cls(data, prng=prng, world=world)
        self.mode = mode
        self.hdf_file = grp

        return self


    def plot_summary(self, **kwargs):

        pp = self.parameter_posterior()
        mm = self.mean_parameters()

        kwargs['x'] = kwargs.get('x', 'x')

        axes = kwargs.get('axes', None)
        if axes is None:
            nrows = 3
            fig = kwargs.pop('fig', plt.figure(figsize=(20, 10)))

            gs = fig.add_gridspec(ncols=2, nrows=6, hspace=2.0)

            ax = fig.add_subplot(gs[0, 0])
        else:
            ax = axes.pop(0)

        self.plot_burned_in(x = kwargs['x'], high=self.data.channel_saturation, ax=ax)
        self.plot_channel_saturation(linewidth=1.0, x = kwargs['x'], ax=ax, flipX=kwargs.get('flipX', False), wrap_ylabel=True)

        ax = fig.add_subplot(gs[1, 0], sharex=ax) if axes is None else axes.pop(0)
        self.plot_k_layers(x = kwargs['x'], wrap_clabel=True)

        ax0 = fig.add_subplot(gs[0, 1], sharex=ax) if axes is None else axes.pop(0)
        self.plot_mean_model(ax=ax0, wrap_clabel=True, **kwargs);
        self.plot_data_elevation(linewidth=0.3, x = kwargs['x'], ax=ax0);
        self.plot_elevation(linewidth=0.3, x = kwargs['x'], ax=ax0);
        ylim = ax0.get_ylim()

        # By adding the useVariance keyword, we can make regions of lower confidence more transparent
        ax = fig.add_subplot(gs[1, 1], sharex=ax0, sharey=ax0) if axes is None else axes.pop(0)
        self.plot_mode_model(mask_by_confidence=True, ax=ax, wrap_clabel=True, **kwargs);
        self.plot_data_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);
        self.plot_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);

        ax = fig.add_subplot(gs[2, 1], sharex=ax0, sharey=ax0) if axes is None else axes.pop(0)
        self.plot_best_model(ax=ax, wrap_clabel=True, **kwargs);
        self.plot_data_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);
        self.plot_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);
        ax.set_ylim(ylim)

        kwargs.pop('vmin', None)
        kwargs.pop('vmax', None)

        ax = fig.add_subplot(gs[3, 1], sharex=ax0, sharey=ax0) if axes is None else axes.pop(0)
        plt.title('5%')
        # a = self.percentile(0.05)
        # b = self.percentile(0.95)

        # kwargs['vmin'] = minimum(nanmin(a.values), nanmin(b.values))
        # kwargs['vmax'] = maximum(nanmax(a.values), nanmax(b.values))



        self.plot_percentile(percent=0.05, ax=ax, wrap_clabel=True, **kwargs)
        self.plot_data_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);
        self.plot_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);

        ax = fig.add_subplot(gs[4, 1], sharex=ax0, sharey=ax0) if axes is None else axes.pop(0)
        plt.title('50%')
        self.plot_percentile(percent=0.5, ax=ax, wrap_clabel=True, **kwargs)
        self.plot_data_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);
        self.plot_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);

        ax = fig.add_subplot(gs[5, 1], sharex=ax0, sharey=ax0) if axes is None else axes.pop(0)
        plt.title('95%')
        self.plot_percentile(percent=0.95, ax=ax, wrap_clabel=True, **kwargs)
        self.plot_data_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);
        self.plot_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);

        # del kwargs['vmin']
        # del kwargs['vmax']

        ################################################################################
        # Now we can start plotting some more interesting posterior properties.
        # How about the confidence?
        ax = fig.add_subplot(gs[2, 0], sharex=ax0, sharey=ax0) if axes is None else axes.pop(0)
        self.plot_confidence(x = kwargs['x'], ax=ax);
        self.plot_data_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);
        self.plot_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);

        ################################################################################
        # We can take the interface depth posterior for each data point,
        # and display an interface probability cross section
        # This posterior can be washed out, so the clim_scaling keyword lets me saturate
        # the top and bottom 0.5% of the colour range
        ax = fig.add_subplot(gs[3, 0], sharex=ax0, sharey=ax0) if axes is None else axes.pop(0)
        plt.title('P(Interface)')
        self.plot_interfaces(cmap='Greys', clim_scaling=0.5, x = kwargs['x'], ax=ax);
        self.plot_data_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);
        self.plot_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);

        ax = fig.add_subplot(gs[4, 0], sharex=ax0, sharey=ax0) if axes is None else axes.pop(0)
        self.plot_entropy(cmap='Greys', clim_scaling=0.5, x = kwargs['x'], ax=ax);
        self.plot_data_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);
        self.plot_elevation(linewidth=0.3, x = kwargs['x'], ax=ax);

        if kwargs.get('flipX', False):
            ax.invert_xaxis()
        if kwargs.get('flipY', False):
            ax.invert_yaxis()

        return fig
