import os
import numpy as np
import numpy.ma as ma
import h5py
from cached_property import cached_property
from datetime import timedelta
from ..classes.core.myObject import myObject
from ..classes.core import StatArray
from ..classes.statistics.Distribution import Distribution
from ..classes.statistics.mixStudentT import mixStudentT
from ..classes.statistics.Histogram1D import Histogram1D
from ..classes.statistics.Histogram2D import Histogram2D
from ..classes.statistics.Hitmap2D import Hitmap2D
from ..classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from ..classes.mesh.TopoRectilinearMesh2D import TopoRectilinearMesh2D
from ..classes.data.dataset.FdemData import FdemData
from ..classes.data.dataset.TdemData import TdemData
from ..classes.model.Model1D import Model1D
from ..base.HDF import hdfRead
from ..base import customPlots as cP
from ..base import customFunctions as cF
from ..base import fileIO as fIO
from ..base.MPI import loadBalance1D_shrinkingArrays
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspecA
from os.path import (split, join)
from geobipy.src.inversion.Inference1D import Inference1D
import progressbar

try:
    from pyvtk import VtkData, UnstructuredGrid, CellData, Scalars
except:
    pass

class Inference2D(myObject):
    """ Class to define results from EMinv1D_MCMC for a line of data """
    def __init__(self, hdf5_file_path=None, system_file_path=None, hdfFile=None, mode='r+', world=None):
        """ Initialize the lineResults """
        if (hdf5_file_path is None): return

        assert not system_file_path is None, Exception("Please also specify the path to the system file")

        self._burnedIn = None
        self._marginalProbability = None
        self.range = None
        self.systemFilepath = system_file_path
        self._zPosterior = None

        self.fName = hdf5_file_path
        self.directory = split(hdf5_file_path)[0]
        self.line = np.float64(os.path.splitext(split(hdf5_file_path)[1])[0])

        self._world = None
        self.hdfFile = None
        if (hdfFile is None): # Open the file
            self.open(mode, world)
        else:
            self.hdfFile = hdfFile

    @property
    def world(self):
        return self._world

    def open(self, mode='r+', world=None):
        """ Check whether the file is open """
        try:
            self.hdfFile.attrs
        except:

            if world is None:
                self.hdfFile = h5py.File(self.fName, mode)
            else:
                self._world = world
                self.hdfFile = h5py.File(self.fName, mode, driver='mpio', comm=world)


    def close(self):
        """ Check whether the file is open """
        if (self.hdfFile is None): return
        try:
            self.hdfFile.close()
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


    def crossplotErrors(self, system=0, **kwargs):
        """ Create a crossplot of the relative errors against additive errors for the most probable data point, for each data point along the line """
        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','none')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.0)

        if (self.nSystems > 1):
            r = range(self.nSystems)
            for i in r:
                fc = cP.wellSeparated[i+2]
                cP.plot(x=self.relativeError[:,i], y=self.additiveError[:,i], c=fc,
                    alpha = 0.7,label='System ' + str(i + 1), **kwargs)

            plt.legend()

        else:
            fc = cP.wellSeparated[2]
            cP.plot(x=self.relativeError, y=self.additiveError, c=fc,
                    alpha = 0.7,label='System ' + str(1), **kwargs)

        cP.xlabel(self.relativeError.getNameUnits())
        cP.ylabel(self.additiveError.getNameUnits())


    @cached_property
    def additiveError(self):
        """ Get the Additive error of the best data points """
        return self.getAttribute('Additive Error')


    @property
    def additiveErrorPosteriors(self):
        return self.data._addErr.posterior


    def compute_additive_error_opacity(self, percent=95.0, log=None):

        self.addErr_opacity = self.compute_posterior_opacity(self.additiveErrorPosteriors, percent, log)


    def compute_relative_error_opacity(self, percent=95.0, log=None):

        self.relErr_opacity = self.compute_posterior_opacity(self.relativeErrorPosteriors, percent, log)


    def compute_posterior_opacity(self, posterior, percent=95.0, log=None):
        opacity = StatArray.StatArray(np.zeros(self.nPoints))

        for i in range(self.nPoints):
            h = Histogram1D(bins = self.additiveErrorPosteriors._cellEdges + self.additiveErrorPosteriors.relativeTo[i])
            h._counts[:] = self.additiveErrorPosteriors.counts[i, :]
            opacity[i] = h.credibleRange(percent, log)

        opacity = opacity.normalize()
        return 1.0 - opacity



    @cached_property
    def bestData(self):
        """ Get the best data """

        attr = self._attrTokey('best data')
        dtype = self.hdfFile[attr[0]].attrs['repr']
        if "FdemDataPoint" in dtype:
            bestData = FdemData().fromHdf(self.hdfFile[attr[0]])
        elif "TdemDataPoint" in dtype:
            bestData = TdemData().fromHdf(self.hdfFile[attr[0]], systemFilepath = self.systemFilepath)
        return bestData


    @cached_property
    def bestParameters(self):
        """ Get the best model of the parameters """
        return StatArray.StatArray(self.getAttribute('bestinterp'), dtype=np.float64, name=self.parameterName, units=self.parameterUnits).T


    @cached_property
    def burned_in(self):
        return StatArray.StatArray(self.getAttribute('burned in'))


    @cached_property
    def credibleLower(self):
        # Read in the opacity if present
        if "credible_lower" in self.hdfFile.keys():
            return StatArray.StatArray().fromHdf(self.hdfFile['credible_lower'])
        else:
            cl, _ = self.computeCredibleInterval(log=10)
            return cl


    @cached_property
    def credibleUpper(self):
        # Read in the opacity if present
        if "credible_upper" in self.hdfFile.keys():
            return StatArray.StatArray().fromHdf(self.hdfFile['credible_upper'])
        else:
            _, cu = self.computeCredibleInterval(log=10)
            return cu


    def computeCredibleInterval(self, percent=95.0, log=None, progress=False):

        s = 'percent={}'.format(percent)

        credibleLower = StatArray.StatArray(np.zeros(self.mesh.shape), '{}% Credible Interval'.format(100.0 - percent), self.parameterUnits)
        credibleUpper = StatArray.StatArray(np.zeros(self.mesh.shape), '{}% Credible Interval'.format(percent), self.parameterUnits)

        loc = 'currentmodel/par/posterior'
        a = np.asarray(self.hdfFile[loc+'/arr/data'])
        try:
            b = np.asarray(self.hdfFile[loc+'/x/data'])
        except:
            b = np.asarray(self.hdfFile[loc+'/x/x/data'])

        try:
            c = np.asarray(self.hdfFile[loc+'/y/data'])
        except:
            c = np.asarray(self.hdfFile[loc+'/y/x/data'])

        h = Hitmap2D(xBinCentres = StatArray.StatArray(b[0, :]), yBinCentres = StatArray.StatArray(c[0, :]))
        h._counts[:, :] = a[0, :, :]
        m, l, u = h.credibleIntervals(percent=percent, log=log)
        credibleLower[:, 0] = l
        credibleUpper[:, 0] = u

        print('Computing {}% Credible Intervals'.format(percent), flush=True)
        for i in progressbar.progressbar(range(1, self.nPoints)):
            h.x.xBinCentres = b[i, :]
            h._counts[:, :] = a[i, :, :]
            m, l, u = h.credibleIntervals(percent=percent, log=log)
            credibleLower[:, i] = l
            credibleUpper[:, i] = u

        key = 'credible_lower'
        if key in self.hdfFile.keys():
            credibleLower.writeHdf(self.hdfFile, key)
        else:
            credibleLower.toHdf(self.hdfFile, key)

        key = 'credible_upper'
        if key in self.hdfFile.keys():
            credibleUpper.writeHdf(self.hdfFile, key)
        else:
            credibleUpper.toHdf(self.hdfFile, key)

        return credibleLower, credibleUpper


    @property
    def credibleRange(self):
        """ Get the model parameter opacity using the credible intervals """
        return self.credibleUpper - self.credibleLower


    @cached_property
    def data(self):
        """ Get the best data """

        attr = self._attrTokey('current data')
        dtype = self.hdfFile[attr[0]].attrs['repr']
        if "FdemDataPoint" in dtype:
            currentData = FdemData().fromHdf(self.hdfFile[attr[0]])
        elif "TdemDataPoint" in dtype:
            currentData = TdemData().fromHdf(self.hdfFile[attr[0]], systemFilepath = self.systemFilepath)
        return currentData


    @cached_property
    def doi(self):
        if 'doi' in self.hdfFile.keys():
            return StatArray.StatArray(np.asarray(self.getAttribute('doi')), 'Depth of investigation', 'm')
        else:
            self.computeDOI()


    def computeDOI(self, percent=67.0, window=1):
        """ Get the DOI of the line depending on a percentage credible interval cutoff for each data point """

        if 'doi' in self.__dict__:
            del self.__dict__['doi']

        assert window > 0, ValueError("window must be >= 1")
        assert 0.0 < percent < 100.0, ValueError("Must have 0.0 < percent < 100.0")

        opacity = self.opacity

        nz = self.hitMap.z.nCells
        doi = StatArray.StatArray(np.full(self.nPoints, fill_value=self.hitMap.z.cellEdges[-1]), 'Depth of investigation', self.height.units)

        p = 0.01 * percent

        print('Computing Depth of Investigation', flush=True)

        for i in progressbar.progressbar(range(self.nPoints)):
            tmp = opacity[:, i]
            iCell = nz-1
            while tmp[iCell] < p and iCell >= 0:
                iCell -=1

            if iCell >= 0:
                doi[i] = self.hitMap.z.cellCentres[iCell-1]

        doiOut = doi
        if window > 1:
            buffer = np.int(0.5 * window)
            tmp = doi.rolling(np.mean, window)
            doiOut[buffer:-buffer] = tmp
            doiOut[:buffer] = tmp[0]
            doiOut[-buffer:] = tmp[-1]

        if 'doi' in self.hdfFile.keys():
            doiOut.writeHdf(self.hdfFile, 'doi')
        else:
            doiOut.toHdf(self.hdfFile, 'doi')

        return doiOut

    @property
    def easting(self):
        return self.mesh.x.cellCentres

    @property
    def northing(self):
        return self.mesh.y.cellCentres

    @property
    def depth(self):
        return self.mesh.z.cellCentres


    @cached_property
    def elevation(self):
        """ Get the elevation of the data points """
        return StatArray.StatArray(np.asarray(self.getAttribute('elevation')), 'Elevation', 'm')


    def extract1DModel(self, values, index=None, fiducial=None):
        """ Obtain the results for the given iD number """

        assert not (index is None and fiducial is None), Exception("Please specify either an integer index or a fiducial.")
        assert index is None or fiducial is None, Exception("Only specify either an integer index or a fiducial.")

        if not fiducial is None:
            assert fiducial in self.fiducials, ValueError("This fiducial {} is not available from this HDF5 file. The min max fids are {} to {}.".format(fiducial, self.fiducials.min(), self.fiducials.max()))
            # Get the point index
            i = self.fiducials.searchsorted(fiducial)
        else:
            i = index
            fiducial = self.fiducials[index]

        depth = self.mesh.z.cellEdges[:-1]
        parameter = values[:, i]

        return Model1D(self.mesh.z.nCells, depth=depth, parameters=parameter, hasHalfspace=False)


    def fiducialIndex(self, fiducial):

        if np.size(fiducial) == 1:
            return np.where(self.fiducials == fiducial)[0]

        fiducial = np.asarray(fiducial)
        idx = np.searchsorted(self.fiducials, fiducial)

        # Take care of out of bounds cases
        idx[idx==self.nPoints] = 0

        return idx[fiducial == self.fiducials[idx]]


    def _get(self, variable, index=None, reciprocateParameter=False, **kwargs):

        variable = variable.lower()
        assert variable in ['mean', 'best', 'interfaces', 'opacity', 'highestmarginal', 'marginalprobability'], ValueError("variable must be ['mean', 'best', 'interfaces', 'opacity', 'highestMarginal', 'marginalProbability']")

        if variable == 'mean':
            if reciprocateParameter:
                vals = np.divide(1.0, self.meanParameters)
                vals.name = 'Resistivity'
                vals.units = '$\Omega m$'
                return vals
            else:
                return self.meanParameters

        elif variable == 'best':
            if reciprocateParameter:
                vals = 1.0 / self.meanParameters
                vals.name = 'Resistivity'
                vals.units = '$\Omega m$'
                return vals
            else:
                return self.bestParameters

        if variable == 'interfaces':
            return self.interfaces

        if variable == 'opacity':
            return self.opacity

        if variable == 'highestmarginal':
            return self.highestMarginal

        if variable == 'marginalprobability':
            assert not index is None, ValueError('Please specify keyword "index" when requesting marginalProbability')
            return self.marginalProbability[:, :, index].T


    @cached_property
    def height(self):
        """Get the height of the observations. """
        return self.getAttribute('z')


    @cached_property
    def heightPosterior(self):
        zPosterior = self.getAttribute('height posterior')
        zPosterior.bins.name = 'Relative ' + zPosterior.bins.name

        return zPosterior


    @property
    def hitMap(self):
        """ Get the hitmaps for each data point """
        return self.getAttribute('Hit Map', index=0)


    @cached_property
    def fiducials(self):
        """ Get the id numbers of the data points in the line results file """
        try:
            return self.getAttribute('fiducials')
        except:
            return StatArray.StatArray(np.asarray(self.hdfFile.get('ids')), "fiducials")


    def fit_gaussian_mixture(self, intervals, **kwargs):

        distributions = []

        hm = self.hitMap.deepcopy()
        counts = np.asarray(self.hdfFile['currentmodel/par/posterior/arr/data'])

        # Bar = progressbar.ProgressBar()
        for i in range(self.nPoints):

            try:
                dpDistributions = hm.fitMajorPeaks(intervals, **kwargs)
                distributions.append(dpDistributions)
            except:
                pass

            hm._counts = counts[i, :, :]

        return distributions


    def fitMajorPeaks(self, intervals, **kwargs):
        """Fit distributions to the major peaks in each hitmap along the line.

        Parameters
        ----------
        intervals : array_like, optional
            Accumulate the histogram between these invervals before finding peaks

        """
        distributions = []

        hm = self.hitMap.deepcopy()
        counts = np.asarray(self.hdfFile['currentmodel/par/posterior/arr/data'])

        # Bar = progressbar.ProgressBar()
        for i in range(self.nPoints):

            try:
                dpDistributions = hm.fitMajorPeaks(intervals, **kwargs)
                distributions.append(dpDistributions)
            except:
                pass

            hm._counts = counts[i, :, :]

        return distributions


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

        world = self.world

        max_distributions = kwargs.get('max_distributions', 5)

        if intervals is None:
            intervals = self.hitMap.yBins

        nIntervals = np.size(intervals) - 1

        means = StatArray.StatArray([self.nPoints, nIntervals, max_distributions], "fit means")
        variances = StatArray.StatArray([self.nPoints, nIntervals, max_distributions], "fit variances")
        amplitudes = StatArray.StatArray([self.nPoints, nIntervals, max_distributions], "fit amplitudes")
        df = StatArray.StatArray([self.nPoints, nIntervals, max_distributions], "fit df")

        if external_files:
            hdfFile = h5py.File("Line_{}_fits.h5".format(self.line), 'w', driver='mpio', comm=world)

        else:
            hdfFile = self.hdfFile

        try:
            means.createHdf(hdfFile, "/means", fillvalue=np.nan)
            variances.createHdf(hdfFile, "/variances", fillvalue=np.nan)
            df.createHdf(hdfFile, "/degrees", fillvalue=np.nan)
            amplitudes.createHdf(hdfFile, "/amplitudes", fillvalue=np.nan)
        except:
            pass

        # Distribute the points amongst cores.
        starts, chunks = loadBalance1D_shrinkingArrays(self.nPoints, world.size)

        chunk = chunks[world.rank]
        i0 = starts[world.rank]
        i1 = i0 + chunk

        tBase = MPI.Wtime()
        t0 = tBase

        nUpdate = 1
        counter = 0

        nI = intervals.size - 1

        buffer = np.empty((nI, max_distributions))

        for i in range(i0, i1):

            hm = self.get_hitmap(i)

            if not np.all(hm.counts == 0):

                mixtures = hm.fit_estimated_pdf(**kwargs)

                buffer[:, :] = np.nan
                for j in range(nI):
                    mix = mixtures[j]
                    buffer[j, :mix.n_mixtures] = mix.means
                hdfFile['/means/data'][i, :, :] = buffer

                for j in range(nI):
                    mix = mixtures[j]
                    buffer[j, :mix.n_mixtures] = mix.variances
                hdfFile['/variances/data'][i, :, :] = buffer

                for j in range(nI):
                    mix = mixtures[j]
                    buffer[j, :mix.n_mixtures] = mix.degrees
                hdfFile['/degrees/data'][i, :, :] = buffer

                for j in range(nI):
                    mix = mixtures[j]
                    buffer[j, :mix.n_mixtures] = mix.amplitudes
                hdfFile['/amplitudes/data'][i, :, :] = buffer

            counter += 1
            if counter == nUpdate:
                print('rank {}, line/fiducial {}/{}, iteration {}/{},  time/dp {} h:m:s'.format(world.rank, self.line, self.fiducials[i], i-i0+1, chunk, str(timedelta(seconds=MPI.Wtime()-t0)/nUpdate)), flush=True)
                t0 = MPI.Wtime()
                counter = 0

        print('rank {} finished in {} h:m:s'.format(world.rank, str(timedelta(seconds=MPI.Wtime()-tBase))), flush=True)

        if external_files:
            hdfFile.close()


    def depthSlice(self, depth, values, **kwargs):
        """ Obtain a slice at depth from values

        Parameters
        ----------
        depth : float or array_like
            If float: The depth at which to obtain the slice
            If arraylike: length 2 array of an interval over which to average.
        values : array_like
            Values of shape self.mesh.shape from which to obtain the slice.

        Returns
        -------
        out : geobipy.StatArray
            The slice at depth.

        """

        if np.size(depth) > 1:
            assert np.size(depth) == 2, ValueError("depth must be a scalar or size 2 array.")
            depth.sort()
            assert np.all(depth < self.mesh.z.cellEdges[-1]), 'Depths must be lees than max depth {}'.format(self.mesh.z.cellEdges[-1])
            assert depth[0] <= depth[1], ValueError("Depths must be monotonically increasing")
        else:
            assert depth < self.mesh.z.cellEdges[-1], 'Depth must be lees than max depth {}'.format(self.mesh.z.cellEdges[-1])

        assert np.all(np.shape(values)[-2:] == self.mesh.shape), ValueError("values must have shape {} but have shape {}".format(self.mesh.shape, np.shape(values)))

        if np.size(depth) > 1:
            cell1 = self.mesh.z.cellIndex(depth[0])
            cell2 = self.mesh.z.cellIndex(depth[1])
            out = np.mean(values[cell1:cell2+1, :], axis = 0)
        else:
            cell1 = self.mesh.z.cellIndex(depth)
            out = values[cell1, :]

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

        assert np.all(np.shape(values) == self.mesh.shape), ValueError("values must have shape {}".fomat(self.mesh.shape))

        out = np.full(self.nPoints, fill_value=np.nan)

        if np.size(elevation) > 1:

            for i in range(self.nPoints):
                tmp = self.elevation[i] - elevation
                if tmp[1] < self.mesh.z.cellEdges[-1] and tmp[0] > self.mesh.z.cellEdges[0]:
                    cell1 = self.mesh.z.cellIndex(tmp[1], clip=True)
                    cell2 = self.mesh.z.cellIndex(tmp[0], clip=True)

                    out[i] = np.mean(values[cell1:cell2+1, i])

        else:

            for i in range(self.nPoints):
                tmp = self.elevation[i] - elevation
                if tmp > self.mesh.z.cellEdges[0] and tmp < self.mesh.z.cellEdges[-1]:
                    cell1 = self.mesh.z.cellIndex(tmp, clip=True)

                    out[i] = values[cell1, i]

        return out


    def identifyPeaks(self, depths, nBins = 250, width=4, limits=None):
        """Identifies peaks in the parameter posterior for each depth in depths.

        Parameters
        ----------
        depths : array_like
            Depth intervals to identify peaks between.

        Returns
        -------

        """

        from scipy.signal import find_peaks

        assert np.size(depths) > 2, ValueError("Depths must have size > 1.")

        tmp = self.lineHitmap.intervalStatistic(axis=0, intervals = depths, statistic='sum')

        depth = np.zeros(0)
        parameter = np.zeros(0)

        # # Bar = progressbar.ProgressBar()
        # # for i in Bar(range(self.nPoints)):
        for i in range(tmp.y.nCells):
            peaks, _ = find_peaks(tmp.counts[i, :],  width=width)
            values = tmp.x.cellCentres[peaks]
            if not limits is None:
                values = values[(values > limits[0]) & (values < limits[1])]
            parameter = np.hstack([parameter, values])
            depth = np.hstack([depth, np.full(values.size, fill_value=0.5*(depths[i]+depths[i+1]))])

        return np.asarray([depth, parameter]).T


    @cached_property
    def interfaces(self):
        """ Get the layer interfaces from the layer depth histograms """
        maxCount = self.interfacePosterior.counts.max()
        # values = np.vstack([self.interfacePosterior.counts.T, self.interfacePosterior.counts.T[-1, :]])
        return StatArray.StatArray(self.interfacePosterior.counts.T / np.float64(maxCount), "interfaces", "")



    @cached_property
    def interfacePosterior(self):
        return self.getAttribute('layer depth posterior')


    @cached_property
    def labels(self):
        return self.getAttribute('labels')


    @cached_property
    def lineHitmap(self):
        """ """
        # Read in the opacity if present
        # if "line_hitmap" in self.hdfFile.keys():
        #     return Hitmap2D().fromHdf(self.hdfFile['line_hitmap'])
        # else:
        return self.computeLineHitmap()


    def computeLineHitmap(self, nBins=250, log=10):

        # if 'lineHitmap' in self.__dict__:
        #     del self.__dict__['lineHitmap']

        # First get the min max of the parameter hitmaps
        x0 = np.log10(self.minParameter)
        x1 = np.log10(self.maxParameter)

        counts = self.hdfFile['currentmodel/par/posterior/arr/data']

        xBins = StatArray.StatArray(np.logspace(x0, x1, nBins+1), self.parameterName, units = self.parameterUnits)

        lineHitmap = Histogram2D(xBins=xBins, yBins=self.mesh.z.cellEdges)

        parameters = RectilinearMesh1D().fromHdf(self.hdfFile['currentmodel/par/posterior/x'])

        # Bar = progressbar.ProgressBar()
        # for i in Bar(range(self.nPoints)):
        for i in range(self.nPoints):
            p = RectilinearMesh1D(cellEdges=parameters.cellEdges[i, :])

            pj = lineHitmap.x.cellIndex(p.cellCentres, clip=True)

            cTmp = counts[i, :, :]

            lineHitmap._counts[:, pj] += cTmp

        # if 'line_hitmap' in self.hdfFile.keys():
        #     del self.hdfFile['line_hitmap']

        # lineHitmap.toHdf(self.hdfFile, 'line_hitmap')

        return lineHitmap


    @property
    def maxParameter(self):
        """ Get the mean model of the parameters """
        tmp = np.asarray(self.hdfFile["currentmodel/par/posterior/x/x/data"][:, -1])
        return tmp.max()


    @cached_property
    def meanParameters(self):
        return StatArray.StatArray(self.getAttribute('meaninterp').T, name=self.parameterName, units=self.parameterUnits)


    @cached_property
    def mesh(self):
        """Get the 2D topo fitting rectilinear mesh. """

        if (self.hitMap is None):
            tmp = self.getAttribute('hitmap/y', index=0)
            try:
                tmp = RectilinearMesh1D(cellCentres=tmp.cellCentres, edgesMin=0.0)
            except:
                tmp = RectilinearMesh1D(cellCentres=tmp, edgesMin=0.0)
        else:
            tmp = self.hitMap.y
            try:
                tmp = RectilinearMesh1D(cellCentres=tmp.cellCentres, edgesMin=0.0)
            except:
                tmp = RectilinearMesh1D(cellCentres=tmp, edgesMin=0.0)

        return TopoRectilinearMesh2D(xCentres=self.x, yCentres=self.y, zEdges=tmp.cellEdges, heightCentres=self.elevation)


    @property
    def minParameter(self):
        """ Get the mean model of the parameters """
        tmp = np.asarray(self.hdfFile["currentmodel/par/posterior/x/x/data"][:, 0])
        return tmp.min()


    @cached_property
    def modeParameter(self):
        """ """
        # Read in the opacity if present
        if "mode_parameter" in self.hdfFile.keys():
            return StatArray.StatArray().fromHdf(self.hdfFile['mode_parameter'])
        else:
            return self.computeModeParameter()


    def computeModeParameter(self):

        if 'modeParameter' in self.__dict__:
            del self.__dict__['modeParameter']

        modeParameter = StatArray.StatArray(np.zeros(self.mesh.shape), self.parameterName, self.parameterUnits)

        loc = 'currentmodel/par/posterior'
        a = np.asarray(self.hdfFile[loc+'/arr/data'])
        try:
            b = np.asarray(self.hdfFile[loc+'/x/data'])
        except:
            b = np.asarray(self.hdfFile[loc+'/x/x/data'])

        try:
            c = np.asarray(self.hdfFile[loc+'/y/data'])
        except:
            c = np.asarray(self.hdfFile[loc+'/y/x/data'])

        h = Hitmap2D(xBinCentres = StatArray.StatArray(b[0, :]), yBinCentres = StatArray.StatArray(c[0, :]))
        h._counts[:, :] = a[0, :, :]
        modeParameter[:, 0] = h.mode()

        print('Computing Mode Parameter', flush=True)
        for i in progressbar.progressbar(range(1, self.nPoints)):
            h.x.xBinCentres = b[i, :]
            h._counts[:, :] = a[i, :, :]
            modeParameter[:, i] = h.mode()


        if 'mode_parameter' in self.hdfFile.keys():
            modeParameter.writeHdf(self.hdfFile, 'mode_parameter')
        else:
            modeParameter.toHdf(self.hdfFile, 'mode_parameter')

        return modeParameter


    @cached_property
    def nLayers(self):
        """ Get the number of layers in the best model for each data point """
        return StatArray.StatArray(self.getAttribute('# Layers'), '# of Cells')

    @property
    def nPoints(self):
        return self.fiducials.size


    @cached_property
    def nSystems(self):
        """ Get the number of systems """
        return self.getAttribute('# of systems')


    @cached_property
    def opacity(self):
        """ Get the model parameter opacity using the credible intervals """
        if "opacity" in self.hdfFile.keys():
            return StatArray.StatArray().fromHdf(self.hdfFile['opacity'])
        else:
            return self.computeOpacity()


    def computeOpacity(self, percent=95.0, log=10, multiplier=0.5):


        if 'opacity' in self.__dict__:
            del self.__dict__['opacity']

        tmp = self.credibleRange

        mn = np.nanmin(tmp)
        mx = np.nanmax(tmp)

        opacity = 1.0 - StatArray.StatArray(tmp, "Opacity", "").normalize(axis=0)

        if 'opacity' in self.hdfFile.keys():
            opacity.writeHdf(self.hdfFile, 'opacity')
        else:
            opacity.toHdf(self.hdfFile, 'opacity')


        return opacity


    @property
    def parameterName(self):

        return self.hitMap.x.cellCentres.name


    @property
    def parameterUnits(self):

        return self.hitMap.x.cellCentres.units


    def percentageParameter(self, value, depth=None, depth2=None, progress=False):

        # Get the depth grid
        if (not depth is None):
            assert depth <= self.mesh.z.cellEdges[-1], 'Depth is greater than max depth '+str(self.mesh.z.cellEdges[-1])
            j = self.mesh.z.cellIndex(depth)
            k = j+1
            if (not depth2 is None):
                assert depth2 <= self.mesh.z.cellEdges[-1], 'Depth2 is greater than max depth '+str(self.mesh.z.cellEdges[-1])
                assert depth <= depth2, 'Depth2 must be >= depth'
                k = self.mesh.z.cellIndex(depth2)

        percentage = StatArray.StatArray(np.empty(self.nPoints), name="Probability of {} > {:0.2f}".format(self.parameterName, value), units = self.parameterUnits)

        if depth:
            counts = self.hdfFile['currentmodel/par/posterior/arr/data'][:, j:k, :]
            # return StatArray.StatArray(np.sum(counts[:, :, pj:]) / np.sum(counts) * 100.0, name="Probability of {} > {:0.2f}".format(self.meanParameters.name, value), units = self.meanParameters.units)
        else:
            counts = self.hdfFile['currentmodel/par/posterior/arr/data']

        parameters = RectilinearMesh1D().fromHdf(self.hdfFile['currentmodel/par/posterior/x'])

        Bar = progressbar.ProgressBar()
        print('Computing P(X > value)', flush=True)
        for i in Bar(range(self.nPoints)):
            p = RectilinearMesh1D(cellEdges=parameters.cellEdges[i, :])
            pj = p.cellIndex(value)

            cTmp = counts[i, :, :]

            percentage[i] = np.sum(cTmp[:, pj:]) / cTmp.sum()

        return percentage


    @cached_property
    def relativeError(self):
        """ Get the Relative error of the best data points """
        return self.getAttribute('Relative Error')


    @property
    def relativeErrorPosteriors(self):
        """ Get the Relative error of the best data points """
        return self.data._relErr.posterior


    def get_hitmap(self, index=None, fiducial=None):

        assert not (index is None and fiducial is None), Exception("Please specify either an integer index or a fiducial.")
        assert index is None or fiducial is None, Exception("Only specify either an integer index or a fiducial.")

        if not fiducial is None:
            assert fiducial in self.fiducials, ValueError("This fiducial {} is not available from this HDF5 file. The min max fids are {} to {}.".format(fiducial, self.fiducials.min(), self.fiducials.max()))
            # Get the point index
            i = self.fiducials.searchsorted(fiducial)
        else:
            i = index
            fiducial = self.fiducials[index]

        return self.getAttribute('Hit map', index = i)


    def inference_1d(self, index=None, fiducial=None, reciprocateParameter=False):
        """ Obtain the results for the given iD number """

        assert not (index is None and fiducial is None), Exception("Please specify either an integer index or a fiducial.")
        assert index is None or fiducial is None, Exception("Only specify either an integer index or a fiducial.")

        if not fiducial is None:
            assert fiducial in self.fiducials, ValueError("This fiducial {} is not available from this HDF5 file. The min max fids are {} to {}.".format(fiducial, self.fiducials.min(), self.fiducials.max()))
            # Get the point index
            i = self.fiducials.searchsorted(fiducial)
        else:
            i = index
            fiducial = self.fiducials[index]

        hdfFile = self.hdfFile

        R = Inference1D(reciprocateParameter=reciprocateParameter).fromHdf(hdfFile, index=i, systemFilePath=self.systemFilepath)

        return R



    @cached_property
    def totalError(self):
        """ Get the total error of the best data points """
        return self.getAttribute('Total Error')


    @cached_property
    def x(self):
        """ Get the X co-ordinates (Easting) """
        x = self.getAttribute('x')
        if x.name in [None, '']:
            x.name = 'Easting'
        if x.units in [None, '']:
            x.units = 'm'

        return x


    def x_axis(self, axis, centres=False):
        return self.mesh.getXAxis(axis, centres=centres)


    @cached_property
    def y(self):
        """ Get the Y co-ordinates (Easting) """
        y = self.getAttribute('y')

        if y.name in [None, '']:
            y.name = 'Northing'
        if y.units in [None, '']:
            y.units = 'm'

        return y


    def pcolorDataResidual(self, abs=False, **kwargs):
        """ Plot a channel of data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=False)

        values = self.bestData.deltaD.T

        if abs:
            values = values.abs()

        cP.pcolor(values, x=xtmp, y=StatArray.StatArray(np.arange(self.bestData.predictedData.shape[1]), name='Channel'), **kwargs)


    def pcolorObservedData(self, **kwargs):
        """ Plot a channel of data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=False)

        cP.pcolor(self.bestData.data.T, x=xtmp, y=StatArray.StatArray(np.arange(self.bestData.predictedData.shape[1]), name='Channel'), **kwargs)


    def pcolorPredictedData(self, **kwargs):
        """ Plot a channel of data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=False)

        cP.pcolor(self.bestData.predictedData.T, x=xtmp, y=StatArray.StatArray(np.arange(self.bestData.predictedData.shape[1]), name='Channel'), **kwargs)


    def plotPredictedData(self, channel=None, **kwargs):
        """ Plot a channel of the best predicted data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=True)

        if channel is None:
            channel = np.s_[:]

        cP.plot(xtmp, self.bestData.predictedData[:, channel], **kwargs)


    def plotDataElevation(self, **kwargs):
        """ Adds the data elevations to a plot """

        xAxis = kwargs.pop('xAxis', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color','k')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.5)

        xtmp = self.mesh.getXAxis(xAxis, centres=False)

        cP.plot(xtmp, self.height.edges() + self.elevation.edges(), **kwargs)

        if (labels):
            cP.xlabel(xtmp.getNameUnits())
            cP.ylabel('Elevation (m)')


    def plotDataResidual(self, channel=None, abs=False, **kwargs):
        """ Plot a channel of the observed data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=True)

        if channel is None:
            channel = np.s_[:]

        values = self.bestData.deltaD[:, channel]

        if abs:
            values = values.abs()

        cP.plot(xtmp, values, **kwargs)


    def plotDoi(self, percent=67.0, window=1, **kwargs):

        xAxis = kwargs.pop('xAxis', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color','k')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.5)

        xtmp = self.mesh.getXAxis(xAxis, centres=True)

        (self.elevation - self.doi).plot(x=xtmp, **kwargs)


    def plotElevation(self, **kwargs):

        xAxis = kwargs.pop('xAxis', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color','k')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.5)

        self.mesh.plotHeight(xAxis=xAxis, **kwargs)

        # if (labels):
        #     cP.xlabel(xtmp.getNameUnits())
        #     cP.ylabel('Elevation (m)')


    def plotHeightPosteriors(self, **kwargs):
        """ Plot the horizontally stacked elevation histograms for each data point along the line """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis)

        post = self.heightPosterior


        c = post.counts.T
        c = np.divide(c, np.max(c,0), casting='unsafe')

        ax1 = plt.subplot(111)
        c.pcolor(xtmp, y=post.bins, noColorbar=True, **kwargs)

        ax2 = ax1.twinx()

        c = cP.wellSeparated[0]
        post.relativeTo.plot(xtmp.internalEdges(), c=c, ax=ax2, label=self.bestData.z.getNameUnits())

        ax2.set_ylabel(self.bestData.z.getNameUnits(), color=c)
        ax2.tick_params(axis='y', labelcolor=c)
        plt.legend()

        cP.title('Data height posterior distributions')


    def plotHighlightedObservationLocations(self, fiducial, **kwargs):

        labels = kwargs.pop('labels', True)
        kwargs['marker'] = kwargs.pop('marker','*')
        kwargs['color'] = kwargs.pop('color',cP.wellSeparated[1])
        kwargs['linestyle'] = kwargs.pop('linestyle','none')
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth','0.1')
        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis)

        i = self.fiducials.searchsorted(fiducial)

        tmp = self.height.reshape(self.height.size) + self.elevation

        plt.plot(xtmp[i], tmp[i], **kwargs)

        if (labels):
            cP.xlabel(xtmp.getNameUnits())
            cP.ylabel('Elevation (m)')


    def plotKlayers(self, **kwargs):
        """ Plot the number of layers in the best model for each data point """
        xAxis = kwargs.pop('xAxis', 'x')
        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth', 1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','none')

        xtmp = self.mesh.getXAxis(xAxis)
        self.nLayers.plot(xtmp, **kwargs)
        # cP.ylabel(self.nLayers.getNameUnits())
        cP.title('# of Layers in Best Model')


    def plotKlayersPosteriors(self, **kwargs):
        """ Plot the horizontally stacked elevation histograms for each data point along the line """

        post = self.getAttribute('layer posterior')

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis)

        c = post.counts.T
        c = np.divide(c, np.max(c,0), casting='unsafe')
        ax, pm, cb = c.pcolor(xtmp, y=post.binCentres[0, :], **kwargs)
        cP.title('# of Layers posterior distributions')


    def plotAdditiveError(self, **kwargs):
        """ Plot the relative errors of the data """
        xAxis = kwargs.pop('xAxis', 'x')
        m = kwargs.pop('marker','o')
        ms = kwargs.pop('markersize',5)
        mfc = kwargs.pop('markerfacecolor',None)
        mec = kwargs.pop('markeredgecolor','k')
        mew = kwargs.pop('markeredgewidth',1.0)
        ls = kwargs.pop('linestyle','-')
        lw = kwargs.pop('linewidth',1.0)

        xtmp = self.mesh.getXAxis(xAxis, centres=True)

        if (self.nSystems > 1):
            r = range(self.nSystems)
            for i in r:
                fc = cP.wellSeparated[i+2]
                cP.plot(xtmp, y=self.additiveError[:,i],
                    marker=m,markersize=ms,markerfacecolor=mfc,markeredgecolor=mec,markeredgewidth=mew,
                    linestyle=ls,linewidth=lw,c=fc,
                    alpha = 0.7,label='System ' + str(i + 1), **kwargs)
            plt.legend()
        else:
            fc = cP.wellSeparated[2]
            cP.plot(xtmp, y=self.additiveError,
                    marker=m,markersize=ms,markerfacecolor=mfc,markeredgecolor=mec,markeredgewidth=mew,
                    linestyle=ls,linewidth=lw,c=fc,
                    alpha = 0.7,label='System ' + str(1), **kwargs)

        # cP.xlabel(xtmp.getNameUnits())
        # cP.ylabel(self.additiveError.getNameUnits())



    def plotAdditiveErrorPosteriors(self, system=0, **kwargs):
        """ Plot the distributions of additive errors as an image for all data points in the line """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis)

        if self.nSystems > 1:
            post = self.additiveErrorPosteriors[system]
        else:
            post = self.additiveErrorPosteriors

        c = post.counts.T
        c = np.divide(c, np.max(c, 0), casting='unsafe')

        # if post.isRelative:
        #     if np.all(post.relativeTo == post.relativeTo[0]):
        #         y = post.binCentres + post.relativeTo[0]
        #     else:
        #         stuff = 2
        # else:
        y = post._cellCentres
        c.pcolor(xtmp, y=y, **kwargs)
        cP.title('Additive error posterior distributions for system {}'.format(system))


    def plotConfidence(self, **kwargs):
        """ Plot the opacity """
        kwargs['cmap'] = kwargs.get('cmap', 'plasma')
        ax, pm, cb = self.plotXsection(values = self.opacity, **kwargs)

        cb.ax.set_yticklabels(['Less', '', '', '', '', 'More'])
        cb.set_label("Confidence")


    def plotError2DJointProbabilityDistribution(self, index, system=0, **kwargs):
        """ For a given index, obtains the posterior distributions of relative and additive error and creates the 2D joint probability distribution """

        # Read in the histogram of relative error for the data point
        rel = self.getAttribute('Relative error histogram', index=index)
        # Read in the histogram of additive error for the data point
        add = self.getAttribute('Additive error histogram', index=index)

        joint = Histogram2D()
        joint.create2DjointProbabilityDistribution(rel[system],add[system])

        joint.pcolor(**kwargs)


    def plotInterfaces(self, cut=0.0, **kwargs):
        """ Plot a cross section of the layer depth histograms. Truncation is optional. """

        kwargs['noColorbar'] = kwargs.pop('noColorbar', True)

        self.plotXsection(values=self.interfaces, **kwargs)



    def plotObservedData(self, channel=None, **kwargs):
        """ Plot a channel of the observed data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=True)

        if channel is None:
            channel = np.s_[:]

        cP.plot(xtmp, self.bestData.data[:, channel], **kwargs)


    def plotOpacity(self, **kwargs):
        """ Plot the opacity """
        self.plotXsection(values = self.opacity, **kwargs)


    def plotRelativeErrorPosteriors(self, system=0, **kwargs):
        """ Plot the distributions of relative errors as an image for all data points in the line """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis)

        if self.nSystems > 1:
            post = self.relativeErrorPosteriors[system]
        else:
            post = self.relativeErrorPosteriors

        c = post.counts.T
        c = np.divide(c, np.max(c,0), casting='unsafe')

        # if post.isRelative:
        #     if np.all(post.relativeTo == post.relativeTo[0]):
        #         y = post.binCentres + post.relativeTo[0]
        #     else:
        #         stuff = 2
        # else:
        y = post._cellCentres #[0, :]
        c.pcolor(xtmp, y=y, **kwargs)

        cP.title('Relative error posterior distributions for system {}'.format(system))


    def plotRelativeError(self, **kwargs):
        """ Plot the relative errors of the data """

        xAxis = kwargs.pop('xAxis', 'x')
        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','-')
        kwargs['linewidth'] = kwargs.pop('linewidth',1.0)

        xtmp = self.mesh.getXAxis(xAxis)

        if (self.nSystems > 1):
            r = range(self.nSystems)
            for i in r:
                kwargs['c'] = cP.wellSeparated[i+2]
                self.relativeError[:, i].plot(xtmp,
                    alpha = 0.7, label='System {}'.format(i + 1), **kwargs)
            plt.legend()
        else:
            kwargs['c'] = cP.wellSeparated[2]
            self.relativeError.plot(xtmp,
                    alpha = 0.7, label='System {}'.format(1), **kwargs)


    def scatter2D(self, **kwargs):
        return self.data.scatter2D(**kwargs)


    def plotTotalError(self, channel, **kwargs):
        """ Plot the relative errors of the data """


        xAxis = kwargs.pop('xAxis', 'x')

        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','-')
        kwargs['linewidth'] = kwargs.pop('linewidth',1.0)

        xtmp = self.mesh.getXAxis(xAxis)

#        if (self.nSystems > 1):
#            r = range(self.nSystems)
#            for i in r:
#                fc = cP.wellSeparated[i+2]
#                cP.plot(x=self.xPlot, y=self.additiveError[:,i],
#                    marker=m,markersize=ms,markerfacecolor=mfc,markeredgecolor=mec,markeredgewidth=mew,
#                    linestyle=ls,linewidth=lw,c=fc,
#                    alpha = 0.7,label='System ' + str(i + 1), **kwargs)
#        else:
        fc = cP.wellSeparated[2]
        self.totalError[:,channel].plot(xtmp,
                alpha = 0.7, label='Channel ' + str(channel), **kwargs)

#        cP.xlabel(self.xPlot.getNameUnits())
#        cP.ylabel(self.additiveError.getNameUnits())
#        plt.legend()


    def plotTotalErrorDistributions(self, channel=0, nBins=100, **kwargs):
        """ Plot the distributions of relative errors as an image for all data points in the line """
        self.setAlonglineAxis(self.plotAgainst)

        H = Histogram1D(values=np.log10(self.totalError[:,channel]),nBins=nBins)

        H.plot(**kwargs)

#        if self.nSystems > 1:
#            c = tmp[system].counts.T
#            c = np.divide(c, np.max(c,0), casting='unsafe')
#            c.pcolor(x=self.xPlot, y=tmp[system].bins, **kwargs)
#            cP.title('Relative error posterior distributions for system '+str(system))
#        else:
#            c = tmp.counts.T
#            c = np.divide(c, np.max(c,0), casting='unsafe')
#            c.pcolor(x=self.xPlot, y=tmp.bins, **kwargs)
#            cP.title('Relative error posterior distributions')

    def histogram(self, nBins, depth1 = None, depth2 = None, reciprocateParameter = False, bestModel = False, **kwargs):
        """ Compute a histogram of the model, optionally show the histogram for given depth ranges instead """

        if (depth1 is None):
            depth1 = np.maximum(self.mesh.z.cellEdges[0], 0.0)
        if (depth2 is None):
            depth2 = self.mesh.z.cellEdges[-1]

        maxDepth = self.mesh.z.cellEdges[-1]

        # Ensure order in depth values
        if (depth1 > depth2):
            tmp = depth2
            depth2 = depth1
            depth1 = tmp

        # Don't need to check for depth being shallower than self.mesh.y.cellEdges[0] since the sortedsearch will return 0
        assert depth1 <= maxDepth, ValueError('Depth1 is greater than max depth {}'.format(maxDepth))
        assert depth2 <= maxDepth, ValueError('Depth2 is greater than max depth {}'.format(maxDepth))

        cell1 = self.mesh.z.cellIndex(depth1, clip=True)
        cell2 = self.mesh.z.cellIndex(depth2, clip=True)

        if (bestModel):
            model = self.bestParameters
            title = 'Best model values between {:.3f} m and {:.3f} m depth'.format(depth1, depth2)
        else:
            model = self.meanParameters
            title = 'Mean model values between {:.3f} m and {:.3f} m depth'.format(depth1, depth2)

        vals = model[:, cell1:cell2+1].deepcopy()

        log = kwargs.pop('log',False)

        if (reciprocateParameter):
            i = np.where(vals > 0.0)[0]
            vals[i] = 1.0 / vals[i]
            name = 'Resistivity'
            units = '$\Omega m$'
        else:
            name = 'Conductivity'
            units = '$Sm^{-1}$'

        if (log):
            vals, logLabel = cF._log(vals,log)
            name = logLabel + name
        binEdges = StatArray.StatArray(np.linspace(np.nanmin(vals), np.nanmax(vals), nBins+1), name, units)

        h = Histogram1D(bins = binEdges)
        h.update(vals)
        h.plot(**kwargs)
        cP.title(title)


    def parameterHistogram(self, nBins, depth = None, depth2 = None, log=None):
        """ Compute a histogram of all the parameter values, optionally show the histogram for given depth ranges instead """

        # Get the depth grid
        if (not depth is None):
            assert depth <= self.mesh.z.cellEdges[-1], 'Depth is greater than max depth '+str(self.mesh.z.cellEdges[-1])
            j = self.mesh.z.cellIndex(depth)
            k = j+1
            if (not depth2 is None):
                assert depth2 <= self.mesh.z.cellEdges[-1], 'Depth2 is greater than max depth '+str(self.mesh.z.cellEdges[-1])
                assert depth <= depth2, 'Depth2 must be >= depth'
                k = self.mesh.z.cellIndex(depth2)

        # First get the min max of the parameter hitmaps
        x0 = np.log10(self.minParameter)
        x1 = np.log10(self.maxParameter)

        if depth:
            counts = self.hdfFile['currentmodel/par/posterior/arr/data'][:, j:k, :]
            # return StatArray.StatArray(np.sum(counts[:, :, pj:]) / np.sum(counts) * 100.0, name="Probability of {} > {:0.2f}".format(self.meanParameters.name, value), units = self.meanParameters.units)
        else:
            counts = self.hdfFile['currentmodel/par/posterior/arr/data']

        parameters = RectilinearMesh1D().fromHdf(self.hdfFile['currentmodel/par/posterior/x'])

        bins = StatArray.StatArray(np.logspace(x0, x1, nBins), self.parameterName, units = self.parameterUnits)

        out = Histogram1D(bins=bins, log=log)


        # Bar = progressbar.ProgressBar()
        # for i in Bar(range(self.nPoints)):
        for i in range(self.nPoints):
            p = RectilinearMesh1D(cellEdges=parameters.cellEdges[i, :])

            pj = out.cellIndex(p.cellCentres, clip=True)

            cTmp = counts[i, :, :]

            out.counts[pj] += np.sum(cTmp, axis=0)

        return out


    def plotBestModel(self, **kwargs):

        values = self.bestParameters
        if (kwargs.pop('reciprocateParameter', False)):
            values = 1.0 / values
            values.name = 'Resistivity'
            values.units = '$\Omega m$'

        return self.plotXsection(values = values, **kwargs)


    def plotHighestMarginal(self, useVariance=True, **kwargs):

        values = self.highestMarginal
        return self.plotXsection(values = values, **kwargs)



    def plotMeanModel(self, **kwargs):

        values = self.meanParameters
        if (kwargs.pop('reciprocateParameter', False)):
            values = 1.0 / values
            values.name = 'Resistivity'
            values.units = '$\Omega m$'

        return self.plotXsection(values = values, **kwargs)


    def plotModeModel(self, **kwargs):

        values = self.modeParameter
        if (kwargs.pop('reciprocateParameter', False)):
            values = 1.0 / values
            values.name = 'Resistivity'
            values.units = '$\Omega m$'

        return self.plotXsection(values = values, **kwargs)


    def plotXsection(self, values, **kwargs):
        """ Plot a cross-section of the parameters """

        if kwargs.pop('useVariance', False):
            opacity = self.opacity.copy()

            if kwargs.pop('only_below_doi', False):
                indices = self.mesh.z.cellIndex(self.doi)

                for i in range(self.nPoints):
                    opacity[:indices[i], i] = 1.0

            kwargs['alpha'] = opacity

        ax, pm, cb = self.mesh.pcolor(values = values, **kwargs)
        return ax, pm, cb


    # def plotFacies(self, mean, var, volFrac, percent=67.0, ylim=None):
    #     """ Plot a cross-section of the parameters """

    #     assert False, ValueError('Double check this')

    #     self.setAlonglineAxis(self.plotAgainst)
    #     zGrd = self.zGrid

    #     a = np.zeros([zGrd.size, self.nPoints + 1],order = 'F')  # Transparency amounts
    #     a[:, 1:] = self.opacity

    #     self._getX_pmesh(zGrd.size)
    #     self._getZ_pmesh(zGrd)

    #     c = np.zeros(self._xMesh.shape, order = 'F')  # Colour value of each cell
    #     c[:-1, 0] = np.nan
    #     c[:-1, -1] = np.nan
    #     c[-1, :] = np.nan

    #     self.assignFacies(mean, var, volFrac)
    #     c[:-1, 1:-1] = self.facies

    #     # Mask the Nans in the colorMap
    #     cm = ma.masked_invalid(c)

    #     # Get the "depth of investigation"
    #     self.getDOI(percent)

    #     ax = plt.gca()
    #     cP.pretty(ax)
    #     p = ax.pcolormesh(self._xMesh, self._zMesh, cm)
    #     cb = plt.colorbar(p, ax=ax)
    #     plt.plot(self.xPlot, self.elevation, color='k')
    #     if (self.z is None):
    #         self.z = np.asarray(self.getAttribute('z'))
    #     plt.plot(self.xPlot, self.z.reshape(self.z.size) + self.elevation, color='k')

    #     plt.plot(self.xPlot, self.elevation - self.doi, color='k', linestyle='-', alpha=0.7, linewidth=1)

    #     plt.savefig('myfig.png')
    #     for i, j in zip(p.get_facecolors(), a.flatten()):
    #         i[3] = j  # Set the alpha value of the RGBA tuple using m2
    #     fIO.deleteFile('myfig.png')

    #     if (not ylim is None):
    #         plt.ylim(ylim)

    #     cP.xlabel(self.xPlot.getNameUnits())
    #     cP.ylabel('Elevation (m)')
    #     cP.clabel(cb, 'Facies')


    @cached_property
    def marginalProbability(self):

        assert 'marginal_probability' in self.hdfFile.keys(), Exception("Marginal probabilities need computing, use Inference_2D.computeMarginalProbability_X()")

        if 'marginal_probability' in self.hdfFile.keys():
            marginal_probability = StatArray.StatArray().fromHdf(self.hdfFile['marginal_probability'])

        return marginal_probability

        # else:
        #     nClasses = self._marginalProbability.shape[0]
        #     return StatArray.StatArray(self._marginalProbability[i, :, :], name = "$\frac{P_{" + str(i+1) + "}}{\sum_{i=1}^{" + str(nClasses) + "}P_{i}}$")


    def read_fit_distributions(self, fit_file, mask_by_doi=True, components='amvd'):

        # Get the fits for the given line
        # Define the depth intervals and plotting axis
        means = None
        amplitudes = None
        variances = None
        degrees = None
        with h5py.File(fit_file, 'r') as f:
            if 'm' in components:
                means = StatArray.StatArray(np.asarray(f['means/data']), 'Conductivity', '$\\frac{S}{m}$')
            if 'a' in components:
                amplitudes = StatArray.StatArray(np.asarray(f['amplitudes/data']), 'Amplitude')
            if 'v' in components:
                variances = StatArray.StatArray(np.asarray(f['variances/data']), 'Variance')
            if 'd' in components:
                degrees = StatArray.StatArray(np.asarray(f['degrees/data']), 'Degrees of freedom')

        intervals = self.mesh.z.cellCentres

        if mask_by_doi:
            indices = intervals.searchsorted(self.doi)
            if 'a' in components:
                for i in range(self.nPoints):
                    amplitudes[i, indices[i]:, :] = np.nan
            if 'm' in components:
                for i in range(self.nPoints):
                    means[i, indices[i]:, :] = np.nan
            if 'v' in components:
                for i in range(self.nPoints):
                    variances[i, indices[i]:, :] = np.nan
            if 'd' in components:
                for i in range(self.nPoints):
                    degrees[i, indices[i]:, :] = np.nan

        iWhere = np.argsort(means, axis=-1)
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


    def compute_marginal_probability_from_labelled_mixtures(self, fit_file, gmm, labels):

        amplitudes, means, variances, degrees = self.read_fit_distributions(fit_file, mask_by_doi=False)

        # self.marginalProbability = StatArray.StatArray(np.zeros([self.nPoints, self.mesh.z.nCells, gmm.n_components]), 'Marginal probability')

        iSort = np.argsort(np.squeeze(gmm.means_))

        print('Computing marginal probability', flush=True)
        for i in progressbar.progressbar(range(self.nPoints)):
            hm = self.get_hitmap(i)
            for j in range(self.mesh.z.nCells):
                m = means[i, j, :]
                inan = ~np.isnan(m)
                m = m[inan]

                if np.size(m) > 0:
                    a = amplitudes[i, j, inan]

                    v = variances[i, j, inan]
                    df = degrees[i, j, inan]
                    l = labels[i, j, inan].astype(np.int)

                    fit_mixture = mixStudentT(m, v, df, a, labels=l)
                    fit_pdfs = fit_mixture.probability(np.log10(hm.xBinCentres), log=False)

                    # gmm_pdfs = np.zeros([gmm.n_components, self.hitMap.x.nCells])

                    # for k_gmm in range(gmm.n_components):
                    #     # Term 1: Get the weight of the labelled fit from the classification
                    #     relative_fraction = gmm.weights_[iSort[k_gmm]]

                    #     for k_mix in range(fit_mixture.n_mixtures):
                    #         # Term 2: Get the probability of each mixture given the mean of the student T.
                    #         pMixture = np.squeeze(gmm.predict_proba(m[k_mix].reshape(-1, 1)))[iSort[k_gmm]] / np.float(fit_mixture.n_mixtures)

                    #         gmm_pdfs[k_gmm, :] += relative_fraction * pMixture * fit_pdfs[:, k_mix]


                    a = gmm.weights_[iSort]
                    b = gmm.predict_proba(m.reshape(-1, 1))[:, iSort] / np.float(fit_mixture.n_mixtures)
                    gmm_pdfs = np.dot(fit_pdfs, a*b).T

                    h = hm.marginalize(index = j)
                    self.marginalProbability[i, j, :] = h._marginal_probability_pdfs(gmm_pdfs)
                else:
                    self.marginalProbability[i, j, :] = np.nan

        if 'marginal_probability' in self.hdfFile.keys():
            self.marginalProbability.writeHdf(self.hdfFile, 'marginal_probability')
        else:
            self.marginalProbability.toHdf(self.hdfFile, 'marginal_probability')


    def compute_marginal_probability_from_fits(self, fit_file, mask_by_doi=True):

        amplitudes, means, variances, degrees = self.read_fit_distributions(fit_file, mask_by_doi)
        self.marginalProbability = StatArray.StatArray(np.zeros([self.nPoints, self.mesh.z.nCells, means.shape[-1]]), 'Marginal probability')

        print('Computing marginal probability', flush=True)
        for i in progressbar.progressbar(range(self.nPoints)):
            hm = self.get_hitmap(i)
            mixtures = []
            for j in range(means.shape[1]):
                a = amplitudes[i, j, :]
                m = means[i, j, :]
                v = variances[i, j, :]
                df = degrees[i, j, :]

                inan = ~np.isnan(m)
                mixtures.append(mixStudentT(m[inan], v[inan], df[inan], a[inan]))

            mp = hm.marginalProbability(1.0, distributions=mixtures, log=10, maxDistributions=means.shape[-1])
            self.marginalProbability[i, :mp.shape[0], :] = mp

        if 'marginal_probability' in self.hdfFile.keys():
            self.marginalProbability.writeHdf(self.hdfFile, 'marginal_probability')
        else:
            self.marginalProbability.toHdf(self.hdfFile, 'marginal_probability')
        # self.marginalProbability.toHdf('line_{}_marginal_probability.h5'.format(self.line), 'marginal_probability')


    def reorder_marginal_probability_by_labels(self, labels=None):


        # if 'reordered_marginal' in self.hdfFile.keys():
        #     self.marginalProbability = StatArray.StatArray().fromHdf(self.hdfFile['reordered_marginal'])
        #     return
        # assert np.all([labels.shape[i] == self.marginalProbability.shape[i] for i in range(self.marginalProbability.shape[-1])])

        self.__dict__.pop('marginalProbability', None)
        out = np.zeros_like(self.marginalProbability)

        if labels is None:
            labels = self.labels

        for i in range(self.nPoints):
            for j in range(self.depth.size):
                l = labels[i, j, :]
                for k in range(np.sum(~np.isnan(l))):
                    out[i, j, np.int(l[k])] = self.marginalProbability[i, j, k]

        self.marginalProbability = out

        if 'reordered_marginal' in self.hdfFile.keys():
            self.marginalProbability.writeHdf(self.hdfFile, 'reordered_marginal')
        else:
            self.marginalProbability.toHdf(self.hdfFile, 'reordered_marginal')





    # def computeMarginalProbability(self, fractions, distributions, **kwargs):

    #     print("Computing marginal probabilities. This can take a while. \n",
    #     "Once you have done this, and you no longer need to change the input parameters, simply use LineResults.marginalProbability to access.")

    #     assert isinstance(distributions, list), TypeError("distributions must be a list")
    #     assert np.size(fractions) == np.size(distributions), ValueError("fractions and distributions must have the same length")

    #     if distributions[0].multivariate:
    #         self._computeMarginalProbability_2D(fractions, distributions, **kwargs)
    #     else:
    #         self._computeMarginalProbability_1D(fractions, distributions, **kwargs)


    # def _computeMarginalProbability_1D(self, fractions, distributions, **kwargs):

    #     nFacies = np.size(distributions)

    #     self._marginalProbability = StatArray.StatArray(np.empty([nFacies, self.hitMap.y.nCells, self.nPoints]), name='Marginal probability')

    #     counts = self.hdfFile['currentmodel/par/posterior/arr/data']
    #     parameters = RectilinearMesh1D().fromHdf(self.hdfFile['currentmodel/par/posterior/x'])

    #     hm = self.getAttribute('hit map', index=0)

    #     Bar = progressbar.ProgressBar()
    #     print('Computing 1D Marginal', flush=True)
    #     for i in Bar(range(self.nPoints)):
    #         hm._counts = counts[i, :, :]
    #         hm._x = RectilinearMesh1D(cellEdges=parameters.cellEdges[i, :])
    #         self._marginalProbability[:, :, i] = hm.marginalProbability(fractions, distributions, axis=0, **kwargs)

    #     if 'marginal_probability' in self.hdfFile.keys():
    #         del self.hdfFile['marginal_probability']
    #     if 'facies_probability' in self.hdfFile.keys():
    #         del self.hdfFile['facies_probability']

    #     self._marginalProbability.toHdf(self.hdfFile, 'marginal_probability')

    #     return self.marginalProbability


    # def _computeMarginalProbability_2D(self, fractions, distributions, **kwargs):

    #     nFacies = np.size(distributions)

    #     self._marginalProbability = StatArray.StatArray(np.empty([nFacies, self.hitMap.y.nCells, self.nPoints]), name='Marginal probability')

    #     counts = self.hdfFile['currentmodel/par/posterior/arr/data']
    #     parameters = RectilinearMesh1D().fromHdf(self.hdfFile['currentmodel/par/posterior/x'])

    #     hm = self.getAttribute('hit map', index=0)

    #     print("Computing 2D Marginal", flush=True)
    #     Bar = progressbar.ProgressBar()
    #     for i in Bar(range(self.nPoints)):
    #         hm._counts = counts[i, :, :]
    #         hm._x = RectilinearMesh1D(cellEdges=parameters.cellEdges[i, :])
    #         self._marginalProbability[:, :, i] = hm.marginalProbability(fractions, distributions, axis=2, **kwargs)

    #     if 'marginal_probability' in self.hdfFile.keys():
    #         del self.hdfFile['marginal_probability']
    #     if 'facies_probability' in self.hdfFile.keys():
    #         del self.hdfFile['facies_probability']

    #     self._marginalProbability.toHdf(self.hdfFile, 'marginal_probability')

    #     return self.marginalProbability


    @cached_property
    def highestMarginal(self):

        mp = self.marginalProbability
        out = np.argmax(mp, axis=-1)

        # mp2 = np.empty(out.shape)
        # for i in range(mp.shape[0]):
        #     for j in range(mp.shape[1]):
        #         mp2[i, j] = mp[i, j, out[i, j]]
        # mask = np.where(np.isnan(mp2))

        # out = out.astype(np.float64)
        # out[mask] = np.nan

        return out


    @property
    def probability_of_highest_marginal(self):

        out = StatArray.StatArray(self.mesh.shape, "Probability")

        hm = self.highestMarginal
        classes = np.unique(hm)

        mp = self.marginalProbability()

        for i, c in enumerate(classes):
            iWhere = np.where(hm == c)
            out[iWhere[0], iWhere[1]] = mp[i, iWhere[0], iWhere[1]]

        return out


    def plot_inference_1d(self, fiducial):
        """ Plot the geobipy results for the given data point """
        R = self.inference_1d(fiducial=fiducial)
        R.initFigure(forcePlot=True)
        R.plot(forcePlot=True)


    def plotSummary(self, data, fiducial, **kwargs):

        R = self.inference_1d(fiducial=fiducial)

        cWidth = 3

        nCols = 15 + (3 * R.nSystems) + 1

        gs = gridspec.GridSpec(18, nCols)
        gs.update(wspace=20.0, hspace=20.0)
        ax = [None]*(7+(2*R.nSystems))

        ax[0] = plt.subplot(gs[:3, :nCols - 10 - (R.nSystems * 3)]) # Data misfit vs iteration
        R._plotMisfitVsIteration(markersize=1, marker='.',)


        ax[1] = plt.subplot(gs[3:6, :nCols - 10 - (R.nSystems * 3)]) # Histogram of # of layers
        R._plotNumberOfLayersPosterior()


        ax[2] = plt.subplot(gs[6:12, :nCols - 13]) # Site Map
        data.scatter2D(c='k', s=1)
        line = data.getLine(line=self.line)
        line.scatter2D(c='cyan')
        cP.plot(R.bestDataPoint.x, R.bestDataPoint.y, color='r', marker='o')


        ax[3] = plt.subplot(gs[6:12, nCols - 13:nCols - 10]) # Data Point
        R._plotObservedPredictedData()
        plt.title('')


        ax[4] = plt.subplot(gs[:12, nCols - 10: nCols - 4]) # Hitmap
        R._plotHitmapPosterior()


        ax[5] = plt.subplot(gs[:12, nCols - 4: nCols-1]) # Interface histogram
        R._plotLayerDepthPosterior()


        ax[6] = plt.subplot(gs[12:, :]) # cross section
        self.plotXsection(**kwargs)


        for i in range(R.nSystems):

            j0 = nCols - 10 - (R.nSystems* 3) + (i * 3)
            j1 = j0 + 3

            ax[7+(2*i)] = plt.subplot(gs[:3, j0:j1])
            R._plotRelativeErrorPosterior(system=i)
            cP.title('System ' + str(i + 1))

            # Update the histogram of additive data errors
            ax[7+(2*i)-1] = plt.subplot(gs[3:6, j0:j1])
            # ax= plt.subplot(self.gs[3:6, 2 * self.nSystemstems + j])
            R._plotAdditiveErrorPosterior(system=i)



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

        d = StatArray.StatArray(1.0 / a, "Best Conductivity", "$\fraq{S}{m}$")
        e = StatArray.StatArray(1.0 / b, "Mean Conductivity", "$\fraq{S}{m}$")

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

        return hdfRead.readKeyFromFile(self.hdfFile, self.fName, iDs, keys, index=index, **kwargs)


    def _attrTokey(self, attributes):
        """ Takes an easy to remember user attribute and converts to the tag in the HDF file """
        if (isinstance(attributes, str)):
            attributes = [attributes]
        res = []
        nSys= None
        for attr in attributes:
            low = attr.lower()
            if (low == 'iteration #'):
                res.append('i')
            elif (low == '# of markov chains'):
                res.append('nmc')
            elif (low == 'burned in'):
                res.append('burnedin')
            elif (low == 'burn in #'):
                res.append('iburn')
            elif (low == 'data multiplier'):
                res.append('multiplier')
            elif (low == 'height posterior'):
                res.append('currentdatapoint/z/posterior')
            elif (low == 'fiducials'):
                res.append('fiducials')
            elif (low == 'labels'):
                res.append('labels')
            elif (low == 'layer posterior'):
                res.append('currentmodel/nCells/posterior')
            elif (low == 'layer depth posterior'):
                res.append('currentmodel/depth/posterior')
            elif (low == 'best data'):
                res.append('bestd')
            elif (low == 'x'):
                res.append('bestd/x')
            elif (low == 'y'):
                res.append('bestd/y')
            elif (low == 'z'):
                res.append('bestd/z')
            elif (low == 'elevation'):
                res.append('bestd/e')
            elif (low == 'observed data'):
                res.append('bestd/d')
            elif (low == 'predicted data'):
                res.append('bestd/p')
            elif (low == 'total error'):
                res.append('bestd/s')
            elif (low == '# of systems'):
                res.append('nsystems')
            elif (low == 'additive error'):
                res.append('bestd/addErr')
            elif (low == 'relative error'):
                res.append('bestd/relErr')
            elif (low == 'best model'):
                res.append('bestmodel')
            elif (low == 'meaninterp'):
                res.append('meaninterp')
            elif (low == 'bestinterp'):
                res.append('bestinterp')
            elif (low == 'opacityinterp'):
                res.append('opacityinterp')
            elif (low == '# layers'):
                res.append('bestmodel/nCells')
            elif (low == 'current data'):
                res.append('currentdatapoint')
            elif (low == 'hit map'):
                res.append('currentmodel/par/posterior')
            elif (low == 'hitmap/y'):
                res.append('currentmodel/par/posterior/y')
            elif (low == 'doi'):
                res.append('doi')
            elif (low == 'data misfit'):
                res.append('phids')
            elif (low == 'relative error posterior'):
                if (nSys is None): nSys = hdfRead.readKeyFromFile(self.hdfFile, self.fName, '/','nsystems')
                for i in range(nSys):
                    res.append('currentdatapoint/relErr/posterior' +str(i))
            elif (low == 'additive error posterior'):
                if (nSys is None): nSys = hdfRead.readKeyFromFile(self.hdfFile, self.fName, '/','nsystems')
                for i in range(nSys):
                    res.append('currentdatapoint/addErr/posterior' +str(i))
            elif (low == 'inversion time'):
                res.append('invtime')
            elif (low == 'saving time'):
                res.append('savetime')
            else:
                assert False, self.possibleAttributes(attr)
        return res


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


    def createHdf(self, hdfFile, fiducials, results):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """

        self.hdfFile = hdfFile

        nPoints = fiducials.size
        self.fiducials = StatArray.StatArray(np.sort(fiducials), "fiducials")
        assert not np.any(np.isnan(self.fiducials)), ValueError("Cannot have fiducials == NaN")

        # Initialize and write the attributes that won't change
        # hdfFile.create_dataset('ids',data=self.fiducials)
        self.fiducials.createHdf(hdfFile, 'fiducials')
        self.fiducials.writeHdf(hdfFile, 'fiducials')
        hdfFile.create_dataset('iplot', data=results.iPlot)
        hdfFile.create_dataset('plotme', data=results.plotMe)
        hdfFile.create_dataset('reciprocateParameter', data=results.reciprocateParameter)

        if not results.limits is None:
            hdfFile.create_dataset('limits', data=results.limits)
        hdfFile.create_dataset('nmc', data=results.nMC)
        hdfFile.create_dataset('nsystems', data=results.nSystems)
        results.ratex.toHdf(hdfFile,'ratex')
#        hdfFile.create_dataset('ratex', [results.ratex.size], dtype=results.ratex.dtype)
#        hdfFile['ratex'][:] = results.ratex


        # Initialize the attributes that will be written later
        hdfFile.create_dataset('i', shape=[nPoints], dtype=results.i.dtype, fillvalue=np.nan)
        hdfFile.create_dataset('iburn', shape=[nPoints], dtype=results.iBurn.dtype, fillvalue=np.nan)
        hdfFile.create_dataset('ibest', shape=[nPoints], dtype=results.iBest.dtype, fillvalue=np.nan)
        hdfFile.create_dataset('burnedin', shape=[nPoints], dtype=type(results.burnedIn))
        # hdfFile.create_dataset('doi',  shape=[nPoints], dtype=float, fillvalue=np.nan)
        hdfFile.create_dataset('multiplier',  shape=[nPoints], dtype=results.multiplier.dtype, fillvalue=np.nan)
        hdfFile.create_dataset('invtime',  shape=[nPoints], dtype=float, fillvalue=np.nan)
        hdfFile.create_dataset('savetime',  shape=[nPoints], dtype=float, fillvalue=np.nan)

        results.meanInterp.createHdf(hdfFile,'meaninterp', nRepeats=nPoints, fillvalue=np.nan)
        results.bestInterp.createHdf(hdfFile,'bestinterp', nRepeats=nPoints, fillvalue=np.nan)
        results.opacityInterp.createHdf(hdfFile,'opacityinterp',nRepeats=nPoints, fillvalue=np.nan)
#        hdfFile.create_dataset('opacityinterp', [nPoints,nz], dtype=np.float64)

        results.rate.createHdf(hdfFile,'rate',nRepeats=nPoints, fillvalue=np.nan)
#        hdfFile.create_dataset('rate', [nPoints,results.rate.size], dtype=results.rate.dtype)
        results.PhiDs.createHdf(hdfFile,'phids',nRepeats=nPoints, fillvalue=np.nan)
        #hdfFile.create_dataset('phids', [nPoints,results.PhiDs.size], dtype=results.PhiDs.dtype)

        results.currentDataPoint.createHdf(hdfFile,'currentdatapoint', nRepeats=nPoints, fillvalue=np.nan)
        results.bestDataPoint.createHdf(hdfFile,'bestd', withPosterior=False, nRepeats=nPoints, fillvalue=np.nan)

        # Since the 1D models change size adaptively during the inversion, we need to pad the HDF creation to the maximum allowable number of layers.

        tmp = results.currentModel.pad(results.currentModel.maxLayers)

        tmp.createHdf(hdfFile, 'currentmodel', nRepeats=nPoints, fillvalue=np.nan)

        tmp = results.bestModel.pad(results.bestModel.maxLayers)
        tmp.createHdf(hdfFile, 'bestmodel', withPosterior=False, nRepeats=nPoints, fillvalue=np.nan)

        if results.verbose:
            results.posteriorComponents.createHdf(hdfFile,'posteriorcomponents',nRepeats=nPoints, fillvalue=np.nan)

        # Add the best data components
#        hdfFile.create_dataset('bestdata.z', [nPoints], dtype=results.bestD.z.dtype)
#        hdfFile.create_dataset('bestdata.p', [nPoints,*results.bestD.p.shape], dtype=results.bestD.p.dtype)
#        hdfFile.create_dataset('bestdata.s', [nPoints,*results.bestD.s.shape], dtype=results.bestD.s.dtype)

        # Add the best model components
#        hdfFile.create_dataset('bestmodel.ncells', [nPoints], dtype=results.bestModel.nCells.dtype)
#        hdfFile.create_dataset('bestmodel.top', [nPoints], dtype=results.bestModel.top.dtype)
#        hdfFile.create_dataset('bestmodel.par', [nPoints,*results.bestModel.par.shape], dtype=results.bestModel.par.dtype)
#        hdfFile.create_dataset('bestmodel.depth', [nPoints,*results.bestModel.depth.shape], dtype=results.bestModel.depth.dtype)
#        hdfFile.create_dataset('bestmodel.thk', [nPoints,*results.bestModel.thk.shape], dtype=results.bestModel.thk.dtype)
#        hdfFile.create_dataset('bestmodel.chie', [nPoints,*results.bestModel.chie.shape], dtype=results.bestModel.chie.dtype)
#        hdfFile.create_dataset('bestmodel.chim', [nPoints,*results.bestModel.chim.shape], dtype=results.bestModel.chim.dtype)

#        self.currentD.createHdf(grp, 'currentd')
#        self.bestD.createHdf(grp, 'bestd')
#
#        tmp=self.bestModel.pad(self.bestModel.maxLayers)
#        tmp.createHdf(grp, 'bestmodel')

    def results2Hdf(self, results):
        """ Given a HDF file initialized as line results, write the contents of results to the appropriate arrays """

        assert results.fiducial in self.fiducials, Exception("The HDF file does not have ID number {}. Available ids are between {} and {}".format(results.fiducial, np.min(self.fiducials), np.max(self.fiducials)))

        hdfFile = self.hdfFile

        # Get the point index
        i = self.fiducials.searchsorted(results.fiducial)

        # Add the iteration number
        hdfFile['i'][i] = results.i

        # Add the burn in iteration
        hdfFile['iburn'][i] = results.iBurn

        # Add the burn in iteration
        hdfFile['ibest'][i] = results.iBest

        # Add the burned in logical
        hdfFile['burnedin'][i] = results.burnedIn

        # Add the depth of investigation
        # hdfFile['doi'][i] = results.doi()

        # Add the multiplier
        hdfFile['multiplier'][i] = results.multiplier

        # Add the inversion time
        hdfFile['invtime'][i] = results.invTime

        # Add the savetime
#        hdfFile['savetime'][i] = results.saveTime

        # Interpolate the mean and best model to the discretized hitmap
        hm = results.currentModel.par.posterior
        results.meanInterp[:] = hm.mean()
        results.bestInterp[:] = results.bestModel.interpPar2Mesh(results.bestModel.par, hm)
        # results.opacityInterp[:] = results.Hitmap.credibleRange(percent=95.0, log='e')

        slic = np.s_[i, :]
        # Add the interpolated mean model
        results.meanInterp.writeHdf(hdfFile, 'meaninterp',  index=slic)
        # Add the interpolated best
        results.bestInterp.writeHdf(hdfFile, 'bestinterp',  index=slic)
        # Add the interpolated opacity

        # Add the acceptance rate
        results.rate.writeHdf(hdfFile, 'rate', index=slic)

        # Add the data misfit
        results.PhiDs.writeHdf(hdfFile,'phids',index=slic)

        results.currentDataPoint.writeHdf(hdfFile,'currentdatapoint',  index=i)

        results.bestDataPoint.writeHdf(hdfFile,'bestd', withPosterior=False, index=i)

        results.currentModel.writeHdf(hdfFile,'currentmodel', index=i)

        results.bestModel.writeHdf(hdfFile,'bestmodel', withPosterior=False, index=i)

