""" @DataSetResults
Class to handle the HDF5 result files for a full data set.
 """
import time
from datetime import timedelta
from ..base import Error as Err
import matplotlib.pyplot as plt
import numpy as np
import h5py
#import numpy.ma as ma
from cached_property import cached_property
from ..classes.core.myObject import myObject
from ..classes.core import StatArray
from ..base import fileIO

from ..classes.statistics.Histogram1D import Histogram1D
from ..classes.statistics.Hitmap2D import Hitmap2D
from ..classes.pointcloud.PointCloud3D import PointCloud3D
from ..base import interpolation as interpolation
from .Inv_MCMC import Initialize
from .LineResults import LineResults
from .Results import Results

from ..classes.data.dataset.Data import Data
from ..classes.data.datapoint.DataPoint import DataPoint

#from ..classes.statistics.Distribution import Distribution
from ..base.HDF import hdfRead
from ..base import customPlots as cP
from ..base import customFunctions as cF
from os.path import join
from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
#from ..base import fileIO as fIO
#from os.path import split
#from pyvtk import VtkData, UnstructuredGrid, CellData, Scalars

from os import listdir
import progressbar

class DataSetResults(myObject):
    """ Class to define results from Inv_MCMC for a full data set """

    def __init__(self, directory, systemFilepath, files = None):
        """ Initialize the lineResults
        directory = directory containing folders for each line of data results
        """
        self.directory = directory
        self._h5files = None
        self._nPoints = None
        self.cumNpoints = None
        self.bounds = None

        if files is None:
            self._h5files = self._get_h5Files_from_directory(directory)
        else:
            self._h5files = self._get_h5Files_from_list(directory, files)

        self._lines = []
        self._lineNumbers = np.empty(self.nLines)
        for i in range(self.nLines):
            fName = self.h5files[i]
            LR = LineResults(fName, systemFilepath=systemFilepath)
            self._lines.append(LR)
            self._lineNumbers[i] = LR.line

        self.kdtree = None
        self.doi = None
        self.doi2D = None
        self.mean3D = None
        self.best3D = None
        self._facies = None


    def open(self):
        """ Check whether the file is open """
        for line in self.lines:
            line.open()


    def close(self):
        """ Check whether the file is open """
        for line in self.lines:
            line.close()


    def createHDF5(self, data, userParameters):
        """Create HDF5 files based on the data

        Parameters
        ----------
        data : geobipy.Data or geobipy.DataPoint
            Data to create the HDF5 file(s) for
        userParameters : geobipy.userParameters
            Input parameters for geobipy

        Returns
        -------
        out : list of H5py.File
            HDF5 files

        """

        if isinstance(data, Data):
            return self._createHDF5_dataset(data, userParameters)
        else:
            return self._createHDF5_datapoint(data, userParameters)


    def _createHDF5_dataset(self, dataset, userParameters):

        t0 = time.time()

        dataset.readSystemFile(userParameters.systemFilename)

        # Prepare the dataset so that we can read a point at a time.
        dataset._initLineByLineRead(userParameters.dataFilename, userParameters.systemFilename)
        # Get a datapoint from the file.
        DataPoint = dataset._readSingleDatapoint()
        dataset._closeDatafiles()

        # Initialize the user parameters
        options = userParameters.userParameters(DataPoint)

        # While preparing the file, we need access to the line numbers and fiducials in the data file
        tmp = fileIO.read_columns(options.dataFilename[0], dataset._indicesForFile[0][:2], 1, dataset.nPoints)

        dataset._openDatafiles(options.dataFilename)

        # Get the line numbers in the data
        self._lineNumbers = np.sort(np.unique(tmp[:, 0]))
        fiducials = tmp[:, 1]

        # Initialize the inversion to obtain the sizes of everything
        options, Mod, DataPoint, _, _, _, _ = Initialize(options, DataPoint)

        # Create the results template
        Res = Results(DataPoint, Mod,
                      save=options.save, plot=options.plot, savePNG=options.savePNG,
                      nMarkovChains=options.nMarkovChains, plotEvery=options.plotEvery,
                      reciprocateParameters=options.reciprocateParameters, verbose=options.verbose)

        print('Creating HDF5 files, this may take a few minutes...')
        print('Files are being created for data files {} and system files {}'.format(options.dataFilename, options.systemFilename))

        # No need to create and close the files like in parallel, so create and keep them open
        self._lines = []
        for line in self.lineNumbers:
            fiducialsForLine = np.where(tmp[:, 0] == line)[0]
            H5File = h5py.File(join(self.directory, '{}.h5'.format(line)), 'w')
            lr = LineResults()
            lr.createHdf(H5File, fiducials[fiducialsForLine], Res)
            self._lines.append(lr)
            print('Time to create line {} with {} data points: {} h:m:s'.format(line, fiducialsForLine.size, str(timedelta(seconds=time.time()-t0))))

    def _createHDF5_datapoint(self, datapoint, userParameters):

        print('stuff')

    @property
    def h5files(self):
        """ Get the list of line result files for the dataset """
        return self._h5files

    @property
    def lines(self):
        return self._lines

    @property
    def lineNumbers(self):
        return self._lineNumbers


    @property
    def nLines(self):
        return np.size(self._h5files)


    def _get_h5Files_from_list(self, directory, files):
        if not isinstance(files, list):
            files = [files]
        h5files = []
        for f in files:
            fName = join(directory, f)
            assert fileIO.fileExists(fName), Exception("File {} does not exist".format(fName))
            h5files.append(fName)
        return h5files


    def _get_h5Files_from_directory(self, directory):
        h5files = []
        for file in [f for f in listdir(directory) if f.endswith('.h5')]:
            fName = join(directory, file)
            fileIO.fileExists(fName)
            h5files.append(fName)

        h5files = sorted(h5files)

        # assert len(h5files) > 0, 'Could not find .h5 files in directory {}'.format(directory)

        return h5files


    @cached_property
    def additiveError(self):

        additiveError = StatArray.StatArray([self.nSystems, self.nPoints], name=self.lines[0].additiveError.name, units=self.lines[0].additiveError.units, order = 'F')

        print("Reading Additive Errors Posteriors", flush=True)
        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            additiveError[:, self.lineIndices[i]] = self.lines[i].additiveError.T
            del self.lines[i].__dict__['additiveError'] # Free memory
        return additiveError


    @cached_property
    def bestData(self):

        bestData = self.lines[0].bestData
        del self.lines[0].__dict__['bestData']

        print("Reading Most Probable Data", flush=True)
        Bar = progressbar.ProgressBar()

        for i in Bar(range(1, self.nLines)):
            bestData = bestData + self.lines[i].bestData
            del self.lines[i].__dict__['bestData']

        return bestData


    @cached_property
    def bestParameters(self):


        bestParameters = StatArray.StatArray([self.zGrid.nCells, self.nPoints], name=self.lines[0].bestParameters.name, units=self.lines[0].bestParameters.units, order = 'F')

        print('Reading best parameters', flush=True)
        Bar=progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            bestParameters[:, self.lineIndices[i]] = self.lines[i].bestParameters.T
            del self.lines[i].__dict__['bestParameters'] # Free memory

        return bestParameters


    def computeMarginalProbability(self, fractions, distributions, **kwargs):
        for line in self.lines:
            line.computeMarginalProbability(fractions, distributions, **kwargs)


    @cached_property
    def dataMisfit(self):
        return self.bestData.dataMisfit()


    @property
    def hitmapCounts(self):
        if (self._counts is None):
            mesh = self.lines[0].mesh
            self._counts = np.empty([])


    @property
    def elevation(self):
        return self.pointcloud.elevation


    @property
    def facies(self):
        assert not self._facies is None, Exception("Facies must be set using self.setFaciesProbabilities()")
        return self._facies


    @property
    def height(self):
        return self.pointcloud.z


    def identifyPeaks(self, depths, nBins = 250, width=4, limits=None):
        """Identifies peaks in the parameter posterior for each depth in depths.

        Parameters
        ----------
        depths : array_like
            Depth intervals to identify peaks between.

        Returns
        -------

        """

        print(limits)

        out = self.lines[0].identifyPeaks(depths, nBins, width, limits)
        for line in self.lines[1:]:
            out = np.vstack([out, line.identifyPeaks(depths, nBins, width, limits)])

        return out


    @cached_property
    def interfaces(self):
        interfaces = StatArray.StatArray([self.zGrid.nCells, self.nPoints], name=self.lines[0].interfaces.name, units=self.lines[0].interfaces.units)

        print("Reading Depth Posteriors", flush=True)
        Bar=progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            interfaces[:, self.lineIndices[i]] = self.lines[i].interfaces.T
            del self.lines[i].__dict__['interfaces'] # Free memory

        return interfaces


    @cached_property
    def lineIndices(self):

        lineIndices = []
        i0 = 0
        for i in range(self.nLines):
            i1 = i0 + self.lines[i].nPoints
            lineIndices.append(np.s_[i0:i1])
            i0 = i1

        return lineIndices


    @cached_property
    def meanParameters(self):

        meanParameters = StatArray.StatArray([self.zGrid.nCells, self.nPoints], name=self.lines[0].meanParameters.name, units=self.lines[0].meanParameters.units, order = 'F')

        print("Reading Mean Parameters", flush=True)
        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            meanParameters[:, self.lineIndices[i]] = self.lines[i].meanParameters
            del self.lines[i].__dict__['meanParameters'] # Free memory

        return meanParameters


    @cached_property
    def nActive(self):
        nActive = np.empty(self.nPoint, dtype=np.int)
        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            nActive[self.lineIndices[i]] = self.lines[i].bestData.nActiveChannels
            del self.lines[i].__dict__['bestData'] # Free memory

        return nActive


    @cached_property
    def nPoints(self):
        """ Get the total number of data points """
        tmp = np.asarray([this.nPoints for this in self.lines])
        self._cumNpoints = np.cumsum(tmp)
        return np.sum(tmp)


    @property
    def nSystems(self):
        """ Get the number of systems """
        return self.lines[0].nSystems


    def parameterHistogram(self, nBins, depth = None, depth2 = None, log=None):
        """ Compute a histogram of all the parameter values, optionally show the histogram for given depth ranges instead """

        out = self.lines[0].parameterHistogram(nBins=nBins, depth=depth, depth2=depth2, log=log)

        for line in self.lines[1:]:
            tmp = line.parameterHistogram(nBins=nBins, depth=depth, depth2=depth2, log=log)
            out._counts += tmp.counts

        return out


    @cached_property
    def pointcloud(self):

        x = StatArray.StatArray(self.nPoints, name=self.lines[0].x.name, units=self.lines[0].x.units)
        y = StatArray.StatArray(self.nPoints, name=self.lines[0].y.name, units=self.lines[0].y.units)
        z = StatArray.StatArray(self.nPoints, name=self.lines[0].height.name, units=self.lines[0].height.units)
        e = StatArray.StatArray(self.nPoints, name=self.lines[0].elevation.name, units=self.lines[0].elevation.units)
        # Loop over the lines in the data set and get the attributes
        print('Reading co-ordinates', flush=True)
        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            indices = self.lineIndices[i]
            x[indices] = self.lines[i].x
            y[indices] = self.lines[i].y
            z[indices] = self.lines[i].height
            e[indices] = self.lines[i].elevation

            del self.lines[i].__dict__['x'] # Free memory
            del self.lines[i].__dict__['y'] # Free memory
            del self.lines[i].__dict__['height'] # Free memory
            del self.lines[i].__dict__['elevation'] # Free memory

        return PointCloud3D(self.nPoints, x, y, z, e)


    @cached_property
    def relativeError(self):

        relativeError = StatArray.StatArray([self.nSystems, self.nPoints], name=self.lines[0].relativeError.name, units=self.lines[0].relativeError.units, order = 'F')

        print('Reading Relative Error Posteriors', flush=True)
        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            relativeError[:, self.lineIndices[i]] = self.lines[i].relativeError.T
            del self.lines[i].__dict__['relativeError'] # Free memory

        return relativeError


    @property
    def x(self):
        return self.pointcloud.x


    @property
    def y(self):
        return self.pointcloud.y


    def dataPointResults(self, fiducial=None, index=None):
        """Get the inversion results for the given fiducial.

        Parameters
        ----------
        fiducial : float
            Unique fiducial of the data point.

        Returns
        -------
        out : geobipy.Results
            The inversion results for the data point.

        """
        tmp = np.sum([x is None for x in [fiducial, index]])
        assert tmp == 1, Exception("Please specify one argument, fiducial, or index")

        if not index is None:
            fiducial = self.fiducial(index)

        index = self.fiducialIndex(fiducial)
        lineIndex = index[0][0]
        fidIndex = index[1][0]

        return self.lines[lineIndex].getResults(fidIndex)


    def lineIndex(self, lineNumber=None, fiducial=None, index=None):
        """Get the line index """
        tmp = np.sum([not x is None for x in [lineNumber, fiducial, index]])
        assert tmp == 1, Exception("Please specify one argument, lineNumber, fiducial, or index")


        if not lineNumber is None:
            return np.squeeze(np.where(self.lineNumbers == lineNumber)[0])

        if not fiducial is None:
            return self.fiducialIndex(fiducial)[0]

        if index > self.nPoints-1: raise IndexError('index {} is out of bounds for data point index with size {}'.format(index, self.nPoints))
        return self._cumNpoints.searchsorted(index)


    def fiducial(self, index):
        """ Get the fiducial of the given data point """
        iLine = self.lineIndex(index=index)
        if (iLine > 0):
            index -= self.cumNpoints[iLine-1]
        return self.lines[iLine].fiducials[index]


    def fiducialIndex(self, fiducial):
        """Get the line number and index for the specified fiducial.

        Parameters
        ----------
        fiducial : float
            The unique fiducial for the data point

        Returns
        -------
        lineIndex : ints
            lineIndex for each fiducial
        index : ints
            Index of each fiducial in their respective line

        """

        lineIndex = []
        index = []

        for i, line in enumerate(self.lines):
            ids = line.fiducialIndex(fiducial)
            nIds = np.size(ids)
            if nIds > 0:
                lineIndex.append(np.full(nIds, fill_value=i))
                index.append(ids)

        if np.size(index) > 0:
            return np.hstack(lineIndex), np.hstack(index)


    def fit_mixture(self, intervals, **kwargs):
        """Uses Mixture modelling to fit disrtibutions to the hitmaps for the specified intervals.

        The non-mpi version fits a aggregated hitmap for the entire line.
        This can lose detail in the fits, but the time savings in serial are enormous.

        If more precision is required, fit_mixture_mpi should be used instead.

        Parameters
        ----------
        intervals : array_like
            Depth intervals between which the marginal histogram is computed before fitting.

        See Also
        --------
        geobipy.Histogram1D.fit_mixture
            For details on the fitting arguments.

        """
        distributions = []
        active = []
        for line in self.lines:
            d, a = line.lineHitmap.fit_mixture(intervals, **kwargs)
            distributions.append(d)
            active.append(a)

        line = np.empty(0)
        depths = np.empty(0)
        means = np.empty(0)
        variances = np.empty(0)
        for i in range(0, self.nLines, 1):
            dl = distributions[i]
            al = active[i]
            for j in range(len(dl)):
                d = 0.5 * (intervals[j] + intervals[j+1])
                means = np.squeeze(dl[j].means_[al[j]])
                line = np.hstack([line, np.full(means.size, fill_value=self.lineNumbers[i])])
                depths = np.hstack([depths, np.full(means.size, fill_value=d)])
                means = np.hstack([means, means])
                variances = np.hstack([variances, np.squeeze(dl[j].covariances_[al[j]])])

        line = StatArray.StatArray(line, 'Line Number')
        depths = StatArray.StatArray(depths, self.lines[0].mesh.z.name, self.lines[0].mesh.z.units)
        means = StatArray.StatArray(means, "Mean "+ self.lines[0].parameterName, self.lines[0].parameterUnits)
        variances = StatArray.StatArray(variances, "Variance", "("+self.lines[0].parameterUnits+")$^{2}$")

        return line, depths, means, variances


    def fit_mixture_mpi(self, intervals, world, **kwargs):
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

        print('here')

        print(world.size)

        # distributions = []
        # active = []
        # for line in self.lines:
        #     d, a = line.lineHitmap.fit_mixture(intervals, **kwargs)
        #     distributions.append(d)
        #     active.append(a)

        # line = np.empty(0)
        # depths = np.empty(0)
        # means = np.empty(0)
        # variances = np.empty(0)
        # for i in range(0, self.nLines, 1):
        #     dl = distributions[i]
        #     al = active[i]
        #     for j in range(len(dl)):
        #         d = 0.5 * (intervals[j] + intervals[j+1])
        #         means = np.squeeze(dl[j].means_[al[j]])
        #         line = np.hstack([line, np.full(means.size, fill_value=self.lineNumbers[i])])
        #         depths = np.hstack([depths, np.full(means.size, fill_value=d)])
        #         means = np.hstack([means, means])
        #         variances = np.hstack([variances, np.squeeze(dl[j].covariances_[al[j]])])

        # line = StatArray.StatArray(line, 'Line Number')
        # depths = StatArray.StatArray(depths, self.lines[0].mesh.z.name, self.lines[0].mesh.z.units)
        # means = StatArray.StatArray(means, "Mean "+ self.lines[0].parameterName, self.lines[0].parameterUnits)
        # variances = StatArray.StatArray(variances, "Variance", "("+self.lines[0].parameterUnits+")$^{2}$")

        # return line, depths, means, variances



    def fitMajorPeaks(self, intervals, **kwargs):

        distributions = []
        amplitudes = []
        for line in self.lines:
            d, a = line.lineHitmap.fitMajorPeaks(intervals, **kwargs)
            distributions.append(d)
            amplitudes.append(a)

        return distributions, amplitudes


    def histogram(self, nBins, depth1 = None, depth2 = None, reciprocateParameter = False, bestModel = False, withDoi=False, percent=67.0, force = False, **kwargs):
        """ Compute a histogram of the model, optionally show the histogram for given depth ranges instead """

        if (depth1 is None):
            depth1 = self.zGrid.cellCentres[0]
        if (depth2 is None):
            depth2 = self.zGrid.cellCentres[-1]

        # Ensure order in depth values
        if (depth1 > depth2):
            tmp=depth2
            depth2 = depth1
            depth1 = tmp

        # Don't need to check for depth being shallower than zGrid[0] since the sortedsearch with return 0
        if (depth1 > self.zGrid.cellEdges[-1]): Err.Emsg('mapDepthSlice: Depth is greater than max depth - '+str(self.zGrid.cellEdges[-1]))
        if (depth2 > self.zGrid.cellEdges[-1]): Err.Emsg('mapDepthSlice: Depth2 is greater than max depth - '+str(self.zGrid.cellEdges[-1]))

        if (bestModel):
            model = self.bestParameters
        else:
            model = self.meanParameters

        if withDoi:
            depth1 = np.minimum(self.doi, depth1)
            depth2 = np.minimum(self.doi, depth2)
            z = np.repeat(self.zGrid[:,np.newaxis],self.nPoints,axis=1)
            vals = model[(z > depth1)&(z < depth2)]
        else:
            cell1 = self.zGrid.cellCentres.searchsorted(depth1)
            cell2 = self.zGrid.cellCentres.searchsorted(depth2)

            vals = model[cell1:cell2+1, :]

        log = kwargs.pop('log',False)

        if (reciprocateParameter):
            vals = 1.0/vals
            name = 'Resistivity'
            units = '$\Omega m$'
        else:
            name = 'Conductivity'
            units = '$Sm^{-1}$'

        if (log):
            vals,logLabel=cF._log(vals,log)
            name = logLabel + name
        vals = StatArray.StatArray(vals, name, units)

        h = Histogram1D(np.linspace(vals.min(),vals.max(),nBins))
        h.update(vals)
        h.plot(**kwargs)
        return h


    @property
    def zGrid(self):
        """ Gets the discretization in depth """
        return self.lines[0].mesh.z


    # def getAttribute(self,  mean=False, best=False, opacity=False, doi=False, relErr=False, addErr=False, percent=67.0, force=False):
    #     """ Get a subsurface property """

    #     assert (not all([not mean, not best, not opacity, not doi, not relErr, not addErr])), 'Please choose at least one attribute' + help(self.getAttrubute)

    #     # Turn off attributes that are already loaded
    #     if (mean):
    #         mean=self.mean is None
    #     if (best):
    #         best=self.best is None
    #     if (relErr):
    #         relErr=self.relErr is None
    #     # Getting the doi is cheap, so always ask for it even if opacity is requested
    #     doi = opacity or doi
    #     if (doi):
    #         doi=self.doi is None

    #     if (all([not mean, not best, not opacity, not doi, not relErr, not addErr])):
    #         return

    #     # Get the number of systems
    #     if (relErr or addErr):
    #         self.getNsys()

    #     if (mean or best or doi):
    #         # Get the number of cells
    #         nz = self.zGrid.nCells


    #     if (doi):
    #         self.opacity=np.zeros([nz,self.nPoints], order = 'F')
    #         self.doi = StatArray.StatArray(np.zeros(self.nPoints),'Depth of Investigation','m')
    #     if (relErr):
    #         self.lines[0].getRelativeError()
    #         if (self.nSys > 1):
    #             self.relErr = StatArray.StatArray([self.nPoints, self.nSys],name=self.lines[0].relErr.name,units=self.lines[0].relErr.units, order = 'F')
    #         else:
    #             self.relErr = StatArray.StatArray(self.nPoints,name=self.lines[0].relErr.name,units=self.lines[0].relErr.units, order = 'F')

    #     # Loop over the lines in the data set and get the attributes
    #     print('Reading attributes from dataset results')
    #     Bar=progressbar.ProgressBar()
    #     for i in Bar(range(self.nLines)):

    #         # Perform line getters
    #         if (mean):
    #             self.mean = StatArray.StatArray([nz,self.nPoints], name=self.lines[0].meanParameters.name, units=self.lines[0].meanParameters.units, order = 'F')
    #             self.mean[:, self.lineIndices[i]] = self.lines[i].meanParameters.T
    #             self.lines[i].mean = None # Free memory
    #         if (best):
    #             self.best[:, self.lineIndices[i]] = self.lines[i].bestParameters.T
    #             self.lines[i].best = None # Free memory
    #         if (doi):
    #             # Get the DOI for this line
    #             self.lines[i].getDOI(percent)
    #             self.opacity[:, self.lineIndices[i]] = self.lines[i].opacity.T
    #             self.doi[self.lineIndices[i]] = self.lines[i].doi
    #             self.lines[i].opacity = None # Free memory
    #             self.lines[i].doi = None # Free memory
    #         # Deallocate line attributes to save space
    #         self.lines[i]._hitMap = None

    #     if (xy):
    #         self.points.getBounds() # Get the bounding box


    def getMean3D(self, dx, dy, mask = False, clip = False, force=False, method='ct'):
        """ Interpolate each depth slice to create a 3D volume """
        if (not self.mean3D is None and not force): return

        # Test for an existing file, created with the same parameters.
        # Read it and return if it exists.
        file = 'mean3D.h5'
        if fileIO.fileExists(file):
            variables = hdfRead.read_all(file)
            if (dx == variables['dx'] and dy == variables['dy'] and mask == variables['mask'] and clip == variables['clip'] and method == variables['method']):
                self.mean3D = variables['mean3d']
                return


        method = method.lower()
        if method == 'ct':
            self.__getMean3D_CloughTocher(dx=dx, dy=dy, mask=mask, clip=clip, force=force)
        elif method == 'mc':
            self.__getMean3D_minimumCurvature(dx=dx, dy=dy, mask=mask, clip=clip)
        else:
            assert False, ValueError("method must be either 'ct' or 'mc' ")

        with h5py.File('mean3D.h5','w') as f:
            f.create_dataset(name = 'dx', data = dx)
            f.create_dataset(name = 'dy', data = dy)
            f.create_dataset(name = 'mask', data = mask)
            f.create_dataset(name = 'clip', data = clip)
            f.create_dataset(name = 'method', data = method)
            self.mean3D.toHdf(f,'mean3d')


    def __getMean3D_minimumCurvature(self, dx, dy, mask=None, clip=False):


        x = self.pointcloud.x.deepcopy()
        y = self.pointcloud.y.deepcopy()

        values = self.meanParameters[0, :]
        x1, y1, vals = interpolation.minimumCurvature(x, y, values, self.pointcloud.bounds, dx=dx, dy=dy, mask=mask, clip=clip, iterations=2000, tension=0.25, accuracy=0.01)

        # Initialize 3D volume
        mean3D = StatArray.StatArray(np.zeros([self.zGrid.nCells, y1.size+1, x1.size+1], order = 'F'),name = 'Conductivity', units = '$Sm^{-1}$')
        mean3D[0, :, :] = vals

        # Interpolate for each depth
        print('Interpolating using minimum curvature')
        Bar=progressbar.ProgressBar()
        for i in Bar(range(1, self.zGrid.nCells)):
            # Get the model values for the current depth
            values = self.meanParameters[i, :]
            dum1, dum2, vals = interpolation.minimumCurvature(x, y, values, self.pointcloud.bounds, dx=dx, dy=dy, mask=mask, clip=clip, iterations=2000, tension=0.25, accuracy=0.01)
            # Add values to the 3D array
            mean3D[i, :, :] = vals

        self.mean3D = mean3D


    def __getMean3D_CloughTocher(self, dx, dy, mask=None, clip=False, force=False):

        # Get the discretization
        if (dx is None):
            tmp = self.pointcloud.bounds[1]-self.pointcloud.bounds[0]
            dx = 0.01 * tmp
        assert dx > 0.0, "dx must be positive!"

        # Get the discretization
        if (dy is None):
            tmp = self.pointcloud.bounds[3]-self.pointcloud.bounds[2]
            dy = 0.01 * tmp
        assert dy > 0.0, "dy must be positive!"

        tmp = np.column_stack((self.pointcloud.x, self.points.y))

        # Get the points to interpolate to
        x,y,intPoints = interpolation.getGridLocations2D(self.pointcloud.bounds, dx, dy)

        # Create a distance mask
        if mask:
            self.pointcloud.setKdTree(nDims=2) # Set the KdTree on the data points
            g = np.meshgrid(x,y)
            xi = _ndim_coords_from_arrays(tuple(g), ndim=tmp.shape[1])
            dists, indexes = self.points.kdtree.query(xi)
            iMask = np.where(dists > mask)

        # Get the value bounds
        minV = np.nanmin(self.mean)
        maxV = np.nanmax(self.mean)

        # Initialize 3D volume
        mean3D = StatArray.StatArray(np.zeros([self.zGrid.size, y.nCells, x.nCells], order = 'F'),name = 'Conductivity', units = '$Sm^{-1}$')

        # Triangulate the data locations
        dTri = Delaunay(tmp)

        # Interpolate for each depth
        print('Interpolating using clough tocher')
        Bar=progressbar.ProgressBar()
        for i in Bar(range(self.zGrid.size)):
            # Get the model values for the current depth
            vals1D = self.mean[i,:]
            # Create the interpolant
            f=CloughTocher2DInterpolator(dTri,vals1D)
            # Interpolate to the grid
            vals = f(intPoints)
            # Reshape to a 2D array
            vals = vals.reshape(y.size,x.size)

            # clip values to the observed values
            if (clip):
                vals.clip(minV, maxV)

            # Mask based on distance
            if (mask):
                vals[iMask] = np.nan

            # Add values to the 3D array
            mean3D[i,:,:] = vals
        self.mean3D = mean3D #.reshape(self.zGrid.size*y.size*x.size)


    def interpolate(self, dx, dy, values, method='ct', mask=None, clip=True, **kwargs):

        return self.pointcloud.interpolate(dx=dx, dy=dy, values=values, method=method, mask=mask, clip=clip, **kwargs)


    def map(self, dx, dy, values, method='ct', mask = None, clip = True, **kwargs):
        """ Create a map of a parameter """

        assert values.size == self.nPoints, ValueError("values must have size {}".format(self.nPoints))

        # x, y, z = self.interpolate(50.0, 50.0, values=self.depthSlice(10.0, variable='mean'))
        # print(np.linalg.norm(values - self.depthSlice(10.0, variable='mean')))
        # print(dx, dy, method, mask, clip, extrapolate)

        x, y, z, kwargs = self.interpolate(dx=dx, dy=dy, values=values, method=method, mask=mask, clip=clip, **kwargs)

        return z.pcolor(x=x.edges(), y=y.edges(), **kwargs)


    def mapMarginalProbability(self, dx, dy, depth,  **kwargs):

        cell1 = self.zGrid.cellIndex(depth)

        nFacies = self.MarginalProbability().shape[0]

        for i in range(nFacies):
            plt.subplot(nFacies, 1, i+1)
            self.pointcloud.mapPlot(dx = dx, dy = dy, c = self.marginalProbability[i, :, cell1], **kwargs)


    def percentageParameter(self, value, depth, depth2=None):

        percentage = StatArray.StatArray(np.empty(self.nPoints), name="Probability of {} > {:0.2f}".format(self.meanParameters.name, value), units = self.meanParameters.units)

        print('Calculating percentages', flush = True)
        Bar=progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            percentage[self.lineIndices[i]] = self.lines[i].percentageParameter(value, depth, depth2)

        return percentage


    def depthSlice(self, depth, variable, reciprocateParameter=False, **kwargs):

        out = np.empty(self.nPoints)
        for i, line in enumerate(self.lines):
            p = line._get(variable, reciprocateParameter=reciprocateParameter, **kwargs)
            tmp = line.depthSlice(depth, p, **kwargs)

            out[self.lineIndices[i]] = tmp

        return StatArray.StatArray(out, p.name, p.units)


    def getElevationSlice(self, depth, depth2=None, reciprocateParameter=False, bestModel=False, force=False):

        # Get the depth grid
        assert depth <= self.zGrid.cellEdges[-1], 'Depth is greater than max depth '+str(self.zGrid.cellEdges[-1])
        if (not depth2 is None):
            assert depth2 <= self.zGrid.cellEdges[-1], 'Depth2 is greater than max depth '+str(self.zGrid.cellEdges[-1])
            assert depth <= depth2, 'Depth2 must be >= depth'

        if (bestModel):
            model = self.bestParameters
        else:
            model = self.meanParameters

        model[model == 0.0] = 1.0

        cell1 = self.zGrid.cellIndex(depth)

        if (depth2 is None):
            vals1D = model[cell1, :]
        else:
            cell2 = self.zGrid.cellIndex(depth2)
            vals1D = np.mean(model[cell1:cell2+1,:], axis = 0)

        if (reciprocateParameter):
            vals1D = StatArray.StatArray(1.0 / vals1D, name = 'Resistivity', units = '$\Omega m$')
        else:
            vals1D = StatArray.StatArray(vals1D, name = 'Conductivity', units = '$Sm^{-1}$')
        return vals1D


    def mapAdditiveError(self,dx, dy, system=0, mask = None, clip = True, extrapolate=None, **kwargs):
        """ Create a map of a parameter """
        return self.map(dx = dx, dy = dy, mask = mask, clip = clip, extrapolate=extrapolate, values = self.additiveError[system, :], **kwargs)


    def mapDepthSlice(self, dx, dy, depth, variable, method='ct', mask = None, clip = True, reciprocateParameter=False, useVariance=False, **kwargs):
        """ Create a depth slice through the recovered model """

        vals1D = self.depthSlice(depth=depth, variable=variable, reciprocateParameter=reciprocateParameter)

        if useVariance:
            tmp = self.depthSlice(depth=depth, variable='opacity')
            x, y, a = self.interpolate(dx=dx, dy=dy, values=tmp, method=method, clip=True, **kwargs)
            kwargs['alpha'] = a


        return self.map(dx, dy, vals1D, method=method, mask = mask, clip = clip, **kwargs)


    def mapElevation(self,dx, dy, mask = None, clip = True, extrapolate=None, **kwargs):
        """ Create a map of a parameter """
        return self.map(dx = dx, dy = dy, mask = mask, clip = clip, extrapolate=extrapolate, values = self.elevation, **kwargs)


    def mapRelativeError(self,dx, dy, system=0, mask = None, clip = True, extrapolate=None, **kwargs):
        """ Create a map of a parameter """
        return  self.map(dx = dx, dy = dy, mask = mask, clip = clip, extrapolate=extrapolate, values = self.relativeError[system, :], **kwargs)


    def plotDepthSlice(self, depth, variable, mask = None, clip = True, **kwargs):
        """ Create a depth slice through the recovered model """

        vals1D = self.depthSlice(depth=depth, variable=variable, **kwargs)
        return self.scatter2D(c = vals1D, **kwargs)


    def scatter2D(self, **kwargs):

        if (not 'edgecolor' in kwargs):
            kwargs['edgecolor'] = 'k'
        if (not 's' in kwargs):
            kwargs['s'] = 10.0

        return self.pointcloud.scatter2D(**kwargs)


    def plotAdditiveError(self, system=0, **kwargs):
        """ Plot the observation locations """
        return self.scatter2D(c=self.additiveError[system, :], **kwargs)


    def plotDataMisfit(self, normalized=True, **kwargs):
        """ Plot the observation locations """
        x = self.dataMisfit
        if normalized:
            x = x / self.bestData.nActiveChannels
            x._name = "Normalized data misfit"

        return self.scatter2D(c=x, **kwargs)


    def plotInterfaceProbability(self, depth, lowerThreshold=0.0, **kwargs):

        cell1 = self.zGrid.cellIndex(depth)

        slce = self.interfaces[cell1, :]
        if lowerThreshold > 0.0:
            slce = self.interfaces[cell1, :].deepcopy()
            slce[slce < lowerThreshold] = np.nan

        return self.scatter2D(c = slce, **kwargs)


    def plotElevation(self, **kwargs):
        """ Plot the observation locations """
        return self.scatter2D(c=self.elevation, **kwargs)


    def plotRelativeError(self, system=0, **kwargs):
        """ Plot the observation locations """
        return self.scatter2D(c=self.relativeError[system, :], **kwargs)


    def plotCrossPlot(self, bestModel=True, withDoi=True, reciprocateParameter=True, log10=True, **kwargs):
        """ Plot the cross plot of a model against depth """

        tmp = self.getParVsZ(bestModel=bestModel, withDoi=withDoi, reciprocateParameter=reciprocateParameter, log10=log10)
        # Repeat the depths for plotting
        cP.plot(tmp[:,0], tmp[:,1], **kwargs)
        if (bestModel):
            cP.xlabel(self.best.getNameUnits())
        else:
            cP.xlabel(self.mean.getNameUnits())
        cP.ylabel(self.zGrid.getNameUnits())
        return tmp


    def getParVsZ(self, bestModel=False, withDoi=True, reciprocateParameter=True, log10=True, clipNan=True):
        """ Get the depth and parameters, optionally within the doi """
        # Get the depths
        z = np.tile(self.zGrid,self.nPoints)

        if (bestModel):
            self.getAttribute(best=True, doi=withDoi)
            model = np.zeros(self.best.shape)
            model[:,:] = self.best
        else:
            self.getAttribute(mean=True, doi=withDoi)
            model = np.zeros(self.best.shape)
            model[:,:] = self.mean

        if (withDoi):
            zTmp = np.repeat(self.zGrid[:,np.newaxis],self.nPoints,axis=1)
            model[zTmp > self.doi] = np.nan

        model = model.reshape(model.size, order='F')

        if reciprocateParameter:
            model = 1.0/model.reshape(model.size, order='F')

        if log10:
            model = np.log10(model)

        res = StatArray.StatArray(np.column_stack((model,z)))

        if (clipNan):
            res = res[np.logical_not(np.isnan(res[:,0]))]

        return res

    def kMeans(self, nClusters, precomputedParVsZ=None, standardize=False, log10Depth=False, plot=False, bestModel=True, withDoi=True, reciprocateParameter=True, log10=True, clipNan=True, **kwargs):
        """  """
        if (precomputedParVsZ is None):
            ParVsZ = self.getParVsZ(bestModel=bestModel, withDoi=withDoi, reciprocateParameter=reciprocateParameter, log10=log10, clipNan=clipNan)
        else:
            ParVsZ = precomputedParVsZ

        assert isinstance(ParVsZ, StatArray.StatArray), "precomputedParVsZ must be an StatArray"

        if (log10Depth):
            ParVsZ[:,1] = np.log10(ParVsZ[:,1])

        return ParVsZ.kMeans(nClusters, standardize=standardize, nIterations=10, plot=plot, **kwargs)


    def GMM(self, ParVsZ, clusterID, trainPercent=90.0, plot=True):
        """ Classify the subsurface parameters """
        assert isinstance(ParVsZ, StatArray.StatArray), "ParVsZ must be an StatArray"
        ParVsZ.GMM(clusterID, trainPercent=trainPercent, covType=['spherical','tied','diag','full'], plot=plot)



    def toVTK(self, fName, dx, dy, mask=False, clip=False, force=False, method='ct'):
        """ Convert a 3D volume of interpolated values to vtk for visualization in Paraview """

        self.getMean3D(dx=dx, dy=dy, mask=mask, clip=clip, force=force, method=method)
        self.pointcloud.getBounds()

        x, y, intPoints = interpolation.getGridLocations2D(self.pointcloud.bounds, dx, dy)
        z = self.zGrid


        from pyvtk import VtkData, UnstructuredGrid, PointData, CellData, Scalars

        # Get the 3D dimensions
        mx = x.size
        my = y.size
        mz = z.nCells

        nPoints = mx * my * mz
        nCells = (mx-1)*(my-1)*(mz-1)

        # Interpolate the elevation to the grid nodes
        if (method == 'ct'):
            tx,ty, vals, k = self.pointcloud.interpCloughTocher(dx = dx,dy=dy, values=self.elevation, mask = mask, clip = clip, extrapolate='nearest')
        elif (method == 'mc'):
            tx,ty, vals, k = self.pointcloud.interpMinimumCurvature(dx = dx, dy=dy, values=self.elevation, mask = mask, clip = clip)

        vals = vals[:my,:mx]
        vals = vals.reshape(mx*my)

        # Set up the nodes and voxel indices
        points = np.zeros([nPoints,3], order='F')
        points[:,0] = np.tile(x, my*mz)
        points[:,1] = np.tile(y.repeat(mx), mz)
        points[:,2] = np.tile(vals, mz) - z.cellCentres.repeat(mx*my)

        # Create the cell indices into the points
        p = np.arange(nPoints).reshape((mz, my, mx))
        voxels = np.zeros([nCells, 8], dtype=np.int)
        iCell = 0
        for k in range(mz-1):
            k1 = k + 1
            for j in range(my-1):
                j1 = j + 1
                for i in range(mx-1):
                    i1 = i + 1
                    voxels[iCell,:] = [p[k1,j,i],p[k1,j,i1],p[k1,j1,i1],p[k1,j1,i], p[k,j,i],p[k,j,i1],p[k,j1,i1],p[k,j1,i]]
                    iCell += 1

        # Create the various point data
        pointID = Scalars(np.arange(nPoints), name='Point iD')
        pointElev = Scalars(points[:,2], name='Point Elevation (m)')

        tmp = self.mean3D.reshape(np.size(self.mean3D))
        tmp[tmp == 0.0] = np.nan

        print(np.nanmin(tmp), np.nanmax(tmp))
        tmp1 = 1.0 / tmp

        print(np.nanmin(tmp), np.nanmax(tmp))
        pointRes = Scalars(tmp1, name = 'log10(Resistivity) (Ohm m)')
        tmp1 = np.log10(tmp)

        pointCon = Scalars(tmp1, name = 'log10(Conductivity) (S/m)')

        print(nPoints, tmp.size)

        PData = PointData(pointID, pointElev, pointRes)#, pointCon)
        CData = CellData(Scalars(np.arange(nCells),name='Cell iD'))
        vtk = VtkData(
              UnstructuredGrid(points,
                               hexahedron=voxels),
#                               ),
              PData,
              CData,
              'Some Name'
              )

        vtk.tofile(fName, 'binary')
