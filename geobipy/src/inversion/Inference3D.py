""" @DataSetResults
Class to handle the HDF5 result files for a full data set.
 """
import time
from datetime import timedelta
from ..base import Error as Err
import matplotlib.pyplot as plt
import numpy as np
import h5py
from datetime import timedelta

#import numpy.ma as ma
from cached_property import cached_property
from ..classes.core.myObject import myObject
from ..classes.core import StatArray
from ..base import fileIO
from ..base.MPI import loadBalance1D_shrinkingArrays

from ..classes.statistics.Histogram1D import Histogram1D
from ..classes.statistics.Hitmap2D import Hitmap2D
from ..classes.statistics.mixPearson import mixPearson
from ..classes.pointcloud.PointCloud3D import PointCloud3D
from ..base import interpolation as interpolation
from .inference import initialize
from .Inference1D import Inference1D
from .Inference2D import Inference2D


from ..classes.data.dataset.Data import Data
from ..classes.data.datapoint.DataPoint import DataPoint

#from ..classes.statistics.Distribution import Distribution
from ..base.HDF import hdfRead
from ..base import plotting as cP
from ..base import utilities as cF
from os.path import join
from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

from os import listdir
import progressbar

class Inference3D(myObject):
    """ Class to define results from Inv_MCMC for a full data set """

    def __init__(self, directory, system_file_path, files=None, world=None, mode='r+'):
        """ Initialize the 3D inference
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
        self._world = world
        for i in range(self.nLines):
            fName = self.h5files[i]

            LR = Inference2D(fName, system_file_path=system_file_path, mode=mode, world=world)
            self._lines.append(LR)
            self._lineNumbers[i] = LR.line

        self.kdtree = None
        self.doi = None
        self.doi2D = None
        self.mean3D = None
        self.best3D = None
        self._facies = None
        self.system_file_path = system_file_path


    @property
    def world(self):
        return self._world

    @property
    def parallel_access(self):
        return not self.world is None


    def open(self, mode='r+', world=None):
        """ Check whether the file is open """
        for line in self.lines:
            line.open(mode=mode, world=world)


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

        # dataset.readSystemFile(userParameters.systemFilename)

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
        options, Mod, DataPoint, _, _, _, _ = initialize(options, DataPoint)

        # Create the results template
        Res = Inference1D(DataPoint, Mod,
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
            lr = Inference2D()
            lr.createHdf(H5File, fiducials[fiducialsForLine], Res)
            self._lines.append(lr)
            print('Time to create line {} with {} data points: {} h:m:s'.format(line, fiducialsForLine.size, str(timedelta(seconds=time.time()-t0))))

    def _createHDF5_datapoint(self, datapoint, userParameters):

        print('stuff')

    @property
    def h5files(self):
        """ Get the list of line result files for the dataset """
        return self._h5files


    def line(self, line_number):
        """Get the inversion results for the given line.

        Parameters
        ----------
        fiducial : float
            Unique fiducial of the data point.

        Returns
        -------
        out : geobipy.Results
            The inversion results for the line.

        """
        index = self.lineIndex(lineNumber=line_number)
        return self.lines[index]

    @property
    def lines(self):
        return self._lines

    @property
    def lineNumbers(self):
        return self._lineNumbers


    @property
    def nLines(self):
        return np.size(self._h5files)


    def _get(self, variable, reciprocateParameter=False, **kwargs):

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
            assert 'index' in kwargs, ValueError('Please specify keyword "index" when requesting marginalProbability')
            return self.marginalProbability[:, :, kwargs["index"]].T


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

        additiveError = StatArray.StatArray((self.nSystems, self.nPoints), name=self.lines[0].additiveError.name, units=self.lines[0].additiveError.units, order = 'F')

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

        bestParameters = StatArray.StatArray((self.zGrid.nCells, self.nPoints), name=self.lines[0].bestParameters.name, units=self.lines[0].bestParameters.units, order = 'F')

        print('Reading best parameters', flush=True)
        Bar=progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            bestParameters[:, self.lineIndices[i]] = self.lines[i].bestParameters
            del self.lines[i].__dict__['bestParameters'] # Free memory

        return bestParameters


    @cached_property
    def marginalProbability(self):

        marginalProbability = StatArray.StatArray((self.nPoints, self.zGrid.nCells, self.lines[0].marginalProbability.shape[-1]), name=self.lines[0].marginalProbability.name, units=self.lines[0].marginalProbability.units)

        print('Reading marginal probability', flush=True)
        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            marginalProbability[self.lineIndices[i], :, :] = self.lines[i].marginalProbability
            del self.lines[i].__dict__['marginalProbability'] # Free memory

        return marginalProbability


    # def computeMarginalProbability(self, fractions, distributions, **kwargs):
    #     for line in self.lines:
    #         line.computeMarginalProbability(fractions, distributions, **kwargs)


    def compute_MinsleyFoksBedrosian2020_P_lithology(self, global_mixture_hdf5, local_mixture_hdf5, log=None):
        """Compute the cluster probability using Minsley Foks 2020.

        Compute the probability of clusters using both a global mixture model and a local mixture model fit to the histogram.
        In MinsleyFoksBedrosian2020, the local mixture models were generated by fitting the histogram's estimated pdf while the global mixture model
        is used to label all local mixture models on a dataset scale.

        Parameters
        ----------
        global_mixture : sklearn.mixture
            Global mixture model with n components to charactize the potential labels that local mixture might belong to.
        local_mixture : geobipy.Mixture
            Mixture model with k components fit to the estimated pdf of the histogram.
        log : scalar or 'e', optional
            Take the log of the histogram bins.
            Defaults to None.

        Returns
        -------
        probabilities : (self.shape[axis] x n) array of the probability that the local mixtures belong to each global mixture component.

        """

        global_mixture = cF.load_gmm(global_mixture_hdf5, sort_by_means=True)

        z = self.zGrid

        if self.parallel_access:

            local_mixture_h5 = h5py.File(local_mixture_hdf5, 'r', driver='mpio', comm=self.world)
            probabilities_h5 = h5py.File('P_class.h5', 'w', driver='mpio', comm=self.world)

            starts, chunks = loadBalance1D_shrinkingArrays(self.nPoints, self.world.size)
            r = range(starts[self.world.rank], starts[self.world.rank] + chunks[self.world.rank])

            if self.world.rank == 0:
                Bar = progressbar.ProgressBar()
                r = Bar(r)

        else:

            local_mixture_h5 = h5py.File(local_mixture_hdf5, 'r')
            probabilities_h5 = h5py.File('P_class.h5', 'w')

            Bar = progressbar.ProgressBar()
            r = Bar(range(self.nPoints))

        # Create the space in HDF5
        probabilities = StatArray.StatArray((z.nCells, global_mixture.n_components), name='probabilities')
        probabilities.createHdf(probabilities_h5, 'probabilities', nRepeats=self.nPoints)

        for i in r:
            # Read the local fits
            local_fits = []
            for j in range(z.nCells):
                local_fits.append(mixPearson().fromHdf(local_mixture_h5['fits'], index=(i, j)))

            # Get the 2D posterior for parameter
            posterior = self.hitmap(index=i)

            # Compute cluster probabilities
            probabilities = posterior.compute_MinsleyFoksBedrosian2020_P_lithology(global_mixture=global_mixture, local_mixture=local_fits, log=log)

            # Write the probabilities to file
            probabilities.writeHdf(probabilities_h5, 'probabilities', index=i)


        local_mixture_h5.close()
        probabilities_h5.close()


    @cached_property
    def dataMisfit(self):
        return self.bestData.dataMisfit()


    def hitmap(self, fiducial=None, index=None):
        """Get the hitmap for the given fiducial or index

        Parameters
        ----------
        fiducial : float, optional
            Fiducial of the required hitmap.
            Defaults to None.
        index : int, optional
            Index of the required hitmap.
            Defaults to None.

        Returns
        -------
        geobipy.Hitmap : Parameter posterior.

        """
        iLine, index = self.lineIndex(fiducial=fiducial, index=index)
        return self.lines[iLine].hitmap(index=index)


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
        interfaces = StatArray.StatArray((self.zGrid.nCells, self.nPoints), name=self.lines[0].interfaces.name, units=self.lines[0].interfaces.units)

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

        meanParameters = StatArray.StatArray((self.zGrid.nCells, self.nPoints), name=self.lines[0].meanParameters.name, units=self.lines[0].meanParameters.units, order = 'F')

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


    @property
    def nPoints(self):
        """ Get the total number of data points """
        tmp = np.asarray([line.nPoints for line in self.lines])
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

        return PointCloud3D(x, y, z, e)


    @cached_property
    def relativeError(self):

        relativeError = StatArray.StatArray((self.nSystems, self.nPoints), name=self.lines[0].relativeError.name, units=self.lines[0].relativeError.units, order = 'F')

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


    def inference_1d(self, fiducial=None, index=None):
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

        return self.lines[lineIndex].inference_1d(fidIndex)


    def lineIndex(self, lineNumber=None, fiducial=None, index=None):
        """Get the line index """
        tmp = np.sum([not x is None for x in [lineNumber, fiducial, index]])
        assert tmp == 1, Exception("Please specify one argument, lineNumber, fiducial, or index")

        index = np.atleast_1d(index)


        if not lineNumber is None:
            assert lineNumber in self.lineNumbers, ValueError("line {} not found in data set".format(lineNumber))
            return np.squeeze(np.where(self.lineNumbers == lineNumber)[0])

        if not fiducial is None:
            return self.fiducialIndex(fiducial)[0]

        assert np.all(index <= self.nPoints-1), IndexError('index {} is out of bounds for data point index with size {}'.format(index, self.nPoints))

        cumPoints = self._cumNpoints - 1

        iLine = cumPoints.searchsorted(index)
        i = np.squeeze(np.where(iLine > 0))
        index[i] -= self._cumNpoints[iLine[i]-1]

        return np.squeeze(iLine), np.squeeze(index)


    def fiducial(self, index):
        """ Get the fiducial of the given data point """
        iLine, index = self.lineIndex(index=index)
        iLine = np.atleast_1d(iLine)
        index = np.atleast_1d(index)

        out = np.empty(np.size(index))
        for i in range(np.size(index)):
            out[i] = self.lines[iLine[i]].fiducials[index[i]]

        return out


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

        assert False, ValueError("fiducial not present in this data set")


    def fit_estimated_pdf_mpi(self, intervals=None, **kwargs):

        from mpi4py import MPI
        from geobipy.src.base import MPI as myMPI

        max_distributions = kwargs.get('max_distributions', 3)
        kwargs['track'] = False

        if intervals is None:
            intervals = self.hitmap(index=0).yBins

        nIntervals = np.size(intervals) - 1

        hdfFile = h5py.File("fits.h5", 'w', driver='mpio', comm=self.world)

        a = np.zeros(max_distributions)
        mixture = mixPearson(a, a, a, a)
        mixture.createHdf(hdfFile, 'fits', nRepeats=(self.nPoints, nIntervals))

        if self.world.rank == 0:  ## Master Task
            nFinished = 0
            nSent = 0

            # Send out the first indices to the workers
            for iWorker in range(1, self.world.size):
                # Get a datapoint from the file.
                if nSent < self.nPoints:
                    continueRunning = True
                    self.world.send(True, dest=iWorker)
                    self.world.send(nSent, dest=iWorker)

                    nSent += 1
                else:
                    continueRunning = False
                    self.world.send(False, dest=iWorker)

            t0 = MPI.Wtime()

            myMPI.print("Initial posteriors sent. Master is now waiting for requests")

            # Now wait to send indices out to the workers as they finish until the entire data set is finished
            while nFinished < self.nPoints:
                # Wait for a worker to request the next data point
                status = MPI.Status()
                dummy = self.world.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
                requestingRank = status.Get_source()

                nFinished += 1

                # If DataPoint is None, then we reached the end of the file and no more points can be read in.
                if nSent < self.nPoints:
                    # Send the kill switch to the worker to shut down.
                    continueRunning = True
                    self.world.send(True, dest=requestingRank)
                    self.world.send(nSent, dest=requestingRank)

                    nSent += 1
                else:
                    continueRunning = False
                    self.world.send(False, dest=requestingRank)


                report = (nFinished % (self.world.size - 1)) == 0 or nFinished == self.nPoints

                if report:
                    e = MPI.Wtime() - t0
                    elapsed = str(timedelta(seconds=e))
                    eta = str(timedelta(seconds=(self.nPoints / nFinished-1) * e))
                    myMPI.print("Remaining Points {}/{} || Elapsed Time: {} h:m:s || ETA {} h:m:s".format(self.nPoints-nFinished, self.nPoints, elapsed, eta))

        else:
            # Initialize the worker process to go
            Go = True

            # Wait till you are told what to process next
            continueRunning = self.world.recv(source=0)
            # If we continue running, receive the next DataPoint. Otherwise, shutdown the rank
            if continueRunning:
                index = self.world.recv(source=0)
            else:
                Go = False

            while Go:
                hm = self.hitmap(index=index)

                if not np.all(hm.counts == 0):
                    mixtures = hm.fit_estimated_pdf(iPoint=index, rank=self.world.rank, **kwargs)

                    for j, m in enumerate(mixtures):
                        if not m is None:
                            m.writeHdf(hdfFile, 'fits', index=(index, j))

                self.world.send(1, dest=0)
                # Wait till you are told whether to continue or not
                continueRunning = self.world.recv(source=0)

                # If we continue running, receive the next DataPoint. Otherwise, shutdown the rank
                if continueRunning:
                    index = self.world.recv(source=0)
                else:
                    Go = False


    def fit_mixture(self, intervals, **kwargs):

        if self.parallel_access:
            return self.fit_mixture_mpi(intervals, **kwargs)
        else:
            return self.fit_mixture_serial(intervals, **kwargs)


    def fit_mixture_serial(self, intervals, **kwargs):
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


    # def fit_mixture_mpi(self, intervals, **kwargs):
    #     """Uses Mixture modelling to fit disrtibutions to the hitmaps for the specified intervals.

    #     This mpi version fits all hitmaps individually throughout the data set.
    #     This provides detailed fits, but requires a lot of compute, hence the mpi enabled version.

    #     Parameters
    #     ----------
    #     intervals : array_like
    #         Depth intervals between which the marginal histogram is computed before fitting.

    #     See Also
    #     --------
    #     geobipy.Histogram1D.fit_mixture
    #         For details on the fitting arguments.

    #     """

    #     from mpi4py import MPI

    #     world = self.world

    #     kwargs['k'] = kwargs.pop('k', [1, 5])
    #     k = kwargs['k']

    #     maxClusters = (k[1] - k[0]) + 1
    #     nIntervals = np.size(intervals) - 1

    #     tmp = locals()
    #     for key in ['self', 'MPI', 'world', 'k']:
    #         tmp.pop(key, None)
    #     command = str(tmp)

    #     for i in range(self.nLines):

    #         means = StatArray.StatArray((self.lines[i].nPoints, nIntervals, maxClusters), "fit means")
    #         variances = StatArray.StatArray((self.lines[i].nPoints, nIntervals, maxClusters), "fit variances")

    #         if 'mixture_fits' in self.lines[i].hdfFile:
    #             saved_command = self.lines[i].hdfFile['/mixture_fits'].attrs['command']
    #             if command != saved_command:
    #                 del self.lines[i].hdfFile['/mixture_fits']

    #                 means.createHdf(self.lines[i].hdfFile, "/mixture_fits/means")
    #                 variances.createHdf(self.lines[i].hdfFile, "/mixture_fits/variances")

    #                 print('writing \n', command)
    #                 self.lines[i].hdfFile['/mixture_fits'].attrs['command'] = command

    #     # Distribute the points amongst cores.
    #     starts, chunks = loadBalance1D_shrinkingArrays(self.nPoints, self.world.size)

    #     chunk = chunks[self.world.rank]
    #     chunk = 10
    #     i0 = starts[self.world.rank]
    #     i1 = i0 + chunk

    #     iLine, index = self.lineIndex(index=np.arange(i0, i1))

    #     tBase = MPI.Wtime()
    #     t0 = tBase

    #     nUpdate = np.int(0.1 * chunk)
    #     counter = 0

    #     for i in range(chunk):

    #         iL = iLine[i]
    #         ind = index[i]

    #         line = self.lines[iL]

    #         hm = line.get_hitmap(ind)

    #         d, a = hm.fit_mixture(intervals, track=False, **kwargs)

    #         for j in range(nIntervals):
    #             dm = np.squeeze(d[j].means_[a[j]])
    #             dv = np.squeeze(d[j].covariances_[a[j]])

    #             nD = np.size(dm)

    #             line.hdfFile['/mixture_fits/means/data'][ind, j, :nD] = dm
    #             line.hdfFile['/mixture_fits/variances/data'][ind, j, :nD] = dv

    #         counter += 1
    #         if counter == nUpdate:
    #             print('rank {}, line/fiducial {}/{}, iteration {}/{},  time/dp {} h:m:s'.format(world.rank, self.lineNumbers[iL], line.fiducials[ind], i+1, chunk, str(timedelta(seconds=MPI.Wtime()-t0)/nUpdate)), flush=True)
    #             t0 = MPI.Wtime()
    #             counter = 0

    #     print('rank {} finished in {} h:m:s'.format(world.rank, str(timedelta(seconds=MPI.Wtime()-tBase))))


    def histogram(self, nBins, **kwargs):
        """ Compute a histogram of the model, optionally show the histogram for given depth ranges instead """

        log = kwargs.pop('log', False)

        values = self._get(**kwargs)

        if (log):
            values, logLabel = cF._log(values, log)
            values.name = logLabel + values.name

        values = StatArray.StatArray(values, values.name, values.units)

        h = Histogram1D(np.linspace(np.nanmin(values), np.nanmax(values), nBins))
        h.update(values)

        h.plot()
        return h


    @property
    def zGrid(self):
        """ Gets the discretization in depth """
        return self.lines[0].mesh.z


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

        x, y, z, kwargs = self.interpolate(dx=dx, dy=dy, values=values, method=method, mask=mask, clip=clip, **kwargs)

        if 'alpha' in kwargs:
            x, y, a, kwargs = self.interpolate(dx=dx, dy=dy, values=kwargs['alpha'], method=method, mask=mask, clip=clip, **kwargs)
            kwargs['alpha'] = a

        return z.pcolor(x=x.edges(), y=y.edges(), **kwargs)


    def mapMarginalProbability(self, dx, dy, depth,  **kwargs):

        cell1 = self.zGrid.cellIndex(depth)

        nClusters = self.marginalProbability.shape[-1]

        ax = plt.subplot(nClusters, 1, 1)
        for i in range(nClusters):
            if i > 0:
                plt.subplot(nClusters, 1, i+1, sharex=ax, sharey=ax)
            self.pointcloud.mapPlot(dx = dx, dy = dy, c = self.marginalProbability[:, cell1, i], **kwargs)


    def percentageParameter(self, value, depth, depth2=None):

        percentage = StatArray.StatArray(np.empty(self.nPoints), name="Probability of {} > {:0.2f}".format(self.meanParameters.name, value), units = self.meanParameters.units)

        print('Calculating percentages', flush = True)
        Bar=progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            percentage[self.lineIndices[i]] = self.lines[i].percentageParameter(value, depth, depth2)

        return percentage


    def depthSlice(self, depth, variable, reciprocateParameter=False, **kwargs):

        out = np.empty(self.nPoints)

        index = kwargs.pop('index', None)
        for i, line in enumerate(self.lines):
            p = line._get(variable, reciprocateParameter=reciprocateParameter, index=index, **kwargs)
            tmp = line.depthSlice(depth, p, **kwargs)

            out[self.lineIndices[i]] = tmp

        return StatArray.StatArray(out, p.name, p.units)


    def mapAdditiveError(self,dx, dy, system=0, mask = None, clip = True, useVariance=False, **kwargs):
        """ Create a map of a parameter """

        if useVariance:
            for line in self.lines:
                line.compute_additive_error_opacity()
            alpha = np.hstack([line.addErr_opacity for line in self.lines])
            kwargs['alpha'] = alpha

        return self.map(dx = dx, dy = dy, mask = mask, clip = clip, values = self.additiveError[system, :], **kwargs)


    def mapDepthSlice(self, dx, dy, depth, variable, method='ct', mask = None, clip = True, reciprocateParameter=False, useVariance=False, index=None, **kwargs):
        """ Create a depth slice through the recovered model """

        vals1D = self.depthSlice(depth=depth, variable=variable, reciprocateParameter=reciprocateParameter, index=index)

        if useVariance:
            tmp = self.depthSlice(depth=depth, variable='opacity')
            x, y, a = self.interpolate(dx=dx, dy=dy, values=tmp, method=method, clip=True, **kwargs)
            kwargs['alpha'] = a

        return self.map(dx, dy, vals1D, method=method, mask = mask, clip = clip, **kwargs)


    def mapElevation(self,dx, dy, mask = None, clip = True, **kwargs):
        """ Create a map of a parameter """
        return self.map(dx = dx, dy = dy, mask = mask, clip = clip, values = self.elevation, **kwargs)


    def map_highest_marginal(self, dx, dy, depth, method='ct', mask=None, clip=True, reciprocateParameter=False, useVariance=False, **kwargs):

        nClusters = self.marginalProbability.shape[-1]

        vals1D = self.depthSlice(depth=depth, variable='marginalProbability', reciprocateParameter=reciprocateParameter, index=0)
        x, y, z, dum = self.interpolate(dx, dy, vals1D, method=method, mask=mask, clip=clip, **kwargs)

        interpolated_marginal = np.zeros((*z.shape, nClusters))
        interpolated_marginal[:, :, 0] = z

        for i in range(1, nClusters):
            vals1D = self.depthSlice(depth=depth, variable='marginalProbability', reciprocateParameter=reciprocateParameter, index=i)
            x, y, interpolated_marginal[:, :, i], dum = self.interpolate(dx, dy, vals1D, method=method, mask=mask, clip=clip, **kwargs)

        highest = StatArray.StatArray((np.argmax(interpolated_marginal, axis=-1)).astype(np.float32))
        msk = np.all(np.isnan(interpolated_marginal), axis=-1)
        highest[msk] = np.nan

        ax, pc, cb = highest.pcolor(x.edges(), y.edges(), vmin=0, vmax=nClusters-1, cmapIntervals=nClusters, **dum)

        offset = (nClusters-1) / (2*nClusters)
        ticks = np.arange(offset, nClusters-offset, 2.0*offset)
        tick_labels = np.arange(nClusters)+1
        cb.set_ticks(ticks)
        cb.set_ticklabels(tick_labels)

        return ax, pc, cb


    def mapRelativeError(self,dx, dy, system=0, mask = None, clip = True, useVariance=False, **kwargs):
        """ Create a map of a parameter """

        if useVariance:
            for line in self.lines:
                line.compute_relative_error_opacity()
            alpha = np.hstack([line.relErr_opacity for line in self.lines])
            kwargs['alpha'] = alpha

        return  self.map(dx = dx, dy = dy, mask = mask, clip = clip, values = self.relativeError[system, :], **kwargs)


    def plotDepthSlice(self, depth, variable, mask = None, clip = True, index=None, **kwargs):
        """ Create a depth slice through the recovered model """

        vals1D = self.depthSlice(depth=depth, variable=variable, index=index, **kwargs)
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
