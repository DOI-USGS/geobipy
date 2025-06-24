""" @DataSetResults
Class to handle the HDF5 result files for a full data set.
"""
from os.path import join
from os import listdir
from copy import deepcopy
from pathlib import Path
import time
from pprint import pprint
from numpy import atleast_1d, arange, argsort, argwhere, asarray, column_stack, cumsum, divide, empty, exp, float64
from numpy import full, hstack, integer, int32, int64, log10, logical_not, linspace, load, max, nan,  nanmin, nanmax, newaxis
from numpy import unique, r_, repeat, s_, save, size, sort, squeeze, std, sum, tile, uint64, unique, vstack, where, zeros
from numpy import all as npall

from numpy.random import Generator, PCG64DXSM

from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import h5py
from datetime import timedelta

from sklearn.mixture import GaussianMixture
# from smm import SMM
from cached_property import cached_property
from ..classes.core.myObject import myObject
from ..classes.statistics.StatArray import StatArray
from ..base import fileIO
from ..base.MPI import loadBalance1D_shrinkingArrays

from ..classes.statistics.mixPearson import mixPearson
from ..classes.data.dataset.Data import Data
from ..classes.mesh.RectilinearMesh3D import RectilinearMesh3D
from ..classes.model.Model import Model
from .Inference1D import Inference1D
from .Inference2D import Inference2D

from ..base.HDF import hdfRead
from ..base import plotting as cP
from ..base import utilities as cF

import progressbar

class Inference3D(myObject):
    """ Class to define results from Inv_MCMC for a full data set """

    def __init__(self, data, prng, world=None, global_access=True, debug=False):#directory, system_file_path, files=None, mpi_enabled=False, mode='r+', world=None):
        """ Initialize the 3D inference
        directory = directory containing folders for each line of data results
        """
        self.world = world
        self.global_access = global_access

        self.data = data

        self.prng = prng

        # self.directory = directory
        # self._h5files = None
        # self._nPoints = None
        # self.cumNpoints = None
        # self.bounds = None

        # self._get_h5Files(directory, files)

        # if mpi_enabled:
        #     if world is None:
        #         from mpi4py import MPI
        #         world = MPI.COMM_WORLD
        #     else:
        #         world = world

        # self._set_inference2d(system_file_path, mode, world)

        # self.world = world

        # self._mesh3d = None
        # self.kdtree = None
        # self._mean3D = None
        # self.best3D = None
        # self._facies = None
        # self.system_file_path = system_file_path
        # self.nPoints

    def __deepcopy__(self, memo={}):
        return None


    @classmethod
    def fromHdf(cls, directory, prng, world=None, global_access=False, **kwargs):
        """Initialize a 3D set of Inferences with HDF5 files.

        Only opens the files, data are loaded when required.

        Parameters
        ----------
        directory : str
            The folder of h5files, each file is a line.
        prng : numpy.Generator
            The pseudo-random number generator to instantiate. Only used when we are inferring, not reading and plotting
        world : mpi4p4.MPI.COMM_WORLD, optional
            MPI communicator used for parallel processing.
        global_access : bool, optional
            Whether to open all hdf5 files with the communicator or chunks of files locally on each rank.

        """

        # h5_files = Inference3D._get_h5Files(directory)

        # if world is None:
        #     lines = [Inference2D.fromHdf(file, prng=prng, **kwargs) for file in h5_files]
        # else:
        # lines = [Inference2D.fromHdf(file, prng=prng, world = world, **kwargs) for file in h5_files]

        self = cls(None, world=world, prng=prng, global_access=global_access)
        self.open(directory=directory, **kwargs)

        return self

    def open(self, directory, **kwargs):
        assert kwargs.get('mode', 'r') in ('r', 'r+', 'a', 'w'), ValueError("mode must be in ('r', 'r+', 'a', 'w')")
        self._lines = [Inference2D.fromHdf(file, prng=self.prng, world=self.world, **kwargs) for file in Inference3D._get_h5Files(directory)]

    @staticmethod
    def _get_h5Files(directory, files=None, **kwargs):

        if not files is None:
            if not isinstance(files, list):
                files = [files]
        else:
            files = [f for f in listdir(directory) if f.endswith('.h5')]

        out = []
        for file in files:
            fName = join(directory, file)
            assert fileIO.fileExists(fName), Exception("HDF5 file {} does not exist".format(fName))
            out.append(Path(fName))

        return sorted(out)

    def _set_inference2d(self, mode='r+', world=None):
        lines = []
        line_number = empty(self.nLines)

        for i, file in enumerate(self.h5files):
            LR = Inference2D()
            LR.open(filename=file, mode=mode, world=world)
            lines.append(LR)
            line_number[i] = LR.line_number

        self.lines = lines
        self.line_number = line_number

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        # assert isinstance(value, Data), TypeError("data must have type geobipy.Data")
        self._data = value

    @property
    def global_access(self):
        return self._global_access

    @global_access.setter
    def global_access(self, value):
        assert isinstance(value, bool), ValueError("global_access must have type bool")
        self._global_access = value

    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, value):
        self._world = value

    @property
    def parallel_access(self):
        return not self.world is None

    @cached_property
    def point_chunks(self):
        # assert self.parallel_access, Exception("Parallel access not enabled.  Pass an MPI communicator when instantiating Inference3D.")
        if self.parallel_access:
            _, _point_chunks = loadBalance1D_shrinkingArrays(self.nPoints, self.world.size)
        else:
            _point_chunks = full(1, fill_value=self.nPoints, dtype=int32)
        return _point_chunks

    @property
    def point_ends(self):
        # assert self.parallel_access, Exception("Parallel access not enabled.  Pass an MPI communicator when instantiating Inference3D.")
        return self.point_starts + self.point_chunks

    @cached_property
    def point_starts(self):
        # assert self.parallel_access, Exception("Parallel access not enabled.  Pass an MPI communicator when instantiating Inference3D.")
        if self.parallel_access:
            _point_starts, _ = loadBalance1D_shrinkingArrays(self.nPoints, self.world.size)
        else:
            _point_starts = zeros(1, dtype=int32)
        return _point_starts

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

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    @cached_property
    def line_chunks(self):
        # assert self.parallel_access, Exception("Parallel access not enabled.  Pass an MPI communicator when instantiating Inference3D.")
        if self.parallel_access:
            _, _line_chunks = loadBalance1D_shrinkingArrays(self.nLines, self.world.size)
        else:
            _line_chunks = full(1, fill_value=self.nLines, dtype=int32)
        return _line_chunks

    @property
    def line_ends(self):
        # assert self.parallel_access, Exception("Parallel access not enabled.  Pass an MPI communicator when instantiating Inference3D.")
        return self.line_starts + self.line_chunks

    @cached_property
    def line_starts(self):
        # assert self.parallel_access, Exception("Parallel access not enabled.  Pass an MPI communicator when instantiating Inference3D.")
        if self.parallel_access:
            _line_starts, _ = loadBalance1D_shrinkingArrays(self.nLines, self.world.size)
        else:
            _line_starts = zeros(1, dtype=int32)
        return _line_starts

    @property
    def rank(self):
        return self.world.rank if self.parallel_access else 0

    def close(self):
        """ Check whether the file is open """
        for line in self.lines:
            line.close()

    def create_hdf5(self, directory, **kwargs):
        """Create HDF5 files based on the data

        Parameters
        ----------
        data : geobipy.Data or geobipy.DataPoint
            Data to create the HDF5 file(s) for
        kwargs : geobipy.userParameters
            Input parameters for geobipy

        Returns
        -------
        out : list of H5py.File
            HDF5 files

        """
        if self.parallel_access:
            from mpi4py import MPI

            # Split off a single core communicator.
            single_rank_comm = self.world.Create(self.world.Get_group().Incl([0]))

            if (single_rank_comm != MPI.COMM_NULL):
                # Instantiate a new blank inference3d linked to the head rank
                inference3d = Inference3D(self.data, prng=Generator(PCG64DXSM()), world=single_rank_comm)
                # Create the hdf5 files
                inference3d._create_HDF5_dataset(directory, **kwargs)

            self.world.barrier()

        else:
            self._create_HDF5_dataset(directory, **kwargs)

        self.open(directory, mode='r+', **kwargs)

    # def _create_hdf5(self, **kwargs):

    #     # if isinstance(data, Data):
    #     return self._createHDF5_dataset(**kwargs)
    #     # else:
    #     #     return self._createHDF5_datapoint(data, **kwargs)

    def _create_HDF5_dataset(self, directory, **kwargs):

        # Get a datapoint from the file.
        datapoint = self.data._read_record(record=0)

        # While preparing the file, we need access to the line numbers and fiducials in the data file
        kwargs['interactive_plot'] = False
        inference1d = Inference1D(prng=self.prng, **kwargs)

        inference1d.initialize(datapoint=datapoint)

        self.print('Creating HDF5 files, this may take a few minutes...')
        self.print('Files are being created for data files {} and system files {}'.format(kwargs['data_filename'], kwargs['system_filename']))

        # No need to create and close the files like in parallel, so create and keep them open
        for line in self.line_number:

            subset = self.data.line(line)

            kwargs = {}
            if self.parallel_access:
                kwargs['driver'] = 'mpio'
                kwargs['comm'] = self.world

            with h5py.File(join(directory, '{}.h5'.format(line)), 'w', **kwargs) as f:
                Inference2D(subset, prng=self.prng).createHdf(f, inference1d)

            self.print('Created hdf5 file for line {} with {} data points'.format(line, subset.nPoints))
        self.print('Created hdf5 files {} total data points'.format(self.data.nPoints))

        if self.parallel_access:
            self.world.barrier()


    def print(self, *args):
        if self.world is None:
            print(*args)
        else:
            if self.rank == 0:
                print(*args, flush=True)

    # @property
    # def time(self):
    #     return time.time is self.world is None else MPI.Wtime

    # @property
    # def h5files(self):
    #     """ Get the list of line result files for the dataset """
    #     return self._h5files

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
        index = self.line_index(line_number=line_number)
        return self.lines[index]

    @property
    def lines(self):
        if self.parallel_access:
            start, chunk = loadBalance1D_shrinkingArrays(self.nLines, self.world.size)
            return self._lines[start[self.rank] : start[self.rank]+chunk[self.rank]]
            # else:
            #     return self._lines
        else:
            return self._lines

    @cached_property
    def line_number(self):
        return unique(self.data.line_number)

    @property
    def nLines(self):
        return size(self._lines)

    def _get(self, variable, reciprocateParameter=False, **kwargs):

        variable = variable.lower()
        assert variable in ['mean', 'best', 'interfaces', 'opacity', 'highest_marginal', 'marginal_probability'], ValueError("variable must be ['mean', 'best', 'interfaces', 'opacity', 'highestMarginal', 'marginalProbability']")

        if variable == 'mean':
            if reciprocateParameter:
                vals = divide(1.0, self.meanParameters)
                vals.name = 'Resistivity'
                vals.units = r'$\Omega m$'
                return vals
            else:
                return self.meanParameters

        elif variable == 'best':
            if reciprocateParameter:
                vals = 1.0 / self.meanParameters
                vals.name = 'Resistivity'
                vals.units = r'$\Omega m$'
                return vals
            else:
                return self.bestParameters

        if variable == 'interfaces':
            return self.interfaces

        if variable == 'opacity':
            return self.opacity

        if variable == 'highest_marginal':
            return self.highest_marginal

        if variable == 'marginal_probability':
            assert 'index' in kwargs, ValueError('Please specify keyword "index" when requesting marginalProbability')
            return self.marginalProbability[:, :, kwargs["index"]].T

    def additiveError(self, slic=None):
        op = vstack if self.nSystems > 1 else hstack
        out = StatArray(op([line.additiveError for line in self.lines]), name=self.lines[0].additiveError.name, units=self.lines[0].additiveError.units)
        for line in self.lines:
            line.uncache('additiveError')
        return out

    def infer(self, index=None, fiducial=None, line_number=None, **options):

        if self.parallel_access:
            self.infer_mpi(**options)
        else:
            self.infer_serial(index=index, fiducial=fiducial, line_number=line_number, **options)

    def infer_serial(self, index=None, fiducial=None, line_number=None, **options):

        t0 = time.time()
        self.data = self.data._initialize_sequential_reading(options['data_filename'], options['system_filename'])

        nPoints = self.data.nPoints
        r = range(nPoints)
        if index is None:
            if fiducial is not None:

                tmp = (self.data.fiducial == fiducial)

                if unique(self.data.line_number).size > 1:
                    tmp = tmp & (self.data.line_number == line_number)

                index = squeeze(argwhere(tmp))

                nPoints = 1
                r = range(index, index+1)
        else:
            nPoints = 1
            r = range(index, index+1)

        for i in r:
            rec = i if nPoints == 1 else None
            datapoint = self.data._read_record(record = i)

            # Pass through the line results file object if a parallel file system is in use.
            iLine = self.line_number.searchsorted(datapoint.line_number)[0]

            inference = Inference1D(prng=self.prng, **options)

            inference.initialize(datapoint)

            file_handle = None
            if options['save_hdf5']:
                file_handle = self.lines[iLine].hdf_file
            inference.infer(hdf_file_handle=file_handle)

            e = time.time() - t0
            elapsed = str(timedelta(seconds=e))

            eta = str(timedelta(seconds=(float64(nPoints) / float64(i+1)) * e))
            print("Remaining Points {}/{} || Elapsed Time: {} h:m:s || ETA {} h:m:s".format(nPoints-i-1, nPoints, elapsed, eta))

        self.data.close()


    def infer_mpi(self, **options):

        from mpi4py import MPI
        from ..base import MPI as myMPI

        world = self.world

        t0 = MPI.Wtime()

        # Carryout the head-worker tasks
        if (world.rank == 0):
            self._infer_mpi_master_task(**options)
        else:
            self._infer_mpi_worker_task(**options)

    def _infer_mpi_master_task(self, **options):
        """ Define a Send Recv Send procedure on the head rank """

        from mpi4py import MPI
        from ..base import MPI as myMPI

        # Prep the data for point by point reading
        self.data = self.data._initialize_sequential_reading(options['data_filename'], options['system_filename'])

        # Set the total number of data points
        nPoints = self.data.nPoints

        nFinished = 0
        nSent = 0

        world = self.world
        # Send out the first indices to the workers
        for iWorker in range(1, world.size):
            # Get a datapoint from the file.
            datapoint = self.data._read_record(nSent, mpi_enabled=True)

            # If DataPoint is None, then we reached the end of the file and no more points can be read in.
            if datapoint is None:
                # Send the kill switch to the worker to shut down.
                world.send(False, dest=iWorker)
            else:
                world.send(True, dest=iWorker)
                datapoint.Isend(dest=iWorker, world=world)

            nSent += 1

        # Start a timer
        t0 = MPI.Wtime()

        myMPI.print("Initial data points sent. Head rank is now waiting for requests")

        # Now wait to send indices out to the workers as they finish until the entire data set is finished
        while nFinished < nPoints:
            # Wait for a worker to request the next data point
            status = MPI.Status()
            dummy = world.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
            requestingRank = status.Get_source()

            nFinished += 1

            # Read the next data point from the file
            if nSent == nPoints:
                datapoint = None
            else:
                datapoint = self.data._read_record(nSent, mpi_enabled=True)

            # If DataPoint is None, then we reached the end of the file and no more points can be read in.
            if datapoint is None:
                # Send the kill switch to the worker to shut down.
                world.send(False, dest=requestingRank)
            else:
                world.send(True, dest=requestingRank)
                datapoint.Isend(dest=requestingRank, world=world, system=datapoint.system)

                nSent += 1

            report = ((nFinished % (world.size - 1)) == 0) or (nFinished >= nPoints)

            if report:
                e = MPI.Wtime() - t0
                elapsed = str(timedelta(seconds=e))
                eta = str(timedelta(seconds=(nPoints / nFinished-1) * e))
                myMPI.print("Points sent {} || Remaining {}/{} || Elapsed Time: {} h:m:s || ETA {} h:m:s".format(nSent, nPoints-nFinished, nPoints, elapsed, eta))

    def _infer_mpi_worker_task(self, **options):
        """ Define a wait run ping procedure for each worker """

        # Import here so serial code still works...
        from ..base import MPI as myMPI

        line_number = self.line_number
        Inference2D = self._lines
        world = self.world

        # Initialize the worker process to go
        Go = True

        # Wait till you are told what to process next
        continueRunning = world.recv(source=0)
        # If we continue running, receive the next DataPoint. Otherwise, shutdown the rank
        if continueRunning:
            datapoint = self.data.single.Irecv(source=0, world=world)
        else:
            Go = False

        while Go:
            # # initialize the parameters
            # paras = user_parameters.userParameters(datapoint)
            # # Check the user input parameters against the datapoint
            # paras.check(datapoint)

            # Pass through the line results file object if a parallel file system is in use.
            iLine = line_number.searchsorted(datapoint.line_number)[0]

            inference = Inference1D(prng=self.prng, world=self.world, **options)
            inference.initialize(datapoint)

            failed = inference.infer(hdf_file_handle=self._lines[iLine].hdf_file)

            if failed and inference.datapoint.n_active_channels > 0:
                myMPI.print(f"datapoint --line={datapoint.line_number.item()} --fiducial={datapoint.fiducial.item()} failed to converge")

            # Ping the head rank to request a new index
            world.send('requesting', dest=0)

            # Wait till you are told whether to continue or not
            continueRunning = world.recv(source=0)

            # If we continue running, receive the next DataPoint. Otherwise, shutdown the rank
            if continueRunning:
                datapoint = self.data.single.Irecv(source=0, world=world, system=datapoint.system)
            else:
                Go = False

    # @cached_property
    # def additiveError(self):

    #     additiveError = StatArray((self.nSystems, self.nPoints), name=self.lines[0].additiveError.name, units=self.lines[0].additiveError.units, order = 'F')

    #     print("Reading Additive Errors Posteriors", flush=True)
    #     bar = self.loop_over(self.nLines)

    #     for i in bar:
    #         additiveError[:, self.lineIndices[i]] = self.lines[i].additiveError.T
    #         self.lines[i].uncache('additiveError')

    #     return additiveError


    @cached_property
    def bestData(self):

        bestData = self.lines[0].bestData
        lines[0].uncache('bestData')

        print("Reading Most Probable Data", flush=True)
        bar = self.loop_over(self.nLines)

        for i in bar:
            bestData = bestData + self.lines[i].bestData
            lines[i].uncache('bestData')

        return bestData

    def bestParameters(self, slic=None):
        return StatArray(vstack([line.bestParameters(slic) for line in self.lines]), name=self.lines[0].parameterName, units=self.lines[0].parameterUnits)

    def load_marginal_probability(self, filename):

        with h5py.File(filename, 'r') as f:
            values = StatArray.fromHdf(f['probabilities'])

        return values
            # for i in range(self.nLines):
            #     self.lines[i].marginal_probability = StatArray.fromHdf(f['probabilities'], index=s_[self.lineIndices[i], :, :])
            #     self.lines[i].uncache('highest_marginal')

        # self.uncache('marginalProbability')
        # self.uncache('highest_marginal')
        # return self.marginalProbability

    @cached_property
    def marginalProbability(self):

        mp = self.lines[0].marginal_probability()
        marginalProbability = StatArray((self.nPoints, self.zGrid.nCells.item(), mp.shape[-1]), name=mp.name, units=mp.units)
        marginalProbability[self.lineIndices[0], :, :] = mp

        print('Reading marginal probability', flush=True)
        bar = self.loop_over(1, self.nLines)

        for i in bar:
            marginalProbability[self.lineIndices[i], :, :] = self.lines[i].marginal_probability()
            self.lines[i].uncache('marginalProbability')

        self.uncache('highest_marginal')

        return marginalProbability

    def xy_slice(self, x, y, n_test_points, distance_cutoff, variable):

        import pandas as pd
        self.pointcloud.setKdTree(nDims=2)

        x_grid = linspace(x[0], x[1], n_test_points)
        y_grid = linspace(y[0], y[1], n_test_points)

        query = vstack([x_grid, y_grid]).T

        r, i = self.pointcloud.nearest(query)
        j = squeeze(argwhere(r < distance_cutoff))
        i = i[j]
        i = pd.unique(i)

        values = self._get(variable)[:, i]
        mesh2d = RectilinearMesh2D(x_centres=self.x[i], y_centres=self.y[i], z_edges=self.zGrid.edges, heightCentres=self.height[i])
        slic = Model(mesh2d, values=values)

        return slic

    def compute_credible_interval(self, percent=95.0, log=None, progress=False):

        # Need to create HDF memory collectively.
        for line in self.lines:
            key = 'credible_lower'
            if not key in line.hdf_file.keys():
                credibleLower = StatArray(zeros(line.mesh.shape), '{}% Credible Interval'.format(100.0 - percent), line.parameterUnits)
                credibleLower.createHdf(line.hdf_file, key)

            key = 'credible_upper'
            if not key in line.hdf_file.keys():
                credibleUpper = StatArray(zeros(line.mesh.shape), '{}% Credible Interval'.format(percent), line.parameterUnits)
                credibleUpper.createHdf(line.hdf_file, key)

        if self.parallel_access:
            self.world.barrier()

        r = self.loop_over(self.line_starts[self.rank], self.line_ends[self.rank])

        for i in r:
            self.lines[i].computeCredibleInterval(percent, log)

    def compute_doi(self, *args, **kwargs):
        """Compute the depth of investigation.
        """
        for line in self.lines:
            if not 'doi' in line.hdf_file:
                doi = StatArray(line.nPoints, 'Depth of investigation', line.height.units)
                doi.createHdf(line.hdf_file, 'doi')

        if self.parallel_access:
            self.world.barrier()
            kwargs['track'] = False

        r = self.loop_over(self.line_starts[self.rank], self.line_ends[self.rank])
        for i in r:
            self.lines[i].compute_doi(*args, **kwargs)


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
            r = range(starts[self.rank], starts[self.rank] + chunks[self.rank])

            if self.rank == 0:
                Bar = progressbar.ProgressBar()
                r = Bar(r)

        else:

            local_mixture_h5 = h5py.File(local_mixture_hdf5, 'r')
            probabilities_h5 = h5py.File('P_class.h5', 'w')

            Bar = progressbar.ProgressBar()
            r = Bar(range(self.nPoints))



        # Create the space in HDF5
        probabilities = StatArray((z.nCells.value, global_mixture.n_components), name='probabilities')
        probabilities.createHdf(probabilities_h5, 'probabilities', nRepeats=self.nPoints)

        for i in r:
            # Read the local fits
            local_fits = []
            for j in range(z.nCells.value):
                local_fits.append(mixPearson.fromHdf(local_mixture_h5['fits'], index=(i, j)))

            # Get the 2D posterior for parameter
            posterior = self.hitmap(index=i)

            # Compute cluster probabilities
            probabilities = posterior.compute_MinsleyFoksBedrosian2020_P_lithology(global_mixture=global_mixture, local_mixture=local_fits, log=log)

            # Write the probabilities to file
            probabilities.writeHdf(probabilities_h5, 'probabilities', index=i)


        local_mixture_h5.close()
        probabilities_h5.close()

    def compute_probability(self, distribution, log=None, log_probability=False, axis=1, **kwargs):

        if self.parallel_access:
            filename = kwargs['filename']

            hdf_file = h5py.File(filename, 'w', driver='mpio', comm=self.world)

            StatArray().createHdf(hdf_file, 'probabilities', shape=(self.nPoints, distribution.ndim, self.lines[0].mesh.y.nCells), fillvalue=nan)

            r = range(self.line_starts[self.rank], self.line_ends[self.rank])
            self.world.barrier()

            if self.rank == 0:
                Bar = progressbar.ProgressBar()
                r = Bar(r)

            for i in r:
                p_local = self.lines[i].compute_probability(distribution, log=log, log_probability=log_probability, axis=axis, track=False, **kwargs)

                p_local.writeHdf(hdf_file, 'probabilities', index=(self.lineIndices[i], s_[:], s_[:]))

            self.world.barrier()
            hdf_file.close()

        else:
            return StatArray(vstack([line.compute_probability(distribution, log=log, log_probability=log_probability, axis=axis, save=True, **kwargs) for line in self.lines]))

    def cluster_fits_gmm(self, n_clusters, plot=False):

        std = std(self.fits[2], axis=0)
        whitened = (self.fits[2] / std).reshape(-1, 1)

        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(whitened)
        gmm.means_ *= std

        order = argsort(gmm.means_[:, 0])
        weights = gmm.weights_[order]
        means = gmm.means_[order, :]
        covariances = gmm.covariances_[order, :, :]

        cF.set_gmm(weights, means, covariances)
        cF.save_gmm(gmm, "gmm_{}_clusters.h5".format(gmm.n_components))

        if plot:
            bins = StatArray(linspace(self.fits[2].min(), self.fits[2].max(), 200))
            binCentres = bins.internalEdges()
            x_predict = binCentres
            x_predict = x_predict.reshape(-1, 1)

            logprob = gmm.score_samples(x_predict)
            responsibilities = gmm.predict_proba(x_predict)
            pdf = exp(logprob)
            pdf_individual = responsibilities * pdf[:, newaxis]

            h = Histogram1D(edges = bins)
            h.update(self.fits[2])

            h._counts = h._counts / max(h._counts)
            h.plot(alpha=0.4, linewidth=0)

            for i in range(gmm.n_components):
                plt.plot(binCentres, pdf_individual[:, i], '--k', linewidth=1)

        return gmm


    # def cluster_fits_smm(self, n_clusters, plot=False, **kwargs):

    #     std = std(self.fits[2], axis=0)
    #     whitened = (self.fits[2] / std).reshape(-1, 1)

    #     model = SMM(n_components=n_clusters, **kwargs).fit(whitened)
    #     model.means_ *= std

    #     order = argsort(model.means_[:, 0])
    #     model.weights_ = model.weights_[order]
    #     model.means_ = model.means_[order, :]
    #     if model.covariance_type == 'diag':
    #         model.covars_ = model.covars_[order, :]
    #     else:
    #         model.covars_ = model.covariances[order, :, :]

    #     model.degrees_ = model.degrees[order]

    #     if plot:
    #         bins = StatArray(linspace(self.fits[2].min(), self.fits[2].max(), 200))
    #         binCentres = bins.internalEdges()
    #         x_predict = binCentres
    #         x_predict = x_predict.reshape(-1, 1)

    #         pdf, responsibilities = model.score_samples(x_predict)
    #         pdf_individual = responsibilities * pdf[:, newaxis]

    #         h = Histogram1D(edges = bins)
    #         h.update(self.fits[2])

    #         h._counts = h._counts / max(h._counts)
    #         h.plot(alpha=0.4, linewidth=0)

    #         for i in range(model.n_components):
    #             plt.plot(binCentres, pdf_individual[:, i], '--k', linewidth=1)

    #     return model

    @cached_property
    def data_misfit(self):
        return self.bestData.data_misfit()

    @cached_property
    def doi(self):
        return hstack([line.doi for line in self.lines])

    def compute_doi(self, *args, **kwargs):

        if self.parallel_access:
            self.close()
            self.world.barrier()
            r = self.loop_over(self.line_starts[self.world.rank], self.line_ends[self.worlf.rank])
            for i in r:
                line = self.lines[i]
                line.open()
                line.compute_doi(*args, **kwargs)

            self.open(world=self.world)
        else:
            for line in self.lines:
                line.compute_doi(*args, **kwargs)

    def parameter_posterior(self, fiducial=None, index=None):
        return self.hitmap(fiducial, index)

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
        iLine, index = self.line_index(fiducial=fiducial, index=index)
        return self.lines[iLine].parameter_posterior(index=index)


    @property
    def hitmapCounts(self):
        if (self._counts is None):
            mesh = self.lines[0].mesh
            self._counts = empty([])


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
            out = vstack([out, line.identifyPeaks(depths, nBins, width, limits)])

        return out

    @cached_property
    def interface_probability(self):
        interfaces = StatArray((self.nPoints, self.zGrid.nCells.item()), name='P(interface)')

        print("Reading Depth Posteriors", flush=True)
        Bar=progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            interfaces[self.lineIndices[i], :] = self.lines[i].interface_probability().values
            self.lines[i].uncache('interface_probability')

        return interfaces

    @property
    def lineIndices(self):

        lineIndices = []
        i0 = 0
        for i in range(self.nLines):
            i1 = i0 + self.lines[i].nPoints
            lineIndices.append(s_[i0:i1])
            i0 = i1

        return lineIndices

    def meanParameters(self, slic=None):
        return StatArray(vstack([line.mean_parameters(slic) for line in self.lines]), name=self.lines[0].parameterName, units=self.lines[0].parameterUnits)

    def mesh2d(self, dx, dy, **kwargs):
        return self.pointcloud.centred_mesh(dx, dy, **kwargs)

    def mesh3d(self, dx, dy, **kwargs):
        """Generate a 3D mesh using dx, dy, and the apriori discretized vertical dimension before inversion.

        Parameters
        ----------
        dx : float
            Increment in x.
        dy : float
            Increment in y.

        Returns
        -------
        geoobipy.RectilinearMesh3D : 3D rectilinear mesh with a draped top surface.
        """
        kwargs['mask'] = None
        mesh = self.mesh2d(dx, dy)
        # Interpolate the draped surface of the mesh
        height, dum = self.pointcloud.interpolate(mesh=mesh, values=self.pointcloud.elevation, block=True, **kwargs)
        return RectilinearMesh3D(x_edges=height.x.edges, y_edges=height.y.edges, z_edges=self.zGrid.edges, height=height.values)

    @cached_property
    def nActive(self):
        nActive = empty(self.nPoint, dtype=int32)
        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            nActive[self.lineIndices[i]] = self.lines[i].bestData.nActiveChannels
            self.lines[i].uncache('bestData')

        return nActive

    @property
    def nPoints(self):
        """ Get the total number of data points """
        tmp = asarray([line.nPoints for line in self.lines])
        self._cumNpoints = cumsum(tmp)
        return sum(tmp)

    @property
    def nSystems(self):
        """ Get the number of systems """
        return self.lines[0].nSystems

    @cached_property
    def opacity(self):

        opacity = StatArray((self.zGrid.nCells.item(), self.nPoints), order = 'F')

        print("Reading opacity", flush=True)
        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            opacity[:, self.lineIndices[i]] = self.lines[i].opacity
            self.lines[i].uncache('opacity')

        return opacity


    def parameterHistogram(self, nBins, depth = None, depth2 = None, log=None):
        """ Compute a histogram of all the parameter values, optionally show the histogram for given depth ranges instead """

        out = self.lines[0].parameterHistogram(nBins=nBins, depth=depth, depth2=depth2, log=log)

        for line in self.lines[1:]:
            tmp = line.parameterHistogram(nBins=nBins, depth=depth, depth2=depth2, log=log)
            out._counts += tmp.counts

        return out


    @cached_property
    def pointcloud(self):

        x = StatArray(self.nPoints, name=self.lines[0].x.name, units=self.lines[0].x.units)
        y = StatArray(self.nPoints, name=self.lines[0].y.name, units=self.lines[0].y.units)
        z = StatArray(self.nPoints, name=self.lines[0].height.name, units=self.lines[0].height.units)
        e = StatArray(self.nPoints, name=self.lines[0].elevation.name, units=self.lines[0].elevation.units)
        # Loop over the lines in the data set and get the attributes
        print('Reading co-ordinates', flush=True)
        bar = self.loop_over(self.nLines)

        for i in bar:
            indices = self.lineIndices[i]
            x[indices] = self.lines[i].x
            y[indices] = self.lines[i].y
            z[indices] = self.lines[i].height
            e[indices] = self.lines[i].elevation

            self.lines[i].uncache(['x', 'y', 'height', 'elevation'])

        return Point(x, y, z, e)

    def read_fit_distributions(self, fit_file, mask_by_doi=False, skip=None, components='mve', mean_limits=None, flatten=True):
        if skip is None:
            s = s_[:]
            skip = 1
        else:
            s = s_[::skip]

        with h5py.File(fit_file, 'r') as f:
            amp_3D = asarray(f['/fits/params/data'][s, :, 0::4])
            if 'm' in components:
                mean_3D = asarray(f['/fits/params/data'][s, :, 1::4])
            if 'v' in components:
                var_3D = asarray(f['/fits/params/data'][s, :, 2::4])**2.0
            if 'e' in components:
                exp_3D = asarray(f['/fits/params/data'][s, :, 3::4])

        assert amp_3D.shape[0] == self.nPoints, Exception("fit file {} has {} fits, but the dataset has {} points".format(fit_file, amp_3D.shape[0], self.nPoints))

        z = self.zGrid.centres
        # d2D = repeat(d1D[None, :], mean_3D.shape[0], axis=0)
        z3D = repeat(repeat(z[None, :], mean_3D.shape[0], axis=0)[:, :, None], mean_3D.shape[2], axis=2)

        if not mean_limits is None:
            if 'm' in components:
                amp_3D[mean_3D < mean_limits[0]] = 0.0
                amp_3D[mean_3D > mean_limits[1]] = 0.0

        if mask_by_doi:
            indices = z.searchsorted(self.doi[s])

            for i in range(amp_3D.shape[0]):
                amp_3D[i, indices[i]:, :] = 0.0

        if flatten:
            # Mask out nulls
            i0, i1, i2 = amp_3D.nonzero()

            depth = StatArray(z3D[i0, i1, i2].flatten(), 'Depth')
            amplitudes = StatArray(amp_3D[i0, i1, i2].flatten(), 'Amplitude')
            means = None
            variances = None
            exponents = None
            if 'm' in components:
                means = StatArray(mean_3D[i0, i1, i2].flatten(), 'Mean')
            if 'v' in components:
                variances = StatArray(var_3D[i0, i1, i2].flatten(), 'Variance')
            if 'e' in components:
                exponents = StatArray(exp_3D[i0, i1, i2].flatten(), 'Exponent')
        else:
            depth = z3D
            amplitudes = amp_3D
            means = mean_3D
            variances = var_3D
            exponents = exp_3D

        self.fits = {
            'depth' : depth,
            'amplitude' : amplitudes,
            'mean' : means,
            'variance' : variances,
            'exponent' : exponents,
        }
        return self.fits

    def relativeError(self):
        op = vstack if self.nSystems > 1 else hstack
        out = StatArray(op([line.relativeError for line in self.lines]), name=self.lines[0].relativeError.name, units=self.lines[0].relativeError.units)
        for line in self.lines:
            line.uncache('relativeError')
        return out

    @property
    def x(self):
        return self.pointcloud.x

    @property
    def y(self):
        return self.pointcloud.y


    def inference_1d(self, fiducial=None, index=None, line_index=None):
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
        line_index, fidIndex = self.line_index(fiducial=fiducial, index=index)
        if size(fidIndex) > 1:
            assert line_index is not None, ValueError("Multiple fiducials found, please specify which line_index out of {}".format(line_index))
            line_index = line_index
        return self.lines[line_index].inference_1d(fidIndex)

    def line_index(self, line_number=None, fiducial=None, index=None):
        """Get the line index """
        tmp = sum([not x is None for x in [line_number, fiducial, index]])
        assert tmp == 1, Exception("Please specify one argument, line_number, fiducial, or index")

        index = atleast_1d(index)

        if line_number is not None:
            assert line_number in self.line_number, ValueError("line {} not found in data set".format(line_number))
            return squeeze(where(self.line_number == line_number)[0])

        if fiducial is not None:
            return squeeze(self.fiducialIndex(fiducial))

        assert npall(index <= self.nPoints-1), IndexError('index {} is out of bounds for data point index with size {}'.format(index, self.nPoints))

        cumPoints = self._cumNpoints - 1

        iLine = cumPoints.searchsorted(index)
        i = squeeze(where(iLine > 0))
        index[i] -= self._cumNpoints[iLine[i]-1]

        return squeeze(iLine), squeeze(index)

    def progress_bar(self, iterable, *args, **kwargs):
        """Generate a loop range.

        Tracks progress on the head rank only if parallel.

        Parameters
        ----------
        value : int
            Size of the loop to generate
        """
        if isinstance(iterable, (int, integer)):
            this = range(iterable, *args, **kwargs)
        else:
            this = iterable

        bar = progressbar.ProgressBar()

        if self.parallel_access:
            if self.rank == 0:
                return bar(this)
            else:
                return this
        else:
            return bar(this)

    @property
    def fiducials(self):
        return StatArray(hstack([line.fiducials for line in self.lines]), name='fiducials')

    def fiducial(self, index):
        """ Get the fiducial of the given data point """
        iLine, index = self.line_index(index=index)
        iLine = atleast_1d(iLine)
        index = atleast_1d(index)

        out = empty(size(index))
        for i in range(size(index)):
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
        line_index : ints
            line_index for each fiducial
        index : ints
            Index of each fiducial in their respective line

        """

        line_index = []
        index = []

        for i, line in enumerate(self.lines):
            ids = line.fiducialIndex(fiducial)
            nIds = size(ids)
            if nIds > 0:
                line_index.append(full(nIds, fill_value=i))
                index.append(ids)

        if size(index) > 0:
            return squeeze(hstack(line_index)), squeeze(hstack(index))

        assert False, ValueError("fiducial not present in this data set")

    def fit_mixture_to_pdf(self, intervals=None, **kwargs):

        if self.parallel_access:
            return self.fit_mixture_to_pdf_mpi(intervals, **kwargs)
        else:
            return self.fit_mixture_to_pdf_serial(intervals, **kwargs)

    def fit_mixture_to_pdf_mpi(self, intervals=None, **kwargs):

        from mpi4py import MPI
        from geobipy.src.base import MPI as myMPI

        rank = self.rank

        kwargs['max_distributions'] = kwargs.get('max_distributions', 3)
        kwargs['track'] = False

        hdf_file = h5py.File("fits.h5", 'w', driver='mpio', comm=self.world)

        a = zeros(kwargs['max_distributions'])
        mixture = mixPearson(a, a, a, a)
        mixture.createHdf(hdf_file, 'fits', add_axis=(self.nPoints, self.lines[0].mesh.y.nCells))

        if rank == 0:  ## Head rank
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

            myMPI.print("Initial posteriors sent. Head rank is now waiting for requests")

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
                hm = self.parameter_posterior(index=index)

                if not npall(hm.counts == 0):

                    try:
                        mixtures = hm.fit_mixture_to_pdf(mixture=mixPearson, **kwargs)

                    except:
                        print('rank {} point {} failed'.format(rank, index), flush=True)
                        mixtures = None

                    if mixtures is not None:
                        for j, m in enumerate(mixtures):
                            if not m is None:
                                m.writeHdf(hdf_file, 'fits', index=(index, j))

                self.world.send(1, dest=0)
                # Wait till you are told whether to continue or not
                continueRunning = self.world.recv(source=0)

                # If we continue running, receive the next DataPoint. Otherwise, shutdown the rank
                if continueRunning:
                    index = self.world.recv(source=0)
                else:
                    Go = False

    def fit_mixture_to_pdf_serial(self, intervals, **kwargs):
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

        hm = self.parameter_posterior(index=0)
        return hm.fit_mixture_to_pdf(**kwargs)

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
    #     nIntervals = size(intervals) - 1

    #     tmp = locals()
    #     for key in ['self', 'MPI', 'world', 'k']:
    #         tmp.pop(key, None)
    #     command = str(tmp)

    #     for i in range(self.nLines):

    #         means = StatArray((self.lines[i].nPoints, nIntervals, maxClusters), "fit means")
    #         variances = StatArray((self.lines[i].nPoints, nIntervals, maxClusters), "fit variances")

    #         if 'mixture_fits' in self.lines[i].hdf_file:
    #             saved_command = self.lines[i].hdf_file['/mixture_fits'].attrs['command']
    #             if command != saved_command:
    #                 del self.lines[i].hdf_file['/mixture_fits']

    #                 means.createHdf(self.lines[i].hdf_file, "/mixture_fits/means")
    #                 variances.createHdf(self.lines[i].hdf_file, "/mixture_fits/variances")

    #                 print('writing \n', command)
    #                 self.lines[i].hdf_file['/mixture_fits'].attrs['command'] = command

    #     # Distribute the points amongst cores.
    #     starts, chunks = loadBalance1D_shrinkingArrays(self.nPoints, self.world.size)

    #     chunk = chunks[self.rank]
    #     chunk = 10
    #     i0 = starts[self.rank]
    #     i1 = i0 + chunk

    #     iLine, index = self.line_index(index=arange(i0, i1))

    #     tBase = MPI.Wtime()
    #     t0 = tBase

    #     nUpdate = int(0.1 * chunk)
    #     counter = 0

    #     for i in range(chunk):

    #         iL = iLine[i]
    #         ind = index[i]

    #         line = self.lines[iL]

    #         hm = line.get_hitmap(ind)

    #         d, a = hm.fit_mixture(intervals, track=False, **kwargs)

    #         for j in range(nIntervals):
    #             dm = squeeze(d[j].means_[a[j]])
    #             dv = squeeze(d[j].covariances_[a[j]])

    #             nD = size(dm)

    #             line.hdf_file['/mixture_fits/means/data'][ind, j, :nD] = dm
    #             line.hdf_file['/mixture_fits/variances/data'][ind, j, :nD] = dv

    #         counter += 1
    #         if counter == nUpdate:
    #             print('rank {}, line/fiducial {}/{}, iteration {}/{},  time/dp {} h:m:s'.format(world.rank, self.line_number[iL], line.fiducials[ind], i+1, chunk, str(timedelta(seconds=MPI.Wtime()-t0)/nUpdate)), flush=True)
    #             t0 = MPI.Wtime()
    #             counter = 0

    #     print('rank {} finished in {} h:m:s'.format(world.rank, str(timedelta(seconds=MPI.Wtime()-tBase))))

    @cached_property
    def highest_marginal(self):
        return StatArray(argmax(self.marginalProbability, axis=-1), name='Highest marginal').T

    def histogram(self, nBins, **kwargs):
        """ Compute a histogram of the model, optionally show the histogram for given depth ranges instead """

        log = kwargs.pop('log', False)

        values = self._get(**kwargs)

        if (log):
            values, logLabel = cF._log(values, log)
            values.name = logLabel + values.name

        values = StatArray(values, values.name, values.units)

        h = Histogram1D(linspace(nanmin(values), nanmax(values), nBins))
        h.update(values)

        h.plot()
        return h


    @property
    def zGrid(self):
        """ Gets the discretization in depth """
        return self.lines[0].mesh.y

    def _z_slice(self, depth=None):
        return self.lines[0]._z_slice(depth)


    # def getMean3D(self, dx, dy, mask = False, clip = False, force=False, method='ct'):
    #     """ Interpolate each depth slice to create a 3D volume """
    #     if (not self.mean3D is None and not force): return

    #     # Test for an existing file, created with the same parameters.
    #     # Read it and return if it exists.
    #     file = 'mean3D.h5'
    #     if fileIO.fileExists(file):
    #         variables = hdfRead.read_all(file)
    #         if (dx == variables['dx'] and dy == variables['dy'] and mask == variables['mask'] and clip == variables['clip'] and method == variables['method']):
    #             self.mean3D = variables['mean3d']
    #             return


    #     method = method.lower()
    #     if method == 'ct':
    #         self.__getMean3D_CloughTocher(dx=dx, dy=dy, mask=mask, clip=clip, force=force)
    #     elif method == 'mc':
    #         self.__getMean3D_minimumCurvature(dx=dx, dy=dy, mask=mask, clip=clip)
    #     else:
    #         assert False, ValueError("method must be either 'ct' or 'mc' ")

    #     with h5py.File('mean3D.h5','w') as f:
    #         f.create_dataset(name = 'dx', data = dx)
    #         f.create_dataset(name = 'dy', data = dy)
    #         f.create_dataset(name = 'mask', data = mask)
    #         f.create_dataset(name = 'clip', data = clip)
    #         f.create_dataset(name = 'method', data = method)
    #         self.mean3D.toHdf(f,'mean3d')


    # def highest_marginal_3D(self, dx, dy, mask=None, clip=False):


    #     x = self.pointcloud.x.deepcopy()
    #     y = self.pointcloud.y.deepcopy()

    #     values = self.meanParameters[0, :]
    #     x1, y1, vals = interpolation.minimumCurvature(x, y, values, self.pointcloud.bounds, dx=dx, dy=dy, mask=mask, clip=clip, iterations=2000, tension=0.25, accuracy=0.01)

    #     # Initialize 3D volume
    #     mean3D = StatArray(zeros([self.zGrid.nCells.value, y1.size+1, x1.size+1], order = 'F'),name = 'Conductivity', units = '$Sm^{-1}$')
    #     mean3D[0, :, :] = vals

    #     # Interpolate for each depth
    #     print('Interpolating using minimum curvature')
    #     bar = self.loop_over(1, self.zGrid.nCells.value)
    #     for i in bar:
    #         # Get the model values for the current depth
    #         values = self.meanParameters[i, :]
    #         dum1, dum2, vals = interpolation.minimumCurvature(x, y, values, self.pointcloud.bounds, dx=dx, dy=dy, mask=mask, clip=clip, iterations=2000, tension=0.25, accuracy=0.01)
    #         # Add values to the 3D array
    #         mean3D[i, :, :] = vals

    #     self.mean3D = mean3D


    # def __getMean3D_minimumCurvature(self, dx, dy, mask=None, clip=False):


    #     x = self.pointcloud.x.deepcopy()
    #     y = self.pointcloud.y.deepcopy()

    #     values = self.meanParameters[0, :]
    #     x1, y1, vals = interpolation.minimumCurvature(x, y, values, self.pointcloud.bounds, dx=dx, dy=dy, mask=mask, clip=clip, iterations=2000, tension=0.25, accuracy=0.01)

    #     # Initialize 3D volume
    #     mean3D = StatArray(zeros([self.zGrid.nCells.value, y1.size+1, x1.size+1], order = 'F'),name = 'Conductivity', units = '$Sm^{-1}$')
    #     mean3D[0, :, :] = vals

    #     # Interpolate for each depth
    #     print('Interpolating using minimum curvature')
    #     bar = self.loop_over(1, self.zGrid.nCells.value)
    #     for i in bar:
    #         # Get the model values for the current depth
    #         values = self.meanParameters[i, :]
    #         dum1, dum2, vals = interpolation.minimumCurvature(x, y, values, self.pointcloud.bounds, dx=dx, dy=dy, mask=mask, clip=clip, iterations=2000, tension=0.25, accuracy=0.01)
    #         # Add values to the 3D array
    #         mean3D[i, :, :] = vals

    #     self.mean3D = mean3D

    def interpolate(self, dx, dy, values, method='ct', mask=None, clip=True, i=None, block=True, **kwargs):
        return self.pointcloud.interpolate(mesh=self.mesh2d(dx, dy), values=values, method=method, mask=mask, clip=clip, i=i, block=block, **kwargs)

    def map(self, dx, dy, values, method='ct', mask = None, clip = True, **kwargs):
        """ Create a map of a parameter """

        assert values.size == self.nPoints, ValueError("values must have size {}".format(self.nPoints))

        values, kwargs = self.interpolate(dx=dx, dy=dy, values=values, method=method, mask=mask, clip=clip, **kwargs)

        kwargs.pop('operator', None)
        kwargs.pop('condition', None)
        kwargs.pop('clip_min', None)
        kwargs.pop('clip_max', None)


        if 'alpha' in kwargs:
            alpha, kwargs = self.interpolate(dx=dx, dy=dy, values=kwargs['alpha'], method=method, mask=mask, clip=clip, **kwargs)
            kwargs['alpha'] = alpha.values

        return values.pcolor(**kwargs)


    def mapMarginalProbability(self, dx, dy, depth, index, **kwargs):

        z_slice = self._z_slice(depth)
        self.pointcloud.map(dx = dx, dy = dy, values = self.marginalProbability[:, z_slice, index], **kwargs)

    def percentageParameter(self, value, depth, depth2=None):

        percentage = StatArray(empty(self.nPoints), name="Probability of {} > {:0.2f}".format(self.meanParameters.name, value), units = self.meanParameters.units)

        print('Calculating percentages', flush = True)
        bar = self.loop_over(self.nLines)
        for i in bar:
            percentage[self.lineIndices[i]] = self.lines[i].percentageParameter(value, depth, depth2)

        return percentage

    def depth_slice(self, depth, variable, reciprocateParameter=False, **kwargs):

        out = empty(self.nPoints)

        for i, line in enumerate(self.lines):
            values = line.depth_slice(depth, variable, reciprocateParameter=reciprocateParameter, **kwargs)
            out[self.lineIndices[i]] = values

        return StatArray(out, values.name, values.units)

    def interpolate_3d(self, dx, dy, variable, block=True, **kwargs):

        if self.parallel_access:
            return self._interpolate_3d_mpi(dx, dy, variable, block=block, **kwargs)
        else:
            return self._interpolate_3d(dx, dy, variable, block=block, **kwargs)

    def _interpolate_3d(self, dx, dy, variable, block=True, **kwargs):

        tmp = self.depth_slice(depth=0, variable=variable, **kwargs)
        values, dum = self.interpolate(dx, dy, values=tmp, block=block, **kwargs)

        out = Model(self.mesh3d(dx, dy, **kwargs))
        out.values[0, :, :] = values.values

        r = self.loop_over(1, self.zGrid.nCells.value)

        for i in r:
            tmp = self.depth_slice(depth=i, variable=variable, **kwargs)
            values, dum = self.interpolate(dx, dy, values=tmp, block=block, **kwargs)
            values.values, dum = cF._log(values.values, kwargs.get('log', None))
            out.values[i, :, :] = values.values

        # Save the 3D model
        out.toHdf("{}_{}_{}.h5".format(variable, dx, dy), "{}".format(variable))

        return out

    def _interpolate_3d_mpi(self, dx, dy, variable, **kwargs):

        kwargs['block'] = kwargs.pop('block', True)

        starts, chunks = loadBalance1D_shrinkingArrays(self.zGrid.nCells.item(), self.world.size)
        ends = starts + chunks
        tmp = self.depth_slice(depth=starts[self.rank], variable=variable, **kwargs)
        values, dum = self.interpolate(dx, dy, values=tmp, **kwargs)
        values.values, dum = cF._log(values.values, kwargs.get('log', None))

        out = Model(self.mesh3d(dx, dy, **kwargs))

        f = h5py.File("{}_{}_{}.h5".format(variable, dx, dy), 'w', driver='mpio', comm=self.world)
        grp = out.createHdf(f, "{}".format(variable))

        self.world.barrier()

        values.writeHdf(f, "{}".format(variable), index=starts[self.rank])

        r = self.loop_over(starts[self.rank]+1, ends[self.rank])

        for i in r:
            tmp = self.depth_slice(depth=i, variable=variable, **kwargs)
            values, dum = self.interpolate(dx, dy, values=tmp, **kwargs)
            values.values, dum = cF._log(values.values, kwargs.get('log', None))
            values.writeHdf(f, "{}".format(variable), index=i)

        f.close()

    def interpolate_marginal_3d(self, dx, dy, block=True, **kwargs):

        if self.parallel_access:
            return self._interpolate_marginal_3d_mpi(dx, dy, block=block, **kwargs)
        else:
            return self._interpolate_marginal_3d(dx, dy, block=block, **kwargs)

    def _interpolate_marginal_3d(self, dx, dy, block=True, **kwargs):

        nClusters = self.marginalProbability.shape[-1]

        values, _, _ = self.interpolate_marginal(dx, dy, depth=0, **kwargs)

        f = h5py.File("marginal_probability_3d_{}_{}.h5".format(dx, dy), 'w')

        values.createHdf(f, 'marginal_probability', nRepeats=self.zGrid.nCells.value)
        values.writeHdf(f, 'marginal_probability', index=0)

        r = self.loop_over(1, self.zGrid.nCells.item())

        for i in r:
            values, _, _ = self.interpolate_marginal(dx, dy, depth=i, **kwargs)
            values.writeHdf(f, 'marginal_probability', index=i)

        f.close()

    def _interpolate_marginal_3d_mpi(self, dx, dy, variable, **kwargs):

        kwargs['block'] = kwargs.pop('block', True)

        starts, chunks = loadBalance1D_shrinkingArrays(self.zGrid.nCells.item(), self.world.size)
        ends = starts + chunks

        values, _, _ = self.interpolate_marginal(dx, dy, depth=starts[self.rank], **kwargs)

        f = h5py.File("marginal_probability_3d_{}_{}.h5".format(dx, dy), 'w', driver='mpio', comm=self.world)

        values.createHdf(f, 'marginal_probability', nRepeats=self.zGrid.nCells.value)

        self.world.barrier()

        values.writeHdf(f, 'marginal_probability', index=starts[world.rank])

        r = self.loop_over(starts[self.rank]+1, ends[self.rank])

        for i in r:
            values, _, _ = self.interpolate_marginal(dx, dy, depth=i, **kwargs)
            values.writeHdf(f, 'marginal_probability', index=i)

        f.close()

    def summarize_lines(self, lines=None, **kwargs):

        if lines is None:
            lines = self.lines

        bar = self.progress_bar(lines)

        output = kwargs.pop('output_directory', '.')

        for this in bar:
            plt.close('all')
            try:
                fig = this.plot_summary(**kwargs)
                fig.savefig(f"{output}//{this.line_number}.png")
                plt.close(fig)
            except:
                pass
            # this.close()
            # del this


    def scatter_z_slice_animate(self, variable, filename, **kwargs):

        fig = kwargs.pop('fig', plt.figure(figsize=(9, 9)))

        # Do the first slice
        ax, pc, cb = self.plotdepth_slice(depth=self.zGrid.centres[0], variable=variable, **kwargs)

        kwargs['colorbar'] = False

        def animate(i):
            plt.title('{:.2f} m depth'.format(self.zGrid.centres[i]))
            values = self.depth_slice(depth=self.zGrid.centres[i], variable=variable, **kwargs)
            values, dum = cF._log(values, kwargs.get('log', None))
            pc.set_array(values.flatten())

        anim = FuncAnimation(fig, animate, interval=300, frames=self.zGrid.nCells.value)

        plt.draw()
        anim.save(filename)


    def map_z_slice_animate(self, dx, dy, variable, filename, **kwargs):

        from matplotlib.animation import FuncAnimation

        fig = kwargs.pop('fig', plt.figure(figsize=(9, 9)))

        # Do the first slice
        ax, pc, cb = self.mapdepth_slice(dx, dy, depth=self.zGrid.centres[0], variable=variable, **kwargs)

        kwargs['colorbar'] = False

        def animate(i):
            plt.title('{:.2f} m depth'.format(self.zGrid.centres[i]))
            tmp = self.depth_slice(depth=self.zGrid.centres[i], variable=variable, **kwargs)
            values, _ = self.interpolate(dx, dy, values=tmp, **kwargs)
            values, _ = cF._log(values.values, kwargs.get('log', None))
            pc.set_array(values.flatten())

        anim = FuncAnimation(fig, animate, interval=300, frames=self.zGrid.nCells.value)

        plt.draw()
        anim.save(filename)


    def map_highest_marginal_animate(self, dx, dy, filename, **kwargs):

        from matplotlib.animation import FuncAnimation

        fig = kwargs.pop('fig', plt.figure(figsize=(9, 9)))

        # Do the first slice
        ax, pc, cb = self.map_highest_marginal(dx, dy, depth=self.zGrid.centres[0], **kwargs)

        kwargs['colorbar'] = False

        def animate(i):
            plt.title('{:.2f} m depth'.format(self.zGrid.centres[i]))
            highest, x, y, z, dum = self.interpolate_highest_marginal(dx, dy, depth=self.zGrid.centres[i], **kwargs)
            pc.set_array(highest.flatten())

        anim = FuncAnimation(fig, animate, interval=300, frames=self.zGrid.nCells.value)

        plt.draw()
        anim.save(filename)


    def mapAdditiveError(self,dx, dy, system=0, mask = None, clip = True, useVariance=False, **kwargs):
        """ Create a map of a parameter """

        if useVariance:
            for line in self.lines:
                line.compute_additive_error_opacity()
            alpha = hstack([line.additive_error_opacity for line in self.lines])
            kwargs['alpha'] = alpha

        return self.map(dx = dx, dy = dy, mask = mask, clip = clip, values = self.additiveError[system, :], **kwargs)


    def map_depth_slice(self, dx, dy, depth, variable, method='ct', mask = None, clip = True, reciprocateParameter=False, useVariance=False, index=None, **kwargs):
        """ Create a depth slice through the recovered model """

        vals1D = self.depth_slice(depth=depth, variable=variable, reciprocateParameter=reciprocateParameter, index=index)

        if useVariance:
            alpha = self.depth_slice(depth=depth, variable='opacity')
            kwargs['alpha'] = alpha

        return self.map(dx, dy, vals1D, method=method, mask = mask, clip = clip, **kwargs)

    def mapElevation(self,dx, dy, mask = None, clip = True, **kwargs):
        """ Create a map of a parameter """
        return self.map(dx = dx, dy = dy, mask = mask, clip = clip, values = self.elevation, **kwargs)

    def interpolate_marginal(self, dx, dy, depth, **kwargs):

        nClusters = self.marginalProbability.shape[-1]

        vals1D = self.depth_slice(depth=depth, variable='marginal_probability', index=0, **kwargs)
        values, dum = self.interpolate(dx, dy, vals1D, **kwargs)

        interpolated_marginal = StatArray((*values.shape, nClusters), 'Marginal probability')
        interpolated_marginal[:, :, 0] = values.values

        for i in range(1, nClusters):
            vals1D = self.depth_slice(depth=depth, variable='marginal_probability', index=i, **kwargs)
            values, dum = self.interpolate(dx, dy, vals1D, **kwargs)
            interpolated_marginal[:, :, i] = values.values

        return interpolated_marginal, values, dum

    def interpolate_highest_marginal(self, dx, dy, depth, **kwargs):

        interpolated_marginal, values, dum = self.interpolate_marginal(dx, dy, depth, **kwargs)

        highest = StatArray((argmax(interpolated_marginal, axis=-1)).astype(float32))
        msk = npall(isnan(interpolated_marginal), axis=-1)
        highest[msk] = nan
        values.values = highest

        return values, dum

    def map_highest_marginal(self, dx, dy, depth, **kwargs):

        nClusters = self.marginalProbability.shape[-1]

        highest, dum = self.interpolate_highest_marginal(dx, dy, depth, **kwargs)

        ax, pc, cb = highest.pcolor(vmin=0, vmax=nClusters-1, cmapIntervals=nClusters, **dum)

        offset = (nClusters-1) / (2*nClusters)
        ticks = arange(offset, nClusters-offset, 2.0*offset)
        tick_labels = arange(nClusters)+1
        if not cb is None:
            cb.set_ticks(ticks)
            cb.set_ticklabels(tick_labels)

        return ax, pc, cb

    def mapRelativeError(self,dx, dy, system=0, mask = None, clip = True, useVariance=False, **kwargs):
        """ Create a map of a parameter """

        if useVariance:
            for line in self.lines:
                line.compute_relative_error_opacity()
            alpha = hstack([line.relative_error_opacity for line in self.lines])
            kwargs['alpha'] = alpha

        return  self.map(dx = dx, dy = dy, mask = mask, clip = clip, values = self.relativeError[system, :], **kwargs)

    def plot_cross_section(self, line_number, values, **kwargs):
        line_index = self.line_number.searchsorted(line_number)
        indices = self.lineIndices[line_index]

        return self.lines[line_index].plot_cross_section(values=values[indices, :], **kwargs)

    def plotdepth_slice(self, depth, variable, mask = None, clip = True, index=None, **kwargs):
        """ Create a depth slice through the recovered model """

        vals1D = self.depth_slice(depth=depth, variable=variable, index=index, **kwargs)
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


    def plotdata_misfit(self, normalized=True, **kwargs):
        """ Plot the observation locations """
        x = self.data_misfit
        if normalized:
            x = x / self.bestData.nActiveChannels
            x._name = "Normalized data misfit"

        return self.scatter2D(c=x, **kwargs)

    # @cached_property
    # def interface_probability_3D(self, dx, dy, lowerThreshold=0.0, **kwargs):


    # def interface_probability(self, depth, lowerThreshold=0.0, **kwargs):
    #     cell1 = self.zGrid.cellIndex(depth)

    #     values = self.interfaces[:, cell1]
    #     if lowerThreshold > 0.0:
    #         values = self.interfaces[:, cell1].deepcopy()
    #         values[values < lowerThreshold] = nan
    #     return values


    def plotInterfaceProbability(self, depth, lowerThreshold=0.0, **kwargs):

        cell1 = self.zGrid.cellIndex(depth)

        slce = self.interfaces[:, cell1]
        if lowerThreshold > 0.0:
            slce = deepcopy(self.interfaces[:, cell1])
            slce[slce < lowerThreshold] = nan

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

    def plot_marginal_probability(self, depth, index, **kwargs):
        cell1 = self.zGrid.cellIndex(depth)
        slce = self.marginalProbability[:, cell1, index]
        return self.scatter2D(c = slce, **kwargs)

    def plot_highest_marginal(self, depth, **kwargs):
        cell1 = self.zGrid.cellIndex(depth)
        slce = self.highest_marginal[:, cell1]
        return self.scatter2D(c = slce, **kwargs)


    def getParVsZ(self, bestModel=False, withDoi=True, reciprocateParameter=True, log10=True, clipNan=True):
        """ Get the depth and parameters, optionally within the doi """
        # Get the depths
        z = tile(self.zGrid,self.nPoints)

        if (bestModel):
            self.getAttribute(best=True, doi=withDoi)
            model = zeros(self.best.shape)
            model[:,:] = self.best
        else:
            self.getAttribute(mean=True, doi=withDoi)
            model = zeros(self.best.shape)
            model[:,:] = self.mean

        if (withDoi):
            zTmp = repeat(self.zGrid[:,newaxis],self.nPoints,axis=1)
            model[zTmp > self.doi] = nan

        model = model.reshape(model.size, order='F')

        if reciprocateParameter:
            model = 1.0/model.reshape(model.size, order='F')

        if log10:
            model = log10(model)

        res = StatArray(column_stack((model,z)))

        if (clipNan):
            res = res[logical_not(isnan(res[:,0]))]

        return res

    def kMeans(self, nClusters, precomputedParVsZ=None, standardize=False, log10Depth=False, plot=False, bestModel=True, withDoi=True, reciprocateParameter=True, log10=True, clipNan=True, **kwargs):
        """  """
        if (precomputedParVsZ is None):
            ParVsZ = self.getParVsZ(bestModel=bestModel, withDoi=withDoi, reciprocateParameter=reciprocateParameter, log10=log10, clipNan=clipNan)
        else:
            ParVsZ = precomputedParVsZ

        assert isinstance(ParVsZ, StatArray), "precomputedParVsZ must be an StatArray"

        if (log10Depth):
            ParVsZ[:,1] = log10(ParVsZ[:,1])

        return ParVsZ.kMeans(nClusters, standardize=standardize, nIterations=10, plot=plot, **kwargs)

    def GMM(self, ParVsZ, clusterID, trainPercent=90.0, plot=True):
        """ Classify the subsurface parameters """
        assert isinstance(ParVsZ, StatArray), "ParVsZ must be an StatArray"
        ParVsZ.GMM(clusterID, trainPercent=trainPercent, covType=['spherical','tied','diag','full'], plot=plot)

    def uncache(self, variable):

        if isinstance(variable, str):
            variable = [variable]

        for var in variable:
            if var in self.__dict__:
                del self.__dict__[var]

#     def toVTK(self, fName, dx, dy, mask=False, clip=False, force=False, method='ct'):
#         """ Convert a 3D volume of interpolated values to vtk for visualization in Paraview """

#         self.getMean3D(dx=dx, dy=dy, mask=mask, clip=clip, force=force, method=method)
#         self.pointcloud.getBounds()

#         x, y, intPoints = interpolation.getGridLocations2D(self.pointcloud.bounds, dx, dy)
#         z = self.zGrid


#         from pyvtk import VtkData, UnstructuredGrid, PointData, CellData, Scalars

#         # Get the 3D dimensions
#         mx = x.size
#         my = y.size
#         mz = z.nCells

#         nPoints = mx * my * mz
#         nCells = (mx-1)*(my-1)*(mz-1)

#         # Interpolate the elevation to the grid nodes
#         if (method == 'ct'):
#             tx,ty, vals, k = self.pointcloud.interpCloughTocher(dx = dx,dy=dy, values=self.elevation, mask = mask, clip = clip, extrapolate='nearest')
#         elif (method == 'mc'):
#             tx,ty, vals, k = self.pointcloud.interpMinimumCurvature(dx = dx, dy=dy, values=self.elevation, mask = mask, clip = clip)

#         vals = vals[:my,:mx]
#         vals = vals.reshape(mx*my)

#         # Set up the nodes and voxel indices
#         points = zeros([nPoints,3], order='F')
#         points[:,0] = tile(x, my*mz)
#         points[:,1] = tile(y.repeat(mx), mz)
#         points[:,2] = tile(vals, mz) - z.centres.repeat(mx*my)

#         # Create the cell indices into the points
#         p = arange(nPoints).reshape((mz, my, mx))
#         voxels = zeros([nCells, 8], dtype=int)
#         iCell = 0
#         for k in range(mz-1):
#             k1 = k + 1
#             for j in range(my-1):
#                 j1 = j + 1
#                 for i in range(mx-1):
#                     i1 = i + 1
#                     voxels[iCell,:] = [p[k1,j,i],p[k1,j,i1],p[k1,j1,i1],p[k1,j1,i], p[k,j,i],p[k,j,i1],p[k,j1,i1],p[k,j1,i]]
#                     iCell += 1

#         # Create the various point data
#         pointID = Scalars(arange(nPoints), name='Point iD')
#         pointElev = Scalars(points[:,2], name='Point Elevation (m)')

#         tmp = self.mean3D.reshape(size(self.mean3D))
#         tmp[tmp == 0.0] = nan

#         print(nanmin(tmp), nanmax(tmp))
#         tmp1 = 1.0 / tmp

#         print(nanmin(tmp), nanmax(tmp))
#         pointRes = Scalars(tmp1, name = 'log10(Resistivity) (Ohm m)')
#         tmp1 = log10(tmp)

#         pointCon = Scalars(tmp1, name = 'log10(Conductivity) (S/m)')

#         print(nPoints, tmp.size)

#         PData = PointData(pointID, pointElev, pointRes)#, pointCon)
#         CData = CellData(Scalars(arange(nCells),name='Cell iD'))
#         vtk = VtkData(
#               UnstructuredGrid(points,
#                                hexahedron=voxels),
# #                               ),
#               PData,
#               CData,
#               'Some Name'
#               )

#         vtk.tofile(fName, 'binary')
