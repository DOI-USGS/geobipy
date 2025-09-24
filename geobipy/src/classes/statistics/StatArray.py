from copy import deepcopy

from numpy import abs, allclose, arange, argmax, array, asarray, atleast_1d, concatenate, delete, hstack, diag
from numpy import diff, divide, expand_dims, flip, float32, float64, histogram, inf, insert, int32, int64, isnan
from numpy import mean, nan, nanmax, nanmin, ndarray, ndim, ones, r_, resize, s_, size, squeeze, sum, unique, where, zeros
from numpy import shape as npshape

from numpy import set_printoptions
from matplotlib.axes import SubplotBase
import h5py

from ...base import utilities as cf
from ...base import plotting as cP
from .Distribution import Distribution
from .baseDistribution import baseDistribution
from ..core.myObject import myObject
from ...base.HDF.hdfWrite import write_nd
from ...base import MPI as myMPI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

from ..core.DataArray import DataArray

class StatArray(DataArray):
    """Class extension to numpy.ndarray

    This subclass to a numpy array contains extra attributes that can describe the parameters it represents.
    One can also attach prior and proposal distributions so that it may be used in an MCMC algorithm easily.
    Because this is a subclass to numpy, the StatArray contains all numpy array methods and when passed to an
    in-place numpy function will return a StatArray.  See the example section for more information.

    StatArray(shape, name=None, units=None, **kwargs)

    Parameters
    ----------
    shape : int or sequence of ints or array_like or StatArray
        * If shape is int or sequence of ints : give the shape of the new StatArray e.g., ``2`` or ``(2, 3)``. All other arguments that can be passed to functions like numpy.zeros or numpy.arange can be used, see Other Parameters.
        * If shape is array_like : any object exposing the array interface, an object whose __array__ method returns an array, or any (nested) sequence. e.g., ``StatArray(numpy.arange(10))`` will cast the result into a StatArray and will maintain the properies passed through to that function. One could then attach the name, units, prior, and/or proposal through this interface too. e.g., ``x = StatArray(numpy.arange(10,dtype=numpy.int), name='aTest', units='someUnits')``
        * If shape is StatArray : the returned object is a deepcopy of the input. If name and units are specified with this option they will replace those parameters in the copy. e.g., ``y = StatArray(x, name='anotherTest')`` will be a deepcopy copy of x, but with a different name.
    name : str, optional
        The name of the StatArray.
    units : str, optional
        The units of the StatArray.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The desired data-type for the array.  Default is
        `numpy.float64`. Only used when shape is int or sequence of ints.
        The data type could also be a class.
    buffer : object exposing buffer interface, optional
        Used to fill the array with data. Only used when shape is int or sequence of ints.
    offset : int, optional
        Offset of array data in buffer. Only used when shape is int or sequence of ints.
    strides : tuple of ints, optional
        Strides of data in memory. Only used when shape is int or sequence of ints.
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C', then the array
        will be in C-contiguous order (rightmost-index varies the fastest).
        If order is 'F', then the returned array will be in
        Fortran-contiguous order (leftmost-index varies the fastest).
        If order is 'A' (default), then the returned array may be in any order (either C-, Fortran-contiguous, or even discontiguous),
        unless a copy is required, in which case it will be C-contiguous. Only used when shape is int or sequence of ints.

    Returns
    -------
    out : StatArray
        Extension to numpy.ndarray with additional attributes attached.

    Raises
    ------
    TypeError
        If name is not a str.
    TypeError
        If units is not a str.

    Notes
    -----
    When the StatArray is passed through a numpy function, the name and units are maintained in the new object.  Any priors or proposals are not kept for two reasons. a) keep computational overheads low, b) assume that a possible change in size or meaning of a parameter warrants a change in any attached distributions.

    Examples
    --------
    Since the StatArray is an extension to numpy, all numpy attached methods can be used.

    >>> from geobipy import StatArray
    >>> import numpy as np
    >>> x = StatArray(arange(10), name='test', units='units')
    >>> print(x.mean())
    4.5

    If the StatArray is passed to a numpy function that does not return a new instantiation, a StatArray will be returned (as opposed to a numpy array)

    >>> delete(x, 5)
    StatArray([0, 1, 2, 3, 4, 6, 7, 8, 9])

    However, if you pass a StatArray to a numpy function that is not in-place, i.e. creates new memory, the return type will be a numpy array and NOT a StatArray subclass

    >>> append(x,[3,4,5])
    array([0, 1, 2, ..., 3, 4, 5])

    See Also
    --------
    geobipy.src.classes.statistics.Distribution : For possible prior and proposal distributions

    """

    # "hidden" methods

    def __new__(cls, shape=None, name=None, units=None, verbose=False, **kwargs):
        """Instantiate a new StatArray """

        self = super().__new__(cls, shape, name, units, **kwargs)

        # Copies a StatArray but can reassign the name and units
        if isinstance(shape, StatArray):
            if (shape.hasPrior):
                self._prior = deepcopy(shape._prior)
            if (shape.hasProposal):
                self._proposal = deepcopy(shape._proposal)
            if (shape.hasPosterior):
                self._posterior = shape._posterior

        return self

    @property
    def n_posteriors(self):
        if self.hasPosterior:
            return size(self._posterior)
        return 0

    @property
    def posterior(self):
        """Returns the posterior if available. """
        if self.hasPosterior:
            return self._posterior
        else:
            return None

    @posterior.setter
    def posterior(self, value):
        if value is None:
            self._posterior = None
            return

        nP = size(value)
        if nP > 1:
            assert nP == self.shape[-1] or (self.shape[-1]%nP == 0), ValueError(f"Number of posteriors {nP} must match size of StatArray's first dimension {self.shape[-1]}")

        if nP == 1:
            if isinstance(value, list):
                value = value[0]

        self._posterior = value

    @property
    def prior(self):
        """Returns the prior if available. """
        if self.hasPrior:
            return self._prior
        else:
            return None

    @prior.setter
    def prior(self, value):
        if value is None:
            self._prior = None
            return
        assert isinstance(value, baseDistribution), TypeError('prior must be a Distribution')
        self._prior = value

    @property
    def proposal(self):
        """Returns the prior if available. """
        if self.hasProposal:
            return self._proposal
        else:
            return None

    @proposal.setter
    def proposal(self, value):
        if value is None:
            self._proposal = None
            return
        assert isinstance(value, baseDistribution), TypeError('proposal must be a Distribution')
        self._proposal = value

    # Methods
    @property
    def addressof(self):

        msg = super().addressof

        msg = 'StatArray: {} {}\n'.format(self.getNameUnits(), hex(id(self)))
        if self.hasPrior:
            msg += "Prior:\n{}".format(("|   "+self.prior.addressof.replace("\n", "\n|   "))[:-4])
        if self.hasProposal:
            msg += "Proposal:\n{}".format(("|   "+self.proposal.addressof.replace("\n", "\n|   "))[:-4])
        if self.hasPosterior:
            if self.n_posteriors > 1:
                for posterior in self.posterior:
                    msg += "Posterior:\n{}".format(("|   "+posterior.addressof.replace("\n", "\n|   "))[:-4])
            else:
                msg += "Posterior:\n{}".format(("|   "+self.posterior.addressof.replace("\n", "\n|   "))[:-4])
        return msg

    @property
    def address(self):
        out = super().address
        if self.hasPrior:
            out = hstack([out, self.prior.address])
        if self.hasProposal:
            out = hstack([out, self.proposal.address])
        return out


    def centred_grid_nodes(self, spacing):
        """Generates grid nodes centred over bounds

        Parameters
        ----------
        bounds : array_like
            bounds of the dimension
        spacing : float
            distance between nodes

        """
        # Get the discretization
        assert spacing > 0.0, ValueError("spacing must be positive!")
        sp = 0.5 * spacing
        return StatArray(arange(self.bounds[0] - sp, self.bounds[1] + (2*sp), spacing), self.name, self.units)

    # def confidence_interval(self, interval):
    #     values = self.flatten()
    #     return st.t.interval(interval, self.size - 1, loc=mean(values), scale=st.sem(values))

    def copy(self, order='F'):
        return StatArray(self)

    def copyStats(self, other):
        """Copy statistical properties from other to self

        [extended_summary]

        Parameters
        ----------
        other : [type]
            [description]
        """
        if other.hasPrior:
            self._prior = deepcopy(other._prior)
        if other.hasProposal:
            self._proposal = deepcopy(other._proposal)
        if other.hasPosterior:
            self._posterior = other._posterior

    def __deepcopy__(self, memo={}):

        other = StatArray(self, dtype=self.dtype)

        # other._name = self._name
        # other._units = self._units

        # if (self.hasPrior):
        #     other._prior = deepcopy(self._prior)
        # if (self.hasProposal):
        #     other._proposal = deepcopy(self._proposal)
        # if (self.hasPosterior):
        #     other._posterior = self._posterior  # deepcopy(self._posterior)

        return other

    def delete(self, i, axis=None):
        """Delete elements

        Parameters
        ----------
        i : slice, int or array of ints
            Indicate which sub-arrays to remove.
        axis : int, optional
            The axis along which to delete the subarray defined by `obj`.
            If `axis` is None, `obj` is applied to the flattened array.

        Returns
        -------
        out : StatArray
            Deepcopy of StatArray with deleted entry(ies).

        """
        out = super().delete(i, axis)
        out.copyStats(self)
        return out



    def insert(self, i, values, axis=0):
        """ Insert values

        Parameters
        ----------
        i : int, slice or sequence of ints
            Object that defines the index or indices before which `values` is inserted.
        values : array_like
            Values to insert into `arr`. If the type of `values` is different from that of `arr`, `values` is converted to the type of `arr`. `values` should be shaped so that ``arr[...,obj,...] = values`` is legal.
        axis : int, optional
            Axis along which to insert `values`.  If `axis` is None then `arr` is flattened first.

        Returns
        -------
        out : StatArray
            StatArray after inserting a value.

        """
        out = super().insert(i, values, axis)

        out.copyStats(self)
        return out

    @property
    def hasPosterior(self):
        """Check that the StatArray has an attached posterior.

        Returns
        -------
        out : bool
            Has an attached posterior.

        """

        try:
            return not self._posterior is None
        except:
            return False

    @property
    def hasPrior(self):
        """Check that the StatArray has an attached prior.

        Returns
        -------
        out : bool
            Has an attached prior.

        """

        try:
            return not self._prior is None
        except:
            return False

    @property
    def hasProposal(self):
        """Check that the StatArray has an attached proposal.

        Returns
        -------
        out : bool
            Has an attached proposal.

        """

        try:
            return not self._proposal is None
        except:
            return False

    def priorDerivative(self, order, i=None):
        """ Get the derivative of the prior.

        Parameters
        ----------
        order : int
            1 or 2 for first or second order derivative

        """
        assert self.hasPrior, TypeError('No prior defined on variable {}. Use StatArray.set_prior()'.format(self.name))
        if i is None:
            i = s_[:]
        out = self.prior.derivative(self[i], order)

        if order == 2 and ndim(out) < 2:
            return diag(out)
        return out

    def proposal_derivative(self, order, i=None):
        """ Get the derivative of the proposal.

        Parameters
        ----------
        order : int
            1 or 2 for first or second order derivative

        """
        assert self.hasProposal, TypeError('No proposal defined on variable {}. Use StatArray.setProposal()'.format(self.name))
        if i is None:
            i = s_[:]
        return self.proposal.derivative(self[i], order)


    def mahalanobis(self):
        assert self.hasPrior, ValueError("No prior attached")
        return self.prior.mahalanobis(self)


    def reset_posteriors(self):
        np = self.n_posteriors
        if np > 1:
            for post in self.posterior:
                post.reset()
        elif np == 1:
            self.posterior.reset()


    def resize(self, new_shape):
        """Resize a StatArray

        Resize a StatArray but copy over any attached attributes

        Parameters
        ----------
        new_shape : int or tuple of ints
            Shape of the resized array

        Returns
        -------
        out : StatArray
            Resized array.

        See Also
        --------
        numpy.resize : For more information.

        """
        out = super().resize(new_shape)
        out.copyStats(self)
        return out

    @property
    def sorted(self):
        return all(self[:-1] <= self[1:])

    @property
    def summary(self):
        """Write a summary of the StatArray

        Parameters
        ----------
        out : bool
            Whether to return the summary or print to screen

        Returns
        -------
        out : str, optional
            Summary of StatArray

        """
        msg = super().summary

        if self.hasPrior:
            msg += "Prior:\n{}".format(("|   "+self.prior.summary.replace("\n", "\n|   "))[:-4])

        if self.hasProposal:
            msg += "Proposal:\n{}".format(("|   "+self.proposal.summary.replace("\n", "\n|   "))[:-4])

        msg += 'has_posterior: {}\n'.format(self.hasPosterior)
        #     if self.n_posteriors > 1:
        #         for p in self.posterior:
        #             msg += "Posterior:\n{}".format(("|   "+p.summary.replace("\n", "\n|   "))[:-4])
        #     else:
        #         msg += "Posterior:\n{}".format(("|   "+self.posterior.summary.replace("\n", "\n|   "))[:-4])

        return msg

    def summaryPlot(self, **kwargs):
        """ Creates a summary plot of the StatArray with any attached priors, proposal, or posteriors. """

        gs1 = gridspec.GridSpec(
            nrows=1, ncols=1, left=0.5, right=0.92, wspace=0.01, hspace=0.13)
        gs2 = gridspec.GridSpec(
            nrows=2, ncols=1, left=0.085, right=0.40, wspace=0.06, hspace=0.5)

        ax = plt.subplot(gs2[0, 0])
        self.prior.plot_pdf()
        ax.set_xlabel(self.getNameUnits())
        ax.set_title('Prior')

        tmp = plt.subplot(gs2[1, 0], sharex=ax, sharey=ax)
        self.proposal.plot_pdf()
        tmp.set_title('Proposal')
        tmp.set_xlabel(self.getNameUnits())

        tmp = plt.subplot(gs1[0, 0])
        self.posterior.plot(**kwargs)
        tmp.set_title("Posterior")
        tmp.set_xlabel(self.getNameUnits())


    def pad(self, N):
        """ Copies the properties of a StatArray including all priors or proposals, but pads everything to the given size

        Parameters
        ----------
        N : int
            Size to pad to.

        Returns
        -------
        out : StatArray
            Padded StatArray

        """

        out = super().pad(N)

        try:
            pTmp = self.prior.pad(N)
            out._prior = pTmp
        except:
            pass
        try:
            pTmp = self.proposal.pad(N)
            out._proposal = pTmp
        except:
            pass
        if self.hasPosterior:
            out.posterior = deepcopy(self.posterior)
        return out

    def perturb(self, i=s_[:], relative=False, imposePrior=False, log=False):
        """Perturb the values of the StatArray using the attached proposal

        The StatArray must have had a proposal set using StatArray.setProposal()

        Parameters
        ----------
        i : slice or int or sequence of ints, optional
            Index or indices of self that should be perturbed.
        relative : bool
            Update the StatArray relative to the current values or assign the new samples to the StatArray.

        Raises
        ------
        TypeError
            If the proposal has not been set

        """
        self[i] = self.propose(i, relative, imposePrior, log)

    def probability(self, log, x=None, i=None, active=None):
        """Evaluate the probability of the values in self using the attached prior distribution

        Parameters
        ----------
        arg1 : array_like, optional
            Will evaluate the probability of the numbers in the arg1 using the prior attached to self
        i : slice or int or sequence of ints, optional
            Index or indices of self that should be evaluated.

        Returns
        -------
        out
            numpy.float

        Raises
        ------
        TypeError
            If the prior has not been set

        """

        assert (self.hasPrior), TypeError(
            'No prior defined on variable {}. Use StatArray.set_prior()'.format(self.name))

        samples = self[:]
        if not x is None:
            samples = x

        if not i is None:
            samples = samples[i]

        return self.prior.probability(x=samples, log=log, i=active)

    def propose(self, i=s_[:], relative=False, imposePrior=False, log=False):
        """Propose new values using the attached proposal distribution

        Parameters
        ----------
        i : ints, optional
            Only propose values for these indices.
        relative : bool, optional
            Use the proposal distribution as a relative change to the parameter values.
        imposePrior : bool, optional
            Continue to propose new values until the prior probability is non-zero or -infinity.
        log : bool, required if imposePrior is True.
            Use log probability when imposing the prior.

        Returns
        -------
        out : array_like
            Valuese generated from the proposal.

        """

        assert (self.hasProposal), TypeError('No proposal defined on variable {}. Use StatArray.setProposal()'.format(self.name))

        mv = self.proposal.multivariate

        if mv:
            nSamples = 1
            assert self.proposal.ndim == self.size, ValueError(
                ("Trying to generate {} samples from an {}-dimensional distribution."
                "Proposal dimensions must match self.size.").format(self.size, self.proposal.ndim))
        else:
            nSamples = self.size

        # Generate new values
        proposed = self.proposal.rng(nSamples)

        if relative:
            proposed = self + proposed

        if not imposePrior:
            return proposed[i] if mv else proposed

        assert self.hasPrior, TypeError('No prior defined on variable {}. Use StatArray.set_prior()'.format(self.name))

        p = self.probability(x=proposed, log=log, active=i)

        num = -inf if log else 0.0
        tries = 0
        while p == num:
            proposed = self.proposal.rng(nSamples)

            if relative:
                proposed = self + proposed

            p = self.probability(x=proposed, log=log, active=i)
            tries += 1
            if tries == 10:
                # print("Could not propose values for {}. Continually produced P(X)={}".format(self.summary, num), flush=True)
                return asarray(self[i]) if mv else self.item()

        return proposed[i] if mv else proposed

    def update_posterior(self, **kwargs):
        """Adds the current values of the StatArray to the attached posterior. """
        if not self.hasPosterior:
            return

        if self.n_posteriors > 1:
            active = atleast_1d(kwargs.get('active', range(self.n_posteriors)))
            for i in active:
                self._posterior[i].update(self.take(indices=i, axis=0), **kwargs)

        else:
            self.posterior.update(squeeze(self), **kwargs)

    # Plotting Routines
    def _init_posterior_plots(self, gs):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """
        if not self.hasPosterior:
            return

        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        if self.n_posteriors == 1:
            ax = plt.subplot(gs)
            cP.pretty(ax)

        else:
            gs = gs.subgridspec(self.n_posteriors, 1, hspace=1)
            ax = [plt.subplot(gs[0, 0])]
            ax += [plt.subplot(gs[i, 0], sharex=ax[0]) for i in range(1, self.n_posteriors)]

            for a in ax:
                cP.pretty(a)

        return ax

    def plot_posteriors(self, **kwargs):
        """Plot the posteriors of the StatArray.

        Parameters
        ----------
        ax : matplotlib axis or list of ax, optional
            Must match the number of attached posteriors.
            * If not specified, subplots are created vertically.

        """
        if not self.hasPosterior:
            return

        if kwargs.get('ax') is None:
            kwargs['ax'] = kwargs.pop('fig', plt.gcf())
        if not isinstance(kwargs['ax'], (list, SubplotBase)):
            kwargs['ax'] = self._init_posterior_plots(kwargs['ax'])

        ax = kwargs['ax']

        kwargs['cmap'] = kwargs.get('cmap', 'gray_r')
        kwargs['normalize'] = kwargs.get('normalize', True)

        if size(ax) > 1:
            assert len(ax) == self.n_posteriors, ValueError("Length of ax {} must equal number of attached posteriors {}".format(size(ax), self.n_posteriors))
            if 'overlay' in kwargs:
                assert len(kwargs['overlay']) == len(ax), ValueError("line in kwargs must have size {}".format(len(ax)))
            overlay = kwargs.pop('overlay', asarray([None for i in range(len(ax))]))

            ax = kwargs.pop('ax')
            for i in range(self.n_posteriors):
                ax[i].set_xscale('linear'); ax[i].cla()
                self.posterior[i].plot(overlay=overlay[i], ax=ax[i], **kwargs)
        else:
            if isinstance(ax, list):
                ax = ax[0]
            ax.cla()
            self.posterior.plot(**kwargs)

    def overlay_on_posteriors(self, overlay, ax, **kwargs):

        assert isinstance(overlay, StatArray), TypeError("overlay must have type StatArray")

        if size(ax) > 1:
            assert len(ax) == self.n_posteriors, ValueError("Length of ax {} must equal number of attached posteriors {}".format(size(ax), self.n_posteriors))
            assert len(overlay) == len(ax), ValueError("line in kwargs must have size {}".format(len(ax)))
            # overlay = kwargs.pop('overlay', asarray([None for i in range(len(ax))]))

            for i in range(self.n_posteriors):
                self.posterior[i].plot_overlay(value=overlay[i], ax=ax[i], **kwargs)
        else:
            if isinstance(ax, list):
                ax = ax[0]

            self.posterior.plot_overlay(value=overlay, ax=ax, **kwargs)

    # HDF Routines
    def createHdf(self, h5obj, name, shape=None, withPosterior=True, add_axis=None, fillvalue=None, upcast=True):
        """Create the Metadata for a StatArray in a HDF file

        Creates a new group in a HDF file under h5obj.
        A nested heirarchy will be created e.g., myName/data, myName/prior, and myName/proposal.
        This method can be used in an MPI parallel environment, if so however, a) the hdf file must have been opened with the mpio driver,
        and b) createHdf must be called collectively, i.e., called by every core in the MPI communicator that was used to open the file.
        In order to create large amounts of empty space before writing to it in parallel, the nRepeats parameter will extend the memory
        in the first dimension.

        Parameters
        ----------
        h5obj : h5py.File or h5py.Group
            A HDF file or group object to create the contents in.
        myName : str
            The name of the group to create.
        withPosterior : bool, optional
            Include the creation of space for any attached posterior.
        nRepeats : int, optional
            Inserts a first dimension into the shape of the StatArray of size nRepeats. This can be used to extend the available memory of
            the StatArray so that multiple MPI ranks can write to their respective parts in the extended memory.
        fillvalue : number, optional
            Initializes the memory in file with the fill value

        Notes
        -----
        This method can be used in serial and MPI. As an example in MPI.
        Given 10 MPI ranks, each with a 10 length array, it is faster to create a 10x10 empty array, and have each rank write its row.
        Rather than creating 10 separate length 10 arrays because the overhead when creating the file metadata can become very
        cumbersome if done too many times.

        Example
        -------
        >>> from geobipy import StatArray
        >>> from mpi4py import MPI
        >>> import h5py

        >>> world = MPI.COMM_WORLD

        >>> x = StatArray(4, name='test', units='units')
        >>> x[:] = world.rank

        >>> # This is a collective open of data in the file
        >>> f = h5py.File(fName,'w', driver='mpio',comm=world)
        >>> # Collective creation of space(padded by number of mpi ranks)
        >>> x.createHdf(f, 'x', nRepeats=world.size)

        >>> world.barrier()

        >>> # In a non collective region, we can write to different sections of x in the file
        >>> # Fake a non collective region
        >>> def noncollectivewrite(x, file, world):
        >>>     # Each rank carries out this code, but it's not collective.
        >>>     x.writeHdf(file, 'x', index=world.rank)
        >>> noncollectivewrite(x, f, world)

        >>> world.barrier()
        >>> f.close()

        """

        hdf_name = None
        if not self.hasPosterior:
            hdf_name = 'DataArray'

        grp = super().createHdf(h5obj, name, shape, add_axis, fillvalue, hdf_name=hdf_name)

        grp = h5obj[name]

        if withPosterior:
            self.create_posterior_hdf(grp, add_axis, fillvalue, upcast)

        return grp

    def create_posterior_hdf(self, grp, add_axis, fillvalue, upcast):
        if self.hasPosterior:
            grp.create_dataset('n_posteriors', data=self.n_posteriors)
            if self.n_posteriors > 1:
                for i in range(self.n_posteriors):
                    self.posterior[i].createHdf(grp, 'posterior{}'.format(i), add_axis=add_axis, fillvalue=fillvalue, withPosterior=False, upcast=upcast)
            else:
                self.posterior.createHdf(grp, 'posterior', add_axis=add_axis, fillvalue=fillvalue, withPosterior=False, upcast=upcast)

    def writeHdf(self, h5obj, name, withPosterior=True, index=None):
        """Write the values of a StatArray to a HDF file

        Writes the contents of the StatArray to an already created group in a HDF file under h5obj.
        This method can be used in an MPI parallel environment, if so however, the hdf file must have been opened with the mpio driver.
        Unlike createHdf, writeHdf does not have to be called collectively, each rank can call writeHdf independently,
        so long as they do not try to write to the same index.

        Parameters
        ----------
        h5obj : h5py._hl.files.File or h5py._hl.group.Group
            A HDF file or group object to write the contents to.
        myName : str
            The name of the group to write to. The group must have been created previously.
        withPosterior : bool, optional
            Include writing any attached posterior.
        index : int, optional
            If the group was created using the nRepeats option, index specifies the index'th entry at which to write the data

        Example
        -------
        >>> from geobipy import StatArray
        >>> from mpi4py import MPI
        >>> import h5py

        >>> world = MPI.COMM_WORLD

        >>> x = StatArray(4, name='test', units='units')
        >>> x[:] = world.rank

        >>> # This is a collective open of data in the file
        >>> f = h5py.File(fName,'w', driver='mpio',comm=world)
        >>> # Collective creation of space(padded by number of mpi ranks)
        >>> x.createHdf(f, 'x', nRepeats=world.size)

        >>> world.barrier()

        >>> # In a non collective region, we can write to different sections of x in the file
        >>> # Fake a non collective region
        >>> def noncollectivewrite(x, file, world):
        >>>     # Each rank carries out this code, but it's not collective.
        >>>     x.writeHdf(file, 'x', index=world.rank)
        >>> noncollectivewrite(x, f, world)

        >>> world.barrier()
        >>> f.close()

        """
        grp = super().writeHdf(h5obj, name, index)

        if withPosterior and (self.size > 0):
            self.write_posterior_hdf(grp, index)

        return grp

    def write_posterior_hdf(self, grp, index=None):
        if self.hasPosterior:
            if ndim(index) > 0:
                index = index[0]
            if self.n_posteriors > 1:
                for i in range(self.n_posteriors):
                    self.posterior[i].writeHdf(grp, 'posterior{}'.format(i), index=index)
            else:
                self.posterior.writeHdf(grp, 'posterior', index=index)

    @classmethod
    def fromHdf(cls, grp, name=None, index=None, skip_posterior=False, posterior_index=None):
        """Read the StatArray from a HDF group

        Given the HDF group object, read the contents into a StatArray.

        Parameters
        ----------
        h5obj : h5py._hl.group.Group
            A HDF group object to write the contents to.
        index : slice, optional
            If the group was created using the nRepeats option, index specifies the index'th entry from which to read the data.

        """

        is_file = False
        if isinstance(grp, str):
            file = h5py.File(grp, 'r')
            is_file = True
            grp = file

        if not "n_posteriors" in grp:
            out = DataArray.fromHdf(grp, name, index)
            if is_file:
                file.close()
            return out

        out = super().fromHdf(grp, name, index)

        if not name is None:
            grp = grp[name]

        if not skip_posterior:
            if posterior_index is None:
                posterior_index = index
            out.posteriors_from_hdf(grp, posterior_index)

        if is_file:
            file.close()

        return out

    def posteriors_from_hdf(self, grp, index):
        from .Histogram import Histogram
        n_posteriors = 0
        if 'n_posteriors' in grp:
            n_posteriors = asarray(grp['n_posteriors'])

        if n_posteriors == 0:
            self.posterior = None
            return

        posterior = None
        iTmp = index
        if index is not None:
            if ndim(index) > 0:
                iTmp = index[0]

        if n_posteriors == 1:
            posterior = Histogram.fromHdf(grp['posterior'], index=iTmp)

        elif n_posteriors > 1:
            posterior = []
            for i in range(n_posteriors):
                posterior.append(Histogram.fromHdf(grp['posterior{}'.format(i)], index=iTmp))

        self.posterior = posterior