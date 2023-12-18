from copy import deepcopy

from numpy import abs, allclose, arange, argmax, array, asarray, atleast_1d, concatenate, delete
from numpy import diff, divide, expand_dims, flip, float32, float64, histogram, inf, insert, int32, int64, isnan
from numpy import mean, nan, nanmax, nanmin, ndarray, ndim, ones, r_, resize, s_, size, squeeze, sum, unique, where, zeros
from numpy import shape as npshape

from numpy import set_printoptions
from matplotlib.axes import SubplotBase
import h5py
import scipy.stats as st


from ...base import utilities as cf
from ...base import plotting as cP
from ..statistics.Distribution import Distribution
from ..statistics.baseDistribution import baseDistribution
from .myObject import myObject
from ...base.HDF.hdfWrite import write_nd
from ...base import MPI as myMPI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

class StatArray(ndarray, myObject):
    """Class extension to numpy.ndarray

    This subclass to a numpy array contains extra attributes that can describe the parameters it represents.
    One can also attach prior and proposal distributions so that it may be used in an MCMC algorithm easily.
    Because this is a subclass to numpy, the StatArray contains all numpy array methods and when passed to an
    in-place numpy function will return a StatArray.  See the example section for more information.

    StatArray(shape, name=None, units=None, \*\*kwargs)

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

    def __new__(subtype, shape=None, name=None, units=None, verbose=False, **kwargs):
        """Instantiate a new StatArray """
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__

        if (not name is None):
            assert (isinstance(name, str)), TypeError('name must be a string')
            # Do some possible LateX checking. some Backslash operatores in LateX do not pass correctly as strings
            name = cf.str_to_raw(name)
        if (not units is None):
            assert (isinstance(units, str)), TypeError('units must be a string')
            # Do some possible LateX checking. some Backslash operatores in LateX do not pass correctly as strings
            units = cf.str_to_raw(units)

        if shape is None:
            shape = 0

        if isinstance(shape, list):
            shape = asarray(shape)

        # Copies a StatArray but can reassign the name and units
        if isinstance(shape, StatArray):
            if ndim(shape) == 0:
                self = ndarray.__new__(subtype, 1, **kwargs)
                self[:] = shape
            else:
                shp = npshape(shape)
                self = StatArray(shp, **kwargs) + shape

            if (shape.hasPrior):
                self._prior = deepcopy(shape._prior)
            if (shape.hasProposal):
                self._proposal = deepcopy(shape._proposal)
            if (shape.hasPosterior):
                self._posterior = shape._posterior
            if name is None and not shape._name is None:
                name = shape._name
            if units is None and not shape._units is None:
                units = shape._units

        # Can pass in a numpy function call like arange(10) as the first argument
        elif isinstance(shape, ndarray):
            # shape = deepcopy(shape)
            self = shape.view(StatArray)

        elif isinstance(shape, (float, float32, float64)):
            self = ndarray.__new__(subtype, 1, **kwargs)
            self[:] = shape

        else:
            self = ndarray.__new__(subtype, asarray(shape), **kwargs)
            self[:] = 0

        # Set the name of the StatArray
        self.name = name
        # Set the Units of the StatArray
        self.units = units

        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        try:
            d = obj.__dict__
            self._name = d.get('_name')
            self._units = d.get('_units')
        except:
            self._name = None
            self._units = None

    def __array_wrap__(self, out_arr, context=None):
        return ndarray.__array_wrap__(self, out_arr, context)

    # Properties

    @property
    def bounds(self):
        return r_[nanmin(self), nanmax(self)]

    @property
    def name(self):
        return "" if self._name is None else self._name

    @name.setter
    def name(self, values):
        if values is None:
            self._name = None
        else:
            assert isinstance(values, str)
            self._name = values

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
            assert nP == self.shape[-1] or (self.shape[-1]%nP == 0), ValueError("Number of posteriors must match size of StatArray's first dimension")

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

    @property
    def units(self):
        return "" if self._units is None else self._units

    @units.setter
    def units(self, values):
        if values is None:
            self._units = None
        else:
            assert isinstance(values, str)
            self._units = values

    # Methods

    def hasLabels(self):
        return not self.getNameUnits() == ""

    @property
    def addressof(self):
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

    def abs(self):
        """Take the absolute value.  In-place operation.

        Returns
        -------
        out : StatArray
            Absolute value

        """

        out = abs(self)
        out.name = "|{}|".format(out.name)

        return out

    def append(self, values, axis=0):
        """Append to a StatArray

        Appends values the end of a StatArray. Be careful with repeated calls to this method as it can be slow due to reallocating memory.

        Parameters
        ----------
        values : scalar or array_like
            Numbers to append

        Returns
        -------
        out : StatArray
            Appended StatArray

        """
        i = self.shape[axis]
        return self.insert(i=i, values=values, axis=axis)

    def argmax_multiple_to_nan(self, axis=0):
        """Perform the numpy argmax function on the StatArray but optionally mask multiple max values as NaN.

        Parameters
        ----------
        nan_multiple : bool
            If multiple locations contain the same max value, mask as nan.

        Returns
        -------
        out : ndarray of floats
            Array of indices into the array. It has the same shape as `self.shape`
            with the dimension along `axis` removed.

        """

        mx = argmax(self, axis=axis).astype(float)
        x = sum((self == max(self, axis=axis)), axis=axis)
        mx[x > 1.0] = nan

        return mx

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

    def confidence_interval(self, interval):
        values = self.flatten()
        return st.t.interval(interval, self.size - 1, loc=mean(values), scale=st.sem(values))

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
        tmp = delete(self, i, axis=axis)
        out = self.resize(tmp.shape)
        out[:] = tmp[:]

        out.copyStats(self)
        return out

    def edges(self, min=None, max=None, axis=-1):
        """Get the midpoint values between elements in the StatArray

        Returns an size(self) + 1 length StatArray of the midpoints between each element.
        The first and last element are projected edges based on the difference between first two and last two elements in self.
        edges[0] = self[0] - 0.5 * (self[1]-self[0])
        edges[-1] = self[-1] + 0.5 * (self[-1] - self[-2])
        If min and max are given, the edges are fixed and not calculated.

        Parameters
        ----------
        min : float, optional
            Fix the leftmost edge to min.
        max : float, optional
            Fix the rightmost edge to max.
        axis : int, optional
            Compute edges along this dimension if > 1D.

        Returns
        -------
        out : StatArray
            Edges of the StatArray

        """
        if self.size == 1:
            d = squeeze(asarray([self - 1, self + 1]))
            return StatArray(d, self.name, self.units)
        else:
            d = 0.5 * diff(self, axis=axis)

        x0 = self.take(indices=0, axis=axis)
        x1 = self.take(indices=-1, axis=axis)
        x2 = self.take(indices=arange(self.shape[axis]-1), axis=axis)

        e0 = expand_dims(x0 - d.take(indices=0, axis=axis), axis)
        e1 = x2 + d
        e2 = expand_dims(x1 + d.take(indices=-1, axis=axis), axis)

        if not min is None:
            e0[:] = min
        if not max is None:
            e2[:] = max

        edges = concatenate([e0, e1, e2], axis=axis)

        return StatArray(edges, self.name, self.units)

    def firstNonZero(self, axis=0, invalid_val=-1):
        """Find the indices of the first non zero values along the axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to find first non zeros.
        invalid_val : int, optional
            When zero is not available, return this index.

        Returns
        -------
        out : array_like
            Indices of the first non zero values.

        """

        msk = self != 0.0
        return where(msk.any(axis=axis), msk.argmax(axis=axis), invalid_val)

    @property
    def label(self):
        return self.getNameUnits()

    def getNameUnits(self):
        """Get the name and units

        Gets the name and units attached to the StatArray. Units, if present are surrounded by parentheses

        Returns
        -------
        out : str
            String containing name(units).

        """
        out = self.name
        u = self.units
        return out if u == "" else "{} ({})".format(out, u)

    # def getName(self):
    #     """Get the name of the StatArray

    #     If the name has not been attached, returns an empty string

    #     Returns
    #     -------
    #     out : str
    #         The name of the StatArray.

    #     """

    #     return "" if self.name is None else self.name

    # def getUnits(self):
    #     """Get the units of the StatArray

    #     If the units have not been attached, returns an empty string

    #     Returns
    #     -------
    #     out : str
    #         The unist of the StatArray

    #     """

    #     return "" if self.units is None else self.units

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
        tmp = insert(arr=self, obj=i, values=values, axis=axis)
        # tmp = insert(self, i, values, axis)
        out = self.resize(tmp.shape)  # Keeps the prior and proposal if set.
        out[:] = tmp[:]

        out.copyStats(self)
        return out

    def interleave(self, other):
        """Interleave two arrays together like zip

        Parameters
        ----------
        other : StatArray or array_like
            Must have same size

        Returns
        -------
        out : StatArray
            Interleaved arrays

        """
        assert self.size == other.size, ValueError(
            "other must have size {}".format(self.size))
        out = StatArray((self.size + other.size), dtype=self.dtype)
        out[0::2] = self
        out[1::2] = other
        return out

    def internalEdges(self, axis=-1):
        """Get the midpoint values between elements in the StatArray

        Returns an size(self) + 1 length StatArray of the midpoints between each element

        Returns
        -------
        out : StatArray
            Edges of the StatArray

        """
        assert (self.size > 1), ValueError("Size of StatArray must be > 1")
        d = 0.5 * diff(self, axis=axis)

        x2 = self.take(indices=arange(self.shape[axis]-1), axis=axis)
        edges = x2 + d

        return StatArray(edges, self.name, self.units)

    def diff(self, axis=-1):
        assert (self.size > 1), ValueError("Size of StatArray must be > 1")

        return StatArray(diff(self, axis=axis), self.name, self.units)

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

    def index(self, values):
        """Find the index of values.

        Assumes that self is monotonically increasing!

        Parameters
        ----------
        values : scalara or array_like
            Find the index of these values.

        Returns
        -------
        out : ints
            Indicies into self.

        """

        return self.searchsorted(values, side='right') - 1

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

    def lastNonZero(self, axis=0, invalid_val=-1):
        """Find the indices of the first non zero values along the axis.

        Parameters
        ----------
        axis : int
            Axis along which to find first non zeros.

        Returns
        -------
        out : array_like
            Indices of the first non zero values.

        """

        msk = self != 0.0
        val = self.shape[axis] - flip(msk, axis=axis).argmax(axis=axis)
        return where(msk.any(axis=axis), val, invalid_val)

    def mahalanobis(self):
        assert self.hasPrior, ValueError("No prior attached")
        return self.prior.mahalanobis(self)

    def nanmin(self):
        return nanmin(self)

    def nanmax(self):
        return nanmax(self)

    def normalize(self, axis=None):
        """Normalize to range 0 - 1. """
        mn = nanmin(self, axis=axis)
        mx = nanmax(self, axis=axis)

        t = mx - mn
        return divide((self - mn), t)

    def prepend(self, values, axis=0):
        """Prepend to a StatArray

        Prepends numbers to a StatArray, Do not use this too often as it is quite slow

        Parameters
        ----------
        values : scalar or array_like
            A number to prepend.

        Returns
        -------
        out : StatArray
            StatArray with prepended values.

        """
        i = zeros(size(values), dtype=int32)
        return self.insert(i, values, axis=axis)

    @property
    def range(self):
        return nanmax(self) - nanmin(self)

    def rescale(self, a, b):
        """Rescale to the interval (a, b)

        Parameters
        ----------
        a : float
            Lower limit
        b : float
            Upper limit

        Returns
        -------
        out : StatArray
            Rescaled Array

        """

        return (((b - a) * (self - self.min())) / (self.max() - self.min())) + a

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
        out = StatArray(resize(self, new_shape), self.name, self.units)
        out.copyStats(self)
        return out

    def smooth(self, a):
        return StatArray(cf.smooth(self, a))

    def standardize(self, axis=None):
        """Standardize by subtracting the mean and dividing by the standard deviation. """

        mn = mean(self, axis=axis)
        std = std(self, axis=axis)

        return (self - mn) / std

    def strip_nan(self):
        i = ~isnan(self)
        return self[i]

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
        set_printoptions(threshold=5)

        if self.size == 0:
            return "None"

        msg = ('Name: {} {}\n'
               'Shape: {}\n'
               'Values: {}\n'
               'Min: {}\n'
               'Max: {}\n').format(self.getNameUnits(), hex(id(self)), self.shape, self, self.min(), self.max())
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

    @property
    def values(self):
        return self.item()

    def verbose(self):
        """Explicit print of every element """
        set_printoptions(threshold=self.size)
        print(self[:])
        set_printoptions(threshold=5)

    def isRegular(self, axis=-1):
        """Checks that the values change regularly

        Returns
        -------
        out : bool
            Is regularly changing.

        """
        if size(self) == 1:
            return True
        tmp = diff(self, axis=axis)
        return allclose(tmp, tmp[0])

    # Statistical Routines

    def hist(self, bins=10, range=None, normed=None, weights=None, density=None, **kwargs):
        """Plot a histogram of the StatArray

        Plots a histogram, estimates the mean and standard deviation and overlays the PDF of a normal distribution with those values, if density=1.

        See Also
        --------
        geobipy.plotting.hist : For geobipy additional arguments
        matplotlib.pyplot.hist : For histogram related arguments

        Example
        -------
        >>> from geobipy import StatArray
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> x = StatArray(random.randn(1000), name='name', units='units')
        >>> plt.figure()
        >>> x.hist()
        >>> plt.show()

        """
        cnts, bins = histogram(
            self, bins=bins, range=range, normed=normed, weights=weights, density=density)
        bins = StatArray(bins, name=self.name, units=self.units)
        cP.bar(cnts, bins, **kwargs)

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

        out = StatArray(N, name=self.name, units=self.units)
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

    def rolling(self, numpyFunction, window=1):
        wd = cf.rolling_window(self, window)
        return StatArray(numpyFunction(wd, -1), self.name, self.units)

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

    def bar(self, x=None, i=None, **kwargs):
        """Plot the StatArray as a bar chart.

        The values in self are the heights of the bars. Auto labels it if x has type geobipy.StatArray

        Parameters
        ----------
        x : array_like or StatArray, optional
            The horizontal locations of the bars
        i : sequence of ints, optional
            Plot the ith indices of self, against the ith indices of x.

        Returns
        -------
        ax
            matplotlib .Axes

        See Also
        --------
        matplotlib.pyplot.bar : For additional keyword arguments you may use.

        """
        if (x is None):
            x = StatArray(arange(size(self)+1), name="Array index")
        # if i is not None:
        #     x = x[i]

        return cP.bar(self, x, **kwargs)

    def pcolor(self, x=None, y=None, **kwargs):
        """Create a pseudocolour plot of the StatArray array, Actually uses pcolormesh for speed.

        If the arguments x and y are geobipy.StatArray classes, the axes can be automatically labelled.
        Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

        Parameters
        ----------
        x : 1D array_like or StatArray, optional
            Horizontal coordinates of the values edges.
        y : 1D array_like or StatArray, optional
            Vertical coordinates of the values edges.

        Other Parameters
        ----------------
        alpha : scalar or arrya_like, optional
            If alpha is scalar, behaves like standard matplotlib alpha and opacity is applied to entire plot
            If array_like, each pixel is given an individual alpha value.
        log : 'e' or float, optional
            Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
            Values in c that are <= 0 are masked.
        equalize : bool, optional
            Equalize the histogram of the colourmap so that all colours have an equal amount.
        nbins : int, optional
            Number of bins to use for histogram equalization.
        xscale : str, optional
            Scale the x axis? e.g. xscale = 'linear' or 'log'
        yscale : str, optional
            Scale the y axis? e.g. yscale = 'linear' or 'log'.
        flipX : bool, optional
            Flip the X axis
        flipY : bool, optional
            Flip the Y axis
        grid : bool, optional
            Plot the grid
        noColorbar : bool, optional
            Turn off the colour bar, useful if multiple plotting plotting routines are used on the same figure.
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.
        classes : dict, optional
            A dictionary containing three entries.
            classes['id'] : array_like of same shape as self containing the class id of each element in self.
            classes['cmaps'] : list of matplotlib colourmaps.  The number of colourmaps should equal the number of classes.
            classes['labels'] : list of str.  The length should equal the number of classes.
            If classes is provided, alpha is ignored if provided.

        Returns
        -------
        ax
            matplotlib .Axes

        See Also
        --------
        matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

        """

        my = y
        if (not y is None):
            assert (isinstance(y, StatArray)), TypeError(
                "y must be a StatArray")
            if size(y) == self.size:
                try:
                    my = y.edges()
                except:
                    pass

        if (self.ndim == 1):
            return cP.pcolor_1D(self, y=my, **kwargs)
        else:
            mx = x
            if (not x is None):
                assert (isinstance(x, StatArray)), TypeError(
                    "x must be a StatArray")
                if size(x) == self.size:
                    try:
                        mx = x.edges()
                    except:
                        pass

            return cP.pcolor(self, x=mx, y=my, **kwargs)

    def plot(self, x=None, i=None, axis=0, **kwargs):
        """Plot self against x

        If x and y are StatArrays, the axes are automatically labelled.

        Parameters
        ----------
        x : array_like or StatArray
            The abcissa
        i : sequence of ints, optional
            Plot the ith indices of self, against the ith indices of x.
        axis : int, optional
            If self is 2D, plot values along this axis.
        log : 'e' or float, optional
            Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
            Values in c that are <= 0 are masked.
        xscale : str, optional
            Scale the x axis? e.g. xscale = 'linear' or 'log'.
        yscale : str, optional
            Scale the y axis? e.g. yscale = 'linear' or 'log'.
        flipX : bool, optional
            Flip the X axis
        flipY : bool, optional
            Flip the Y axis
        labels : bool, optional
            Plot the labels? Default is True.

        Returns
        -------
        ax
            matplotlib .Axes

        See Also
        --------
            matplotlib.pyplot.plot : For additional keyword arguments you may use.

        """
        if (not i is None):
            assert (isinstance(i, (slice, tuple))), TypeError(
                "i must be a slice, use s_[]")

        if (self.ndim == 1):
            if (x is None):
                x = StatArray(arange(self.size), 'Array Index')
            if (i is None):
                i = s_[:self.size]
            j = i
        else:
            if (x is None):
                x = StatArray(arange(self.shape[axis]), 'Array Index')
            if (i is None):
                i = s_[:self.shape[0], :self.shape[1]]
            j = i[axis]

        return cP.plot(x[j], self[i], **kwargs)

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
                ax[i].cla()
                self.posterior[i].plot(overlay=overlay[i], ax=ax[i], **kwargs)
        else:
            if isinstance(ax, list):
                ax = ax[0]
            ax.cla()
            self.posterior.plot(**kwargs)

    def scatter(self, x=None, y=None, i=None, **kwargs):
        """Create a 2D scatter plot.

        Create a 2D scatter plot, if the y values are not given, the colours are used instead.
        If the arrays x, y, and c are geobipy.StatArray classes, the axes are automatically labelled.
        Can take any other matplotlib arguments and keyword arguments e.g. markersize etc.

        Parameters
        ----------
        x : 1D array_like or StatArray
            Horizontal locations of the points to plot
        c : 1D array_like or StatArray
            Colour values of the points
        y : 1D array_like or StatArray, optional
            Vertical locations of the points to plot, if y = None, then y = c.
        i : sequence of ints, optional
            Plot a subset of x, y, c, using the indices in i.

        See Also
        --------
        geobipy.plotting.Scatter2D : For additional keyword arguments you may use.

        """

        assert ndim(self) == 1, TypeError(
            'scatter only works with a 1D array')

        if (x is None):
            x = StatArray(arange(self.size), 'Array Index')
        else:
            assert size(x) == self.size, ValueError(
                'x must be size '+str(self.size))

        if (y is None):
            y = self

        c = kwargs.pop('c', self)

        return cP.scatter2D(x=x, y=y, c=c, i=i, **kwargs)

    def stackedAreaPlot(self, x=None, i=None, axis=0, labels=[], **kwargs):
        """Create stacked area plot where column elements are stacked on top of each other.

        Parameters
        ----------
        x : array_like or StatArray
            The abcissa.
        i : sequence of ints, optional
            Plot a subset of x, y, c, using the indices in i.

        Other Parameters
        ----------------
        axis : int
            Plot along this axis, stack along the other axis.
        labels : list of str, optional
            The labels to assign to each column.
        colors : matplotlib.colors.LinearSegmentedColormap or list of colours
            The colour used for each column.
        xscale : str, optional
            Scale the x axis? e.g. xscale = 'linear' or 'log'.
        yscale : str, optional
            Scale the y axis? e.g. yscale = 'linear' or 'log'.

        Returns
        -------
        ax
            matplotlib .Axes

        See Also
        --------
            matplotlib.pyplot.scatterplot : For additional keyword arguments you may use.

        """

        assert (self.ndim == 2), TypeError(
            'stackedAreaPlot only works with 2D arrays')
        if (not i is None):
            assert (isinstance(i, (slice, tuple))), TypeError(
                "i must be a slice, use s_[]")

        if (x is None):
            x = StatArray(arange(self.shape[axis]), 'Array Index')
        else:
            assert size(x) == self.shape[axis], ValueError(
                'x must be size '+str(self.shape[axis]))

        if (i is None):
            i = s_[:self.shape[0], :self.shape[1]]
        j = i[axis]

        if (axis == 0):
            tmp = self.T
        else:
            tmp = self

        ma = tmp.copy()
        ma[ma >= 0.0] = 0.0
        cP.stackplot2D(x[j], ma[i], labels=labels, **kwargs)
        ma[:] = tmp[:]
        ma[ma <= 0.0] = 0.0
        cP.stackplot2D(x[j], ma[i], labels=[], **kwargs)

    # HDF Routines

    @property
    def hdf_name(self):
        """Create a string that describes class instantiation

        Returns a string that should be used as an attr['repr'] in a HDF group.
        This allows reading of the attribute from the hdf file, evaluating it to return an object,
        and then reading the hdf contents via the object's methods.

        Returns
        -------
        out
            str

        """
        return(r'StatArray()')

    def createHdf(self, h5obj, name, shape=None, withPosterior=True, add_axis=None, fillvalue=None, upcast=True):
        """Create the Metadata for a StatArray in a HDF file

        Creates a new group in a HDF file under h5obj.
        A nested heirarchy will be created e.g., myName\data, myName\prior, and myName\proposal.
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

        # create a new group inside h5obj
        grp = self.create_hdf_group(h5obj, name)

        if not self._name is None:
            grp.attrs['name'] = self.name
        if not self._units is None:
            grp.attrs['units'] = self.units

        # if shape is not None:
        #     grp.create_dataset('data', shape, dtype=self.dtype, fillvalue=fillvalue)
        # else:
        if (add_axis is None):
            shape = self.shape if shape is None else shape
            grp.create_dataset('data', shape, dtype=self.dtype, fillvalue=fillvalue)
        else:
            if isinstance(add_axis, (int, int32, int64)):
                shap = atleast_1d(add_axis).copy()
            elif isinstance(add_axis, tuple):
                shap = add_axis
            else:
                shap = add_axis.shape

            if (self.size <= 1):
                grp.create_dataset('data', [*shap], dtype=self.dtype, fillvalue=fillvalue)
            else:
                shape = self.shape if shape is None else shape
                grp.create_dataset('data', [*shap, *shape], dtype=self.dtype, fillvalue=fillvalue)

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
        grp = h5obj.get(name)

        if self.size > 0:
            write_nd(self, grp, 'data', index=index)

            if withPosterior:
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

        if not name is None:
            grp = grp[name]

        if (index is None):
            d = asarray(grp['data'])
        else:
            if isinstance(index, slice):
                d = asarray(grp['data'][index])
            else:
                d = asarray(grp['data'][s_[index]])

        if ndim(d) >= 2:
            d = squeeze(d)

        # Do not use self, return self here.  You tried it, it doesnt work.
        name = None
        if 'name' in grp.attrs:
            name = grp.attrs['name']
        else:  # Try to get it from the repr
            if '"' in grp.attrs['repr']:
                name = grp.attrs['repr'].split('"')[1]

        units = None
        if 'units' in grp.attrs:
            units = grp.attrs['units']
        else:  # Try to get it from the repr
            if '"' in grp.attrs['repr']:
                units = grp.attrs['repr'].split('"')[3]

        if 'StatArray((1,)' in grp.attrs['repr']:
            out = cls(1, name, units) + d
        else:
            out = cls(d, name, units)

        if not skip_posterior:
            if posterior_index is None:
                posterior_index = index
            out.posteriors_from_hdf(grp, posterior_index)

        if is_file:
            file.close()

        return out

    def posteriors_from_hdf(self, grp, index):
        from ..statistics.Histogram import Histogram
        n_posteriors = 0
        if 'n_posteriors' in grp:
            n_posteriors = asarray(grp['n_posteriors'])
        if 'nPosteriors' in grp:
            n_posteriors = asarray(grp['nPosteriors'])

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

    # Classification Routines
    def kMeans(self, nClusters, standardize=False, nIterations=10, plot=False, **kwargs):
        """ Perform K-Means clustering on the StatArray """
        if (standardize):
            tmp = scale(self)
        else:
            tmp = self

        kmeans = KMeans(init='k-means++', n_clusters=nClusters,
                        n_init=nIterations)
        kmeans.fit(tmp)
        clusterID = StatArray(kmeans.predict(tmp), name='cluster ID')

        if (plot):
            if (self.ndim == 1):
                cP.myscatter2D(self, clusterID, **kwargs)
            elif (self.ndim == 2):
                cP.myscatter2D(self[:, 0], self[:, 1], c=clusterID, **kwargs)
            else:
                # Going to have to do PCA or something and plot on two primary axes?
                # I.e. do a dimensionality reduction
                raise NotImplementedError

        return clusterID, kmeans

    def fit_mixture(self, mixture_type='gaussian', log=None, mean_bounds=None, variance_bounds=None, k=[1, 5], tolerance=0.05):
        """Uses Gaussian mixture models to fit the histogram.

        Starts at the minimum number of clusters and adds clusters until the BIC decreases less than the tolerance.

        Parameters
        ----------
        nSamples

        log

        mean_bounds

        variance_bounds

        k : ints
            Two ints with starting and ending # of clusters

        tolerance

        """

        from sklearn.mixture import GaussianMixture
        # from smm import SMM

        if mixture_type.lower() == 'gaussian':
            mod = GaussianMixture
        else:
            mod = SMM

        # Samples the histogram
        X = self.strip_nan()
        X = X.flatten()[:, None]
        X, _ = cf._log(X, log)

        # of clusters
        k_ = k[0]

        best = mod(n_components=k_).fit(X)
        BIC0 = best.bic(X)
        all_models = []
        all_models.append(best)

        k_ += 1
        go = k_ <= k[1]

        while go:
            model = mod(n_components=k_).fit(X)
            BIC1 = model.bic(X)
            all_models.append(model)

            percent_reduction = abs((BIC1 - BIC0)/BIC0)

            go = True
            if BIC1 < BIC0:
                best = model
                BIC0 = BIC1

            else:
                go = False

            if (percent_reduction < tolerance):
                go = False

            k_ += 1
            go = go & (k_ <= k[1])

        active = ones(best.n_components, dtype=bool)

        means = squeeze(best.means_)
        try:
            variances = squeeze(best.covariances_)
        except:
            variances = squeeze(best.covariances)

        if not mean_bounds is None:
            active = (mean_bounds[0] <= means) & (means <= mean_bounds[1])

        if not variance_bounds is None:
            active = (variance_bounds[0] <= variances) & (
                variances <= variance_bounds[1]) & active

        return best, atleast_1d(active), all_models

    def gaussianMixture(self, clusterID, trainPercent=75.0, covType=['spherical'], plot=True):
        """ Use a Gaussian Mixing Model to classify the data.
        clusterID is the initial assignment of the rows to their clusters """

        assert clusterID.size == self.shape[0], "size of clusterID "+str(
            clusterID.size)+" must match first dimension size "+str(self.shape[0])

        # Get the number of splits
        nSplits = int(100.0/(100.0-trainPercent))

        # Break up the dataset into non-overlapping sets, e.g. training (75%) and testing (25%)
        skf = StratifiedKFold(n_splits=nSplits)

        # Only take the first fold.
        trainIndex, testIndex = next(iter(skf.split(self, clusterID)))

        xTrain = self[trainIndex]
        xTest = self[testIndex]
        yTrain = clusterID[trainIndex]
        yTest = clusterID[testIndex]

        nClusters = unique(clusterID).size

        # Try GMMs using different types of covariances.
        models = dict((cov_type, GaussianMixture(n_components=nClusters,
                                                 covariance_type=cov_type, max_iter=20, random_state=None))
                      for cov_type in covType)

        nModels = len(models)

        if (plot):
            plt.figure(figsize=(3 * nModels // 2, 6))
            plt.subplots_adjust(bottom=.01, top=0.95,
                                hspace=.15, wspace=.05, left=.01, right=.99)

        for index, (name, estimator) in enumerate(models.items()):
            # Since we have class labels for the training data, we can
            # initialize the GMM parameters in a supervised manner.
            estimator.means_init = array(
                [xTrain[yTrain == i].mean(axis=0) for i in range(nClusters)])

            # Train the other parameters using the EM algorithm.
            estimator.fit(xTrain)

            if (plot):
                h = plt.subplot(2, nModels // 2, index + 1)
            #make_ellipses(estimator, h)

            #    tmp = x[z == n]
            #    c=zeros(tmp.shape[0])+n
                #cP.myscatter2D(tmp[:, 0], tmp[:, 1], s=0.8, c=c)
            #    plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color)#,label=ris.target_names[n])

            # Plot the test data with crosses
            # for n, color in enumerate(colors):
            #    data = X_test[y_test == n]
            #    plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

            y_train_pred = estimator.predict(xTrain)
            z = estimator.predict(self)
            if (plot):
                cP.myscatter2D(self[:, 0], self[:, 1], c=z)

            train_accuracy = mean(
                y_train_pred.ravel() == yTrain.ravel()) * 100
            if (plot):
                plt.text(0.05, 0.9, 'Train accuracy: %.1f' %
                         train_accuracy, transform=h.transAxes)

            y_test_pred = estimator.predict(xTest)
            test_accuracy = mean(y_test_pred.ravel() == yTest.ravel()) * 100
            if (plot):
                plt.text(0.05, 0.8, 'Test accuracy: %.1f' %
                         test_accuracy, transform=h.transAxes)

                # plt.xticks(())
                # plt.yticks(())
                cP.title(name)

    # MPI Routines
    def Bcast(self, world, root=0):
        """Broadcast the StatArray to every rank in the MPI communicator.

        Parameters
        ----------
        world : mpi4py.MPI.Comm
            The MPI communicator over which to broadcast.
        root : int, optional
            The rank from which to broadcast.  Default is 0 for the master rank.

        Returns
        -------
        out : StatArray
            The broadcast StatArray on every rank in the MPI communicator.

        """
        name = " " + self.name
        units = " " + self.units
        tmp = name + ',' + units
        nameUnits = world.bcast(tmp)
        name, units = nameUnits.split(',')
        tmp = myMPI.Bcast(self, world, root=root)
        return StatArray(tmp, name, units, dtype=tmp.dtype)

    def Scatterv(self, starts, chunks, world, axis=0, root=0):
        """Scatter variable lengths of the StatArray using MPI

        Takes the StatArray and gives each core in the world a chunk of the array.

        Parameters
        ----------
        starts : array of ints
            1D array of ints with size equal to the number of MPI ranks. Each element gives the starting index for a chunk to be sent to that core. e.g. starts[0] is the starting index for rank = 0.
        chunks : array of ints
            1D array of ints with size equal to the number of MPI ranks. Each element gives the size of a chunk to be sent to that core. e.g. chunks[0] is the chunk size for rank = 0.
        world : mpi4py.MPI.Comm
            The MPI communicator over which to Scatterv.
        axis : int, optional
            This axis is distributed amongst ranks.
        root : int, optional
            The MPI rank to ScatterV from. Default is 0.

        Returns
        -------
        out : StatArray
            The StatArray distributed amongst ranks.

        """
        name = " " + self.name
        units = " " + self.units
        tmp = name + ',' + units
        nameUnits = world.bcast(tmp)
        name, units = nameUnits.split(',')
        tmp = myMPI.Scatterv(self, starts, chunks, world, axis, root)
        return StatArray(tmp, name, units, dtype=tmp.dtype)

    def Isend(self, dest, world, ndim=None, shape=None, dtype=None):
        name = " " + self.name
        units = " " + self.units
        tmp = name + ',' + units
        world.send(tmp, dest=dest)
        myMPI.Isend(self, dest=dest, world=world,
                    ndim=ndim, shape=shape, dtype=dtype)

    @classmethod
    def Irecv(cls, source, world, ndim=None, shape=None, dtype=None):
        nameUnits = world.recv(source=source)
        name, units = nameUnits.split(',')
        tmp = myMPI.Irecv(source=source, world=world,
                          ndim=ndim, shape=shape, dtype=dtype)
        return cls(tmp, name, units)

    def IsendToLeft(self, world):
        """ISend an array to the rank left of world.rank.

        """
        dest = world.size - 1 if world.rank == 0 else world.rank - 1
        self.Isend(dest=dest, world=world)

    def IsendToRight(self, world):
        """ISend an array to the rank right of world.rank.

        """
        dest = 0 if world.rank == world.size - 1 else world.rank + 1
        self.Isend(dest=dest, world=world)

    def IrecvFromRight(self, world, wrap=True):
        """IRecv an array from the rank right of world.rank.

        """
        source = 0 if world.rank == world.size - 1 else world.rank + 1
        return self.Irecv(source=source, world=world)

    def IrecvFromLeft(self, world, wrap=True):
        """Irecv an array from the rank left of world.rank.

        """
        source = world.size - 1 if world.rank == 0 else world.rank - 1
        return self.Irecv(source=source, world=world)
