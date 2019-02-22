from copy import copy, deepcopy
import numpy as np

import matplotlib.pyplot as plt
from ...base import customFunctions as cf
from ...base import customPlots as cP
from ..statistics.Distribution import Distribution
from ..statistics.baseDistribution import baseDistribution
from ...base.customFunctions import str_to_raw, isIntorSlice
from .myObject import myObject
from ...base.HDF.hdfWrite import writeNumpy
from ...base import MPI as myMPI

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold


class StatArray(np.ndarray, myObject):
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
    >>> x = StatArray(np.arange(10), name='test', units='units')
    >>> print(x.mean())
    4.5

    If the StatArray is passed to a numpy function that does not return a new instantiation, a StatArray will be returned (as opposed to a numpy array)

    >>> np.delete(x, 5)
    StatArray([0, 1, 2, 3, 4, 6, 7, 8, 9])

    However, if you pass a StatArray to a numpy function that is not in-place, i.e. creates new memory, the return type will be a numpy array and NOT a StatArray subclass

    >>> np.append(x,[3,4,5])
    array([0, 1, 2, ..., 3, 4, 5])

    See Also
    --------
    geobipy.src.classes.statistics.Distribution : For possible prior and proposal distributions

    """


    def __new__(subtype, shape, name=None, units=None, **kwargs):
        """Instantiate a new StatArray """
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__


        if (not name is None):
            assert (isinstance(name,str)), TypeError('name must be a string')
            name=str_to_raw(name) # Do some possible LateX checking. some Backslash operatores in LateX do not pass correctly as strings
        if (not units is None):
            assert (isinstance(units,str)), TypeError('units must be a string')
            units=str_to_raw(units) # Do some possible LateX checking. some Backslash operatores in LateX do not pass correctly as strings

        # Copies a StatArray but can reassign the name and units
        if isinstance(shape, StatArray):
            shp = np.shape(shape)
            tmp = StatArray(shp, name=shape.name, units=shape.units) + shape

            if not name is None:
                tmp.name = name
            if not units is None:
                tmp.units = units

            if (shape.hasPrior()):
                tmp._prior = shape.prior.deepcopy()
            if (shape.hasProposal()):
                tmp._proposal = shape.proposal.deepcopy()            

            return tmp
        # Can pass in a numpy function call like arange(10) as the first argument
        elif isinstance(shape, np.ndarray):
            self = shape.view(StatArray)

        elif isinstance(shape, float):
            self = np.ndarray.__new__(subtype, 1, **kwargs)
            self[:] = shape
        # Convert the entry to a numpy array
        else:
            self = np.ndarray.__new__(subtype, np.asarray(shape), **kwargs)
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
            self.name = d.get('name')
            self.units = d.get('units')
        except:
            self.name = None
            self.units = None


    def __array_wrap__(self, out_arr, context = None):
        return np.ndarray.__array_wrap__(self, out_arr, context)


    def hasLabels(self):
        return not self.getNameUnits() == ""


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


    def copy( self, order='F'):       
        return StatArray(self)


    def deepcopy(self):
        """Create a deepcopy

        Returns
        -------
        out : StatArray
            Deepcopy of StatArray

        """
        return deepcopy(self)


    def __deepcopy__(self, memo):
        other = StatArray(self.shape, dtype=self.dtype)
        if (np.ndim(self) == 1):
            other[:] = self[:]
        elif (np.ndim(self) == 2):
            other[:, :] = self[:, :]

        other.name = self.name
        other.units = self.units

        if (self.hasPrior()):
            other._prior = self.prior.deepcopy()
        if (self.hasProposal()):
            other._proposal = self.proposal.deepcopy()
        
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
        tmp = np.delete(self, i, axis=axis)
        out = self.resize(tmp.shape)
        out[:] = tmp[:]
        return out


    def edges(self, min=None, max=None):
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
        Returns
        -------
        out : StatArray
            Edges of the StatArray

        """
        assert (self.size > 1), ValueError("Size of StatArray must be > 1")
        d = 0.5 * np.diff(self)
        edges = self.resize(self.size + 1)
        # Middle values
        edges[1:-1] = self[:-1] + d
        # Edge values
        edges[0] = self[0] - d[0] if min is None else min
        edges[-1] = self[-1] + d[-1] if max is None else max

        return StatArray(edges, self.name, self.units)


    def firstNonZero(self, axis=0, invalid_val=-1):
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
        return np.where(msk.any(axis=axis), msk.argmax(axis=axis), invalid_val)


    def getNameUnits(self):
        """Get the name and units

        Gets the name and units attached to the StatArray. Units, if present are surrounded by parentheses

        Returns
        -------
        out : str
            String containing name(units).

        """
        out = self.getName()
        u = self.getUnits()
        return out if u == "" else "{} ({})".format(out, u)


    def getName(self):
        """Get the name of the StatArray

        If the name has not been attached, returns an empty string

        Returns
        -------
        out : str
            The name of the StatArray.

        """

        return "" if self.name is None else self.name


    def getUnits(self):
        """Get the units of the StatArray

        If the units have not been attached, returns an empty string

        Returns
        -------
        out : str
            The unist of the StatArray

        """

        return "" if self.units is None else self.units


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

        tmp = np.insert(self, i, values, axis)
        out = self.resize(tmp.shape) # Keeps the prior and proposal if set.
        out[:] = tmp[:]
        return out


    def internalEdges(self):
        """Get the midpoint values between elements in the StatArray

        Returns an size(self) + 1 length StatArray of the midpoints between each element

        Returns
        -------
        out : StatArray
            Edges of the StatArray

        """
        assert (self.size > 1), ValueError("Size of StatArray must be > 1")
        d = 0.5*np.diff(self)
        edges = self.resize(self.size - 1)
        edges[:] = self[:-1] + d
        return edges


    def hasPrior(self):
        """Check that the StatArray has an attached prior.

        Returns
        -------
        out : bool
            Has an attached prior.

        """

        try:
            return not self.prior is None
        except:
            return False


    def hasProposal(self):
        """Check that the StatArray has an attached proposal.

        Returns
        -------
        out : bool
            Has an attached proposal.

        """

        try:
            return not self.proposal is None
        except:
            return False

        
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
        val = self.shape[axis] - np.flip(msk, axis=axis).argmax(axis=axis)
        return np.where(msk.any(axis=axis), val, invalid_val)


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

        try:
            i = np.zeros(values.size)
        except:
            i = 0
        return self.insert(i, values, axis=axis)


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
        
        if (np.all(np.shape(self) == new_shape)):
            return self.deepcopy()
        out = StatArray(np.resize(self, new_shape), self.name, self.units)
        if self.hasPrior():
            out._prior = self.prior.deepcopy()
        if self.hasProposal():
            out._proposal = self.proposal.deepcopy()

        return out


    def summary(self, out=False):
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
        np.set_printoptions(threshold=5)
        msg = "Name:  " + self.getName() + '\n'
        msg += "    Units: " + self.getUnits() + '\n'
        msg += "    Shape: " + str(self.shape) + '\n'
        msg += "   Values: " + str(self[:]) + '\n'
        if self.hasPrior():
            msg += "Prior: \n"
            msg += self.prior.summary(True)
        else:
            msg += "No attached prior \n"
        
        if self.hasProposal():
            msg += "Proposal: \n"
            msg += self.proposal.summary(True)
        else:
            msg += "No attached proposal \n"

        if (out):
            return msg
        print(msg)


    def verbose(self):
        """Explicit print of every element """
        np.set_printoptions(threshold=self.size)
        print(self[:])
        np.set_printoptions(threshold=5)


    def isRegular(self):
        """Checks that the values change regularly

        Returns
        -------
        out : bool
            Is regularly changing.

        """
        tmp = np.diff(self)
        return np.allclose(tmp, tmp[0])


    ### Statistical Routines

    def hist(self, bins=10, range=None, normed=None, weights=None, density=None, **kwargs):
        """Plot a histogram of the StatArray

        Plots a histogram, estimates the mean and standard deviation and overlays the PDF of a normal distribution with those values, if density=1.

        See Also
        --------
        geobipy.customPlots.hist : For geobipy additional arguments
        matplotlib.pyplot.hist : For histogram related arguments

        Example
        -------
        >>> from geobipy import StatArray
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> x = StatArray(np.random.randn(1000), name='name', units='units')
        >>> plt.figure()
        >>> x.hist()
        >>> plt.show()

        """
        cnts, bins = np.histogram(self, bins=bins, range=range, normed=normed, weights=weights, density=density)
        bins = StatArray(bins, name=self.name, units=self.units)
        cP.hist(cnts, bins, **kwargs)


    def pad(self, N):
        """ Copies the properties of a StatArray including all priors or proposals, but pads everything to the given size
        
        Parameters
        ----------
        N : int
            Size to pad to.

        Returns
        -------
        out
            StatArray
        
        """
        tmp = StatArray(N,name=self.name,units=self.units)
        try:
            pTmp = self.prior.pad(N)
            tmp._prior = pTmp
        except:
            pass
        try:
            pTmp = self.proposal.pad(N)
            tmp._proposal = pTmp
        except:
            pass
        return tmp

    def perturb(self, i=None, relative=False):
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
        if self.proposal.multivariate:
            tmp = self.proposal.rng(1)
        else:
            tmp = self.proposal.rng(self.size)

        if (i is None):
            i = np.arange(self.size)
        if relative:
            self[i] += tmp[i]
        else:
            self[i] = tmp[i]


    def probability(self, *args, i=None):
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

        assert (self.hasPrior()), TypeError('No prior defined on variable ' + self.name + '. Use StatArray.setPrior()')

        if (len(args) > 0):
            return self.prior.probability(*args)

        return self.prior.probability(self[:]) if i is None else self.prior.probability(self[i])

    @property
    def prior(self):
        """Returns the prior if available. """

        try:
            return self._prior
        except:
            return None


    @property
    def proposal(self):
        """Returns the prior if available. """

        try:
            return self._proposal
        except:
            return None


    def setPrior(self, distributionType, *args, **kwargs):
        """Set a prior distribution

        Sets a prior by interfacing through the Distribution method rather than passing any subclasses of the baseDistribution class.

        Parameters
        ----------
        distributionType : str
            The name of the distribution to set.
        \*args
            Variable length argument list.
        \*\*kwargs
            Arbitrary keyword arguments.

        See Also
        --------
        geobipy.src.classes.statistics.Distribution : For available distributions to instantiate.

        """
        self._prior = Distribution(distributionType, *args, **kwargs)


    def setProposal(self, distributionType, *args, **kwargs):
        """Set a proposal distribution

        Sets a proposal by interfacing through the Distribution method rather than passing any subclasses of the baseDistribution class.

        Parameters
        ----------
        distributionType : str
            The name of the distribution to set
        \*args
            Variable length argument list.
        \*\*kwargs
            Arbitrary keyword arguments.

        See Also
        --------
        geobipy.src.classes.statistics.Distribution : For available distributions to instantiate

        """
        self._proposal = Distribution(distributionType, *args, **kwargs)


    ### Plotting Routines

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
        if (i is None):
            i = np.size(self)
        if (x is None):
            x = StatArray(np.arange(i), name="Array index")

        return cP.bar(self, x, i, **kwargs)


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
            Turn off the colour bar, useful if multiple customPlots plotting routines are used on the same figure.   
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis. 

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
            assert (isinstance(y, StatArray)), TypeError("y must be a StatArray")
            if y.size == self.size:
                try:
                    my = y.edges()
                except:
                    pass

        if (self.ndim == 1):
            ax = cP.pcolor_1D(self, y=my, **kwargs)
        else:
            mx = x
            if (not x is None):
                assert (isinstance(x, StatArray)), TypeError("x must be a StatArray")
                if x.size == self.size:
                    try:
                        mx = x.edges()
                    except:
                        pass

            ax = cP.pcolor(self, x=mx, y=my, **kwargs)
        return ax


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

        if (not i is None): assert (isinstance(i,(slice,tuple))), TypeError("i must be a slice, use np.s_[]")

        if (self.ndim == 1):
            if (x is None): x=StatArray(np.arange(self.size),'Array Index')
            if (i is None): i=np.s_[:self.size]
            j=i
        else:
            if (x is None): x=StatArray(np.arange(self.shape[axis]),'Array Index')
            if (i is None): i=np.s_[:self.shape[0], :self.shape[1]]
            j = i[axis]

        cP.plot(x[j],self[i],**kwargs)


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
        geobipy.customPlots.Scatter2D : For additional keyword arguments you may use.

        """

        assert np.ndim(self) == 1, TypeError('scatter only works with a 1D array')

        if (x is None):
            x = StatArray(np.arange(self.size),'Array Index')
        else:
            assert x.size == self.size, ValueError('x must be size '+str(self.size))

        cP.scatter2D(x=x, y=y, c=self, i=i, **kwargs)


    def stackedAreaPlot(self, x=None, i=None, axis=0, labels=[], colors=cP.tatarize, **kwargs):
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

        assert (self.ndim == 2), TypeError('stackedAreaPlot only works with 2D arrays')
        if (not i is None): 
            assert (isinstance(i,(slice,tuple))), TypeError("i must be a slice, use np.s_[]")

        if (x is None):
            x = StatArray(np.arange(self.shape[axis]),'Array Index')
        else:
            assert x.size == self.shape[axis], ValueError('x must be size '+str(self.shape[axis]))


        if (i is None): i=np.s_[:self.shape[0], :self.shape[1]]
        j = i[axis]

        if (axis == 0):
            tmp = self.T
        else:
            tmp = self

        ma = tmp.copy()
        ma[ma >= 0.0] = 0.0
        cP.stackplot2D(x[j], ma[i], labels=labels, colors=colors, **kwargs)
        ma[:] = tmp[:]
        ma[ma <= 0.0] = 0.0
        cP.stackplot2D(x[j], ma[i], labels=[], colors=colors, **kwargs)


    ### HDF Routines

    def hdfName(self):
        """Create a string that describes class instantiation

        Returns a string that should be used as an attr['repr'] in a HDF group.  
        This allows reading of the attribute from the hdf file, evaluating it to return an object, 
        and then reading the hdf contents via the object's methods.

        Returns
        -------
        out
            str

        """
        name = self.getName()
        units = self.getUnits()
        if (not name is None):
            sName = ',"' + name + '"'
        else:
            sName = ''
        if (not units is None):
            sUnits = ',"' + units + '"'
        else:
            sUnits = ''

        return(r'StatArray(' + str(self.shape) + sName + sUnits + ',dtype=np.' + str(self.dtype) + ')')


    def toHdf(self, h5obj, myName):
        """Write the StatArray to an HDF object

        Creates and writes a new group in a HDF file under h5obj. 
        A nested heirarchy will be created e.g., myName\data, myName\prior, and myName\proposal.  
        This function modifies the file metadata and writes the contents at the same time and 
        should not be used in a parallel environment.

        Parameters
        ----------
        h5obj : h5py._hl.files.File or h5py._hl.group.Group
            A HDF file or group object to write the contents to.
        myName : str
            The name of the group to write the StatArray to.

        Examples
        --------
        >>> import h5py
        >>> from geobipy.src.classes.core.StatArray import StatArray
        >>> import numpy as np
        >>> x = StatArray(np.arange(10))
        >>> with h5py.File('test','w') as f:
        >>>     x.toHdf(f, 'aTestGroup')

        """
        # Create a new group inside h5obj
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        grp.create_dataset('data', data=self)
        #compression="gzip",compression_opts=6,shuffle=True
        try:
            self.prior.toHdf(grp,'prior')
        except:
            pass
        try:
            self.proposal.toHdf(grp,'proposal')
        except:
            pass


    def createHdf(self, h5obj, myName, nRepeats=None, fillvalue=None):
        """Create the Metadata for a StatArray in a HDF file

        Creates a new group in a HDF file under h5obj. 
        A nested heirarchy will be created e.g., myName\data, myName\prior, and myName\proposal. 
        This method can be used in an MPI parallel environment, if so however, a) the hdf file must have been opened with the mpio driver, 
        and b) createHdf must be called collectively, i.e., called by every core in the MPI communicator that was used to open the file. 
        In order to create large amounts of empty space before writing to it in parallel, the nRepeats parameter will extend the memory 
        in the first dimension.

        Parameters
        ----------
        h5obj : h5py._hl.files.File or h5py._hl.group.Group
            A HDF file or group object to create the contents in.
        myName : str
            The name of the group to create.
        nRepeats : int, optional
            Inserts a first dimension into the shape of the StatArray of length nRepeats. This can be used to extend the available memory of 
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
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        if (not nRepeats is None):
            if (self.size == 1):
                grp.create_dataset('data', [nRepeats], dtype=self.dtype, fillvalue=fillvalue)
            else:
                grp.create_dataset('data', [nRepeats,*self.shape], dtype=self.dtype, fillvalue=fillvalue)
        else:
            grp.create_dataset('data', self.shape, dtype=self.dtype, fillvalue=fillvalue)


    def writeHdf(self, h5obj, myName, index=None):
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
        writeNumpy(self, h5obj, myName+'/data', index=index)
#        try:
#            self.prior.writeHdf(h5obj,myName+'/prior',create=False)
#        except:
#            pass
#        try:
#            self.proposal.writeHdf(h5obj,myName+'/proposal',create=False)
#        except:
#            pass


    def fromHdf(self, h5grp, index=None):
        """Read the StatArray from a HDF group

        Given the HDF group object, read the contents into a StatArray.

        Parameters
        ----------
        h5obj : h5py._hl.group.Group
            A HDF group object to write the contents to.
        index : slice, optional
            If the group was created using the nRepeats option, index specifies the index'th entry from which to read the data.

        """

        if (index is None):
            try:
                return StatArray(np.atleast_1d(h5grp.get('data')), self.name, self.units)
            except:
                assert False, ValueError("HDF data was created as a larger array, specify the row index to read from")
        else:
            assert isIntorSlice(index), TypeError('index must be an int, slice, or tuple with slices. e.g. use index = np.s_[1,4:5,:] ')
            return StatArray(np.atleast_1d(h5grp.get('data')[index]), self.name, self.units)
#        try:
#            grp = h5grp.get('prior')
#            self.prior = eval(grp.attrs.get('repr'))
#            if (not self.prior is None):
#                self.prior = self.prior.fromHdf(grp, )
#        except:
#            pass
#        try:
#            grp = h5grp.get('proposal')
#            self.proposal = eval(grp.attrs.get('repr'))
#            if (not self.proposal is None):
#                self.proposal = self.proposal.fromHdf(grp)
#        except:
#            pass
        return self

    ### Classification Routines

    def kMeans(self, nClusters, standardize=False, nIterations=10, plot=False, **kwargs):
        """ Perform K-Means clustering on the StatArray """
        if (standardize):
            tmp = scale(self)
        else:
            tmp = self

        kmeans = KMeans(init='k-means++', n_clusters=nClusters, n_init=nIterations)
        kmeans.fit(tmp)
        clusterID = StatArray(kmeans.predict(tmp),name = 'cluster ID')

        if (plot):
            if (self.ndim == 1):
                cP.myscatter2D(self, clusterID, **kwargs)
            elif (self.ndim == 2):
                cP.myscatter2D(self[:,0], self[:,1], c=clusterID, **kwargs)
            else:
                # Going to have to do PCA or something and plot on two primary axes?
                # I.e. do a dimensionality reduction
                raise NotImplementedError

        return clusterID, kmeans


    def gMM(self, clusterID, trainPercent=75.0, covType=['spherical'], plot=True):
        """ Use a Gaussian Mixing Model to classify the data.
        clusterID is the initial assignment of the rows to their clusters """

        assert clusterID.size == self.shape[0], "size of clusterID "+str(clusterID.size)+" must match first dimension size "+str(self.shape[0])

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

        nClusters = np.unique(clusterID).size

        # Try GMMs using different types of covariances.
        models = dict((cov_type, GaussianMixture(n_components=nClusters,
                           covariance_type=cov_type, max_iter=20, random_state=None))
                          for cov_type in covType)

        nModels = len(models)

        if (plot):
            plt.figure(figsize=(3 * nModels // 2, 6))
            plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05, left=.01, right=.99)

        for index, (name, estimator) in enumerate(models.items()):
            # Since we have class labels for the training data, we can
            # initialize the GMM parameters in a supervised manner.
            estimator.means_init = np.array([xTrain[yTrain == i].mean(axis=0) for i in range(nClusters)])

            # Train the other parameters using the EM algorithm.
            estimator.fit(xTrain)

            if (plot):
                h = plt.subplot(2, nModels // 2, index + 1)
            #make_ellipses(estimator, h)


            #    tmp = x[z == n]
            #    c=np.zeros(tmp.shape[0])+n
                #cP.myscatter2D(tmp[:, 0], tmp[:, 1], s=0.8, c=c)
            #    plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color)#,label=ris.target_names[n])

            # Plot the test data with crosses
            #for n, color in enumerate(colors):
            #    data = X_test[y_test == n]
            #    plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

            y_train_pred = estimator.predict(xTrain)
            z = estimator.predict(self)
            if (plot):
                cP.myscatter2D(self[:,0],self[:,1],c=z)

            train_accuracy = np.mean(y_train_pred.ravel() == yTrain.ravel()) * 100
            if (plot):
                plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,transform=h.transAxes)

            y_test_pred = estimator.predict(xTest)
            test_accuracy = np.mean(y_test_pred.ravel() == yTest.ravel()) * 100
            if (plot):
                plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,transform=h.transAxes)

                #plt.xticks(())
                #plt.yticks(())
                cP.title(name)

    ### MPI Routines

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
        name = world.bcast(self.name, root=root)
        units = world.bcast(self.units, root=root)
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

        name = world.bcast(self.name)
        units = world.bcast(self.units)
        tmp = myMPI.Scatterv(self, starts, chunks, world, axis, root)
        return StatArray(tmp, name, units, dtype=tmp.dtype)


    def Isend(self, dest, world):
        
        world.isend(self.name, dest=dest)
        world.isend(self.units, dest=dest)
        req = myMPI.Isend(self, dest=dest, world=world)
        return req


    def Irecv(self, source, world):
        name = world.irecv(source=source).wait()
        units = world.irecv(source=source).wait()
        tmp = myMPI.Irecv(source=source, world=world)
        return StatArray(tmp, name, units)


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















