""" @RectilinearMesh1D_Class
Module describing a 1D Rectilinear Mesh class
"""
from copy import deepcopy

from numpy import arange, argwhere, array, asarray, atleast_1d
from numpy import cumsum, diag, diff, dot, exp,expand_dims, float64, full, hstack, inf, int_
from numpy import int32, int64, integer, interp, isclose, isinf, isnan, linspace, logical_not, kron
from numpy import maximum, mean, min, minimum, nan, ndim, ones, r_, repeat, s_, shape, sign
from numpy import size, sqrt, squeeze, where, zeros
from numpy import log as nplog
from numpy import all as npall
from numpy import any as npany

from numpy.ma import masked_invalid, mask_or

from .Mesh import Mesh
from ..core.DataArray import DataArray
from ..statistics.StatArray import StatArray

from ...base import utilities
from ...base import plotting as cp
from ..statistics.baseDistribution import baseDistribution
from ..statistics.Distribution import Distribution
from ..statistics.CategoricalDistribution import Categorical
from ..statistics import Histogram
from scipy.sparse import diags
from scipy import interpolate
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

brodie = False

class RectilinearMesh1D(Mesh):
    """Class defining a 1D rectilinear mesh with cell centres and edges.

    Contains a simple 1D mesh with cell edges, widths, and centre locations.

    RectilinearMesh1D(centres, edges, edgesMin, edgesMax)

    Parameters
    ----------
    centres : geobipy.DataArray, optional
        The locations of the centre of each cell. Only centres, edges, or widths can be given.
    edges : geobipy.DataArray, optional
        The locations of the edges of each cell, including the outermost edges. Only centres, edges, or widths can be given.
    widths : geobipy.DataArray, optional
        The widths of the cells.
    log : 'e' or float, optional
        Entries are given in linear space, but internally cells are logged.
        Plotting is in log space.
    relative_to : float, optional
        If a float is given, updates will be relative to this value.

    Returns
    -------
    out : RectilinearMesh1D
        The 1D mesh.

    Raises
    ------
    Exception
        If both centres and edges are given.
    TypeError
        centres must be a geobipy.DataArray.
    TypeError
        edges must be a geobipy.DataArray.

    """
    __slots__ = ('_centres', '_widths', '_edges', '_min_width', '_min_edge', '_max_edge', '_dimension',
                 '_max_cells', '_event_proposal', '_action', '_nCells', '_relative_to', '_log')
    def __init__(self, centres=None, edges=None, widths=None, log=None, relative_to=None, dimension=0):
        """ Initialize a 1D Rectilinear Mesh"""
        self._centres = None
        self._edges = None
        self._widths = None
        assert not ((centres is None) & (edges is None) & (widths is None)), ValueError("Must specify either centres, edges, or widths")
        self.dimension = dimension

        self.log = log
        self.relative_to = relative_to

        self.nCells = None

        # assert (not(not centres is None and not edges is None)), Exception('Cannot instantiate with both centres and edges values')
        if not widths is None:
            self.widths = widths
        else:
            if (not centres is None) and (edges is None):
                self.centres = centres
            elif (centres is None) and (not edges is None):
                self.edges = edges
            else:
                self.centres = centres
                self._edges = edges

        # Instantiate extra parameters for Markov chain perturbations.
        self._min_width = None
        self._min_edge = None
        self._max_edge = None
        self._max_cells = None
        # Categorical distribution for choosing perturbation events
        self._event_proposal = None
        # Keep track of actions made to the mesh.
        self._action = ['none', 0, 0.0]

    def __deepcopy__(self, memo={}):
        out = type(self).__new__(type(self))
        out._nCells = deepcopy(self._nCells, memo=memo)
        out.log = deepcopy(self.log, memo=memo)
        out._relative_to = deepcopy(self._relative_to, memo=memo)

        out._centres = deepcopy(self._centres, memo=memo)
        out._edges = deepcopy(self._edges, memo=memo)
        out._widths = deepcopy(self._widths, memo=memo)

        out._min_width = self.min_width
        out._min_edge = self.min_edge
        out._max_edge = self.max_edge
        out._max_cells = self.max_cells

        out._event_proposal = self.event_proposal
        out._action = deepcopy(self.action, memo=memo)
        out._dimension = self.dimension

        return out

    def __getitem__(self, slic):
        """Slice into the class. """

        # if shape(slic) == ():#, ValueError("slic must have one dimension.")
        #     return deepcopy(self)

        s2stop = None
        if isinstance(slic, slice):
            if not slic.stop is None:
                s2stop = slic.stop + 1 if slic.stop > 0 else slic.stop
            slic = slice(slic.start, s2stop, slic.step)
        else:
            slic = slice(slic, slic + 2, 1)

        tmp = self.edges[slic]
        assert tmp.size > 1, ValueError("slic must contain at least one cell.")
        out = type(self)(edges=tmp)

        out.log = self.log
        if self._relative_to is not None:
            out._relative_to = deepcopy(self._relative_to)
        out.dimension = self.dimension

        return out

    def __add__(self, other):
        return RectilinearMesh1D(edges=self.centres + other)

    def __sub__(self, other):
        return RectilinearMesh1D(edges=self.centres - other)

    def __mul__(self, other):
        return RectilinearMesh1D(edges=self.centres * other)

    def __truediv__(self, other):
        return RectilinearMesh1D(edges=self.centres / other)

    @property
    def action(self):
        return self._action

    @property
    def addressof(self):
        msg =  '{}: {}\n'.format(type(self).__name__, hex(id(self)))
        msg += "Edges:\n{}".format(("|   "+self.edges.addressof.replace("\n", "\n|   "))[:-4])
        msg += "Centres:\n{}".format(("|   "+self.centres.addressof.replace("\n", "\n|   "))[:-4])
        msg += "Widths:\n{}".format(("|   "+self.widths.addressof.replace("\n", "\n|   "))[:-4])
        msg += "Relative to:\n{}".format(("|   "+self.relative_to.addressof.replace("\n", "\n|   "))[:-4])
        return msg

    @property
    def area(self):
        return self.widths

    @property
    def bounds(self):
        return r_[self.edges[0], self.edges[-1]]

    @property
    def centres(self):
        return self._centres

    @property
    def centres_absolute(self):
        if self.relative_to.size == 1:
            return utilities._power(self.centres + self.relative_to, self.log)
        else:
            dims = arange(ndim(self.relative_to) + 1)
            dims = tuple(dims[dims != self.dimension])
            return utilities._power( expand_dims(self.centres, dims) + repeat(expand_dims(self.relative_to, self.dimension), self.nCells, self.dimension), self.log)

    @centres.setter
    def centres(self, values):

        values = StatArray(values)

        # assert values.sorted, ValueError("centres must be monotonically increasing")

        values, _ = utilities._log(values, log=self.log)

        if self.relative_to.size == 1:
            values -= self.relative_to

        self._centres = values
        self._edges = self._centres.edges()
        self._widths = self._edges.diff()

        if self._nCells is not None:
            self._nCells[0] = self.centres.size

    @property
    def centreTocentre(self):
        return diff(self.centres)

    @property
    def displayLimits(self):
        dx = 0.02 * self.range
        return (self.plotting_edges[0] - dx, self.plotting_edges[-1] + dx)

    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, value):
        self._dimension = int32(value)

    @property
    def edges(self):
        return self._edges

    @property
    def edges_absolute(self):
        if self.relative_to.size == 1:
            out = utilities._power(self.edges + self.relative_to, self.log)
        else:
            dims = arange(ndim(self.relative_to) + 1)
            dims = tuple(dims[dims != self.dimension])
            out = utilities._power(expand_dims(self.edges, dims) + repeat(expand_dims(self.relative_to, self.dimension), self.nEdges, self.dimension), self.log)

        return out

    @edges.setter
    def edges(self, values):
        values = StatArray(values)

        # assert values.sorted, ValueError("edges must be monotonically increasing")

        values, _ = utilities._log(values, log=self.log)

        if self.relative_to.size == 1:
            values -= self.relative_to

        self._edges = values
        self._centres = values.internalEdges()
        self._widths = values.diff()

        if self._nCells is not None:
            self._nCells[0] = self.centres.size

    @property
    def edge_to_edge(self):

        large = 1e6

        a = self.edges[1]-large if self.open_left else self.edges[0]
        b = self.edges[-2]+large if self.open_right else self.edges[-1]

        return b - a

    @property
    def plotting_centres(self):
        return self.plotting_edges.internalEdges()

    @property
    def plotting_edges(self):
        out = self.edges_absolute
        if self.open_left:
            if self.min_edge is None:
                out[0] = 0.9 * out[1]
            else:
                out[0] = self.min_edge
        if self.open_right:
            if self.max_edge is None:
                out[-1] = 1.1 * out[-2]
            else:
                out[-1] = self.max_edge
        return out

    @property
    def event_proposal(self):
        return self._event_proposal

    @event_proposal.setter
    def event_proposal(self, value):
        assert isinstance(value, Categorical), TypeError("event_proposal must be a geobipy.Categorical distribution")
        self._event_proposal = value

    @property
    def internaledges(self):
        return self._edges[1:-1]

    @property
    def is_regular(self):
        return self.edges.isRegular()

    @property
    def label(self):
        return self.centres.label

    @property
    def log(self):
        return self._log

    @log.setter
    def log(self, value):

        self._log = None

        if value is not None:
            if isinstance(value, str):
                assert value == 'e', ValueError("log must be 'e' if not numeric")
            self._log = value

    @property
    def log_scale(self):
        return 'linear' if self.log is None else 'log'

    @property
    def max_cells(self):
        return self._max_cells

    @max_cells.setter
    def max_cells(self, value):
        assert value > 0, ValueError('max_cells must be > 0')
        self._max_cells = int32(value)

    @property
    def max_edge(self):
        return self._max_edge

    @max_edge.setter
    def max_edge(self, value):
        self._max_edge = float64(value)

    @property
    def min_edge(self):
        return self._min_edge

    @min_edge.setter
    def min_edge(self, value):
        self._min_edge = float64(value)

    @property
    def min_width(self):
        return self._min_width

    @min_width.setter
    def min_width(self, value):
        self._min_width = value
        if value is None:
            self._min_width = (self.max_edge - self.min_edge) / (2.0 * self.max_cells)
        if self._min_width > self.min_edge:
            self._min_edge = self._min_width

    @property
    def nCells(self):
        if self._nCells is None:
            return int32(self._edges.size) - 1
        return self._nCells

    @nCells.setter
    def nCells(self, value):
        if value is None:
            self._nCells = None
            return
        else:
            assert isinstance(value, (int, integer, StatArray)), TypeError("nCells must be an integer, or StatArray")
            assert size(value) == 1, ValueError("nCells must be scalar or length 1")
            assert (value >= 1), ValueError('nCells must >= 1')
            if isinstance(value, int):
                self._nCells = StatArray(1, '# of Cells', dtype=int32) + value
            else:
                self._nCells = deepcopy(value)

    @property
    def ndim(self):
        return 1

    @property
    def nEdges(self):
        return 0 if self._edges is None else self._edges.size

    @property
    def nNodes(self):
        return self.nEdges

    @property
    def name(self):
        return self._centres.name

    @property
    def open_left(self):
        return self.edges[0] == -inf

    @property
    def open_right(self):
        return self.edges[-1] == inf

    @property
    def range(self):
        """Get the difference between end edges."""
        return abs(self._edges[-1] - self._edges[0])

    @property
    def relative_to(self):
        if self._relative_to is None:
            return StatArray(0.0)
        return self._relative_to

    @relative_to.setter
    def relative_to(self, value):
        if value is None:
            self._relative_to = None
            return

        if npall(value > 0.0):
            value, _ = utilities._log(value, self.log)
        self._relative_to = StatArray(value)

    @property
    def size(self):
        return self.nCells.item()

    @property
    def shape(self):
        return (self.nCells.item(), )

    @property
    def units(self):
        return self._centres.units

    @property
    def widths(self):
        return abs(diff(self.edges))

    @widths.setter
    def widths(self, values):
        assert npall(values > 0.0), ValueError("widths must be entirely positive")

        # self._widths = values
        self.edges =  StatArray(hstack([0.0, cumsum(values)]), utilities.getName(values), utilities.getUnits(values))

    def add_axis(self, axis, ax=None, **kwargs):
        from .RectilinearMesh2D import RectilinearMesh2D

        assert 0 <= axis <= 1, ValueError("Invalid axis 0 <= axis <= 1")

        if ax is None:
            ax = RectilinearMesh1D(**kwargs)
        match axis:
            case 0:
                return RectilinearMesh2D(x=self, y=ax)
            case 1:
                return RectilinearMesh2D(x=ax, y=self)

    def axis(self, axis):
        return self

    def map_to_pdf(self, distribution, pdf, log=False, axis=0):
        """ Creates a Hitmap from the model given the variance of each layer.

        For each depth, creates a normal distribution with a mean equal to the interpolated parameter
        at that depth and variance specified with variance.

        Parameters
        ----------
        variance : array_like
            The variance of each layer
        Hitmap : geobipy.Hitmap
            Hitmap to convert the model to.
            Must be instantiated before calling so that the model can be interpolated correctly

        """
        assert (distribution.ndim == self.nCells), ValueError('size of variance must equal number of cells')

        ax = pdf.mesh.axis(axis)

        other = pdf.mesh.other_axis(axis)

        cells_per_layer = diff(ax.cellIndex(self.edges))
        pdfs = distribution.probability(other.centres_absolute, log=False)
        pdf.values = (repeat(pdfs, cells_per_layer, axis=pdf.mesh.ndim-axis-1)).T

        pdf.values = DataArray(pdf.values / (sum(pdf.values) * pdf.mesh.area), 'Density')
        assert isclose(sum(pdf.values * pdf.mesh.area), 1.0), Exception("pdf does not sum to 1.0")

        return pdf


    # def _credibleIntervals(self, values, percent=90.0, log=None, reciprocate=False, axis=0):
    #     """Gets the median and the credible intervals for the specified axis.

    #     Parameters
    #     ----------
    #     values : array_like
    #     Values to use to compute the intervals.
    #     percent : float
    #     Confidence percentage.
    #     log : 'e' or float, optional
    #     Take the log of the credible intervals to a base. 'e' if log = 'e', or a number e.g. log = 10.
    #     axis : int
    #     Along which axis to obtain the interval locations.

    #     Returns
    #     -------
    #     med : array_like
    #     Contains the medians along the specified axis. Has size equal to arr.shape[axis].
    #     low : array_like
    #     Contains the lower interval along the specified axis. Has size equal to arr.shape[axis].
    #     high : array_like
    #     Contains the upper interval along the specified axis. Has size equal to arr.shape[axis].

    #     """

    #     percent = 0.5 * minimum(percent, 100.0 - percent)

    #     tmp = self._percentile(values, r_[50.0, percent, 100.0-percent], log, reciprocate, axis)

    #     return tmp[0], tmp[1], tmp[2]

    # def _percentile(self, values, percent=95.0, log=None, reciprocate=False, axis=0):
    #     """Gets the percent interval along axis.

    #     Get the statistical interval, e.g. median is 50%.

    #     Parameters
    #     ----------
    #     values : array_like
    #         Valus used to compute interval like histogram counts.
    #     percent : float
    #         Interval percentage.  0.0 < percent < 100.0
    #     log : 'e' or float, optional
    #         Take the log of the interval to a base. 'e' if log = 'e', or a number e.g. log = 10.
    #     axis : int
    #         Along which axis to obtain the interval locations.

    #     Returns
    #     -------
    #     interval : array_like
    #         Contains the interval along the specified axis. Has size equal to self.shape[axis].

    #     """
    #     percent *= 0.01

    #     # total of the counts
    #     total = values.sum()
    #     # Cumulative sum
    #     cs = cumsum(values)
    #     # Cumulative "probability"
    #     tmp = divide(cs, total)
    #     # Find the interval
    #     i = searchsorted(tmp, percent)
    #     # Obtain the values at those locations
    #     out = self.centres[i]

    #     return out

    def cellIndex(self, values, clip=False, trim=False, **kwargs):
        """ Get the index to the cell that each value in values falls in.

        Parameters
        ----------
        values : array_like
            The values to find the cell indices for
        clip : bool
            A negative index which would normally wrap will clip to 0 and self.bins.size instead.
        trim : bool
            Do not include out of axis indices. Negates clip, since they wont be included in the output.

        Returns
        -------
        out : array_like
            The cell indices

        """

        edges = self.edges

        values, dum = utilities._log(atleast_1d(values).flatten(), self.log)

        if self.relative_to.size == 1:
            values = values - self.relative_to

        # Remove values that are out of bounds
        if trim:
            values = values[(values >= edges[0]) & (values < edges[-1])]

        reversed = False
        if self.edges[-1] < self.edges[0]:
            reversed = True
            edges = self.edges[::-1]

        # Get the bin indices for all values
        iBin = atleast_1d(edges.searchsorted(values, side='right') - 1)

        if reversed:
            iBin = self.nCells - iBin

        # Force out of bounds to be in bounds if we are clipping
        if clip:
            iBin = maximum(iBin, 0)
            iBin = minimum(iBin, self.nCells.item() - 1)
        # Make sure values outside the lower edge are -1
        else:
            if not trim:
                iBin[values < edges[0]] = -1
                iBin[values >= edges[-1]] = self.nCells.item()-1

        return squeeze(iBin)

    def cellIndices(self, *args, **kwargs):
        return self.cellIndex(*args, **kwargs)

    def _compute_probability(self, distribution, pdf, log=None, log_probability=False, axis=0, **kwargs):
        """Compute the marginal probability of a pdf with a distribution.

        Parameters
        ----------
        distribution : geobipy.distribution
            Distribution
        pdf : array_like
            PDF values usually taken from histogram.pdf.values
        log : 'e' or float, optional
            Take the log of the mesh centres to a base. 'e' if log = 'e', and a number e.g. log = 10.
        log_probability : bool, optional
            Compute log probability.

        Returns
        -------
        out : array_like
            Marginal probabilities

        """
        centres, _ = utilities._log(self.centres_absolute, log)

        shp = (distribution.ndim, self.nCells.item)

        probability = zeros(self.shape)
        p = distribution.probability(centres, log_probability)

        probability = p.T * pdf

        probability = probability / expand_dims(sum(probability, 0), axis=0)

        return DataArray(probability, name='marginal_probability')

    def create_pdf(self, distribution):

        assert isinstance(distribution, baseDistribution), TypeError('distribution must have type geobipy.baseDistribution')
        assert distribution.ndim == self.nCells, TypeError('distribution must have {} dimensions, is has {}'.format(self.nCells, distribution.ndim))

    def delete_edge(self, i, values=None):
        """Delete an edge from the mesh

        Parameters
        ----------
        i : int
            The edge to remove.

        Returns
        -------
        out : RectilinearMesh1D
            Mesh with edge removed.

        """

        if (self.nCells == 0):
            if values is None:
                return deepcopy(self)
            else:
                return deepcopy(self), deepcopy(values)

        assert 1 <= i <= (self.nEdges - 1), ValueError("Required  1 <= i <= {}".format(self.nEdges-1))

        # Deepcopy the Model to ensure priors and proposals are preserved
        out = deepcopy(self)
        # Remove the interface depth
        out.edges = out.edges.delete(i)

        out._action = ['delete', int32(i), squeeze(self.edges[i])]

        if values is not None:
            values = deepcopy(values)
            if isinstance(values, list):
                for j in range(len(values)):
                    tmp = values[j]
                    val = 0.5 * (tmp[i-1] + tmp[i])
                    values[j] = tmp.delete(i)
                    values[j][i-1] = val
            else:
                choice = self.event_proposal.prng.integers(low=0, high=3, size=1).item()
                match choice:
                    case 0:
                        val = values[i-1]
                    case 1:
                        val = 0.5 * (values[i-1] + values[i])
                    case 2:
                         val = values[i]
                values = values.delete(i)
                values[i-1] = val
            return out, values

        return out

    def gradient(self, values):
        r"""Compute the gradient

        Parameter gradient :math:`\nabla_{z}\sigma` at the ith layer is computed via

        .. math::
            :label: dpdz1

            \nabla_{z}^{i}\sigma = \frac{\sigma_{i+1} - \sigma_{i}}{h_{i} - h_{min}}

        where :math:`\sigma_{i+1}` and :math:`\sigma_{i}` are the log-parameters on either side of an interface, :math:`h_{i}` is the log-thickness of the ith layer, and :math:`h_{min}` is the minimum log thickness defined by

        .. math::
            :label: minThickness1

            h_{min} = \frac{z_{max} - z_{min}}{2 k_{max}}

        where :math:`k_{max}` is a maximum number of layers, set to be far greater than the expected final solution.

        """
        if self.nCells.item() > 1:
            # return diff(nplog(values)) / (nplog(self.widths[:-1]) - nplog(self.min_width))
            return diff(nplog(values)) / (nplog(self.widths[:-1]))# - nplog(self.min_width))

    @property
    def cell_weights(self):

        if self.nCells == 1:
            return ones((1, 1))

        x = self.widths.copy()
        e2e = self.edge_to_edge

        # Sort out infinity here
        if self.open_left:
            if self.nCells == 2:
                x[0] = x[-1]
            else:
                if brodie:
                    x[0] = (x[1]**2.0 / x[2])
                else:
                    x[0] = x[1] - e2e

        if self.open_right:
            if self.nCells == 2:
                x[-1] = x[0]
            else:
                if brodie:
                    x[-1] = (x[-2]**2.0 / x[-3])
                else:
                    x[-1] = x[-2] + e2e

        out = x/(self.nCells * mean(x)) if brodie else x/(x.sum())
        return diag(out)

    @property
    def gradient_operator(self):

        if self.nCells == 1:
            return ones((1, 1))

        x = self.widths.copy()

        e2e = self.edge_to_edge

        # Sort out infinity here
        if self.open_left:
            if self.nCells == 2:
                x[0] = x[-1]
            else:
                if brodie:
                    x[0] = x[1]**2.0 / x[2]
                else:
                    x[0] = x[1] - e2e

        if self.open_right:
            if self.nCells == 2:
                x[-1] = x[0]
            else:
                if brodie:
                    x[-1] = x[-2]**2.0 / x[-3]
                else:
                    x[-1] = x[-2] + e2e

        centre_to_centre = 0.5*(x[:-1] + x[1:])

        if brodie:
            s = sqrt(x / mean(x))
            tmp = s[1:] / (centre_to_centre * (self.nCells - 1))
        else:
            tmp = 1.0 / (centre_to_centre * (self.nCells - 1))
        out = diags([-tmp, tmp], [0, 1], shape=(self.nCells.item()-1, self.nCells.item())).toarray()

        return out

    def in_bounds(self, values):
        """Return whether values are inside the cell edges

        Parameters
        ----------
        values : array_like
            Check if these are inside left <= values < right.

        Returns
        -------
        out : bools
            Are the values inside.

        """
        values, _ = utilities._log(values, self.log)
        edges = self.edges_absolute
        return (values >= edges.min()) & (values < edges.max())

    def insert_edge(self, value, values=None):
        """Insert a new edge.

        Parameters
        ----------
        value : numpy.float64
            Location at which to insert a new edge

        Returns
        -------
        out : geobipy.RectilinearMesh1D
            Mesh with inserted edge.

        """
        # Get the index to insert the new layer
        i = self.edges.searchsorted(value)

        # Deepcopy the 1D Model
        out = deepcopy(self)

        # Insert the new layer depth
        out.edges = self.edges.insert(i, value)
        out._action = ['insert', int32(i), value]

        if values is not None:
            values = deepcopy(values)
            if isinstance(values, list):
                for j in range(len(values)):
                    values[j] = values[j].insert(i, values[j][i-1])
            else:
                values = values.insert(i, values[i-1])

            return out, values

        return out

    def is_left(self, value):
        value, _ = utilities._log(value, self.log)
        return value < self.edges[0]

    def is_right(self, value):
        value, _ = utilities._log(value, self.log)
        return value > self.edges[-1]

    def mask_cells(self, distance, values=None):
        """Mask cells by a distance.

        If the edges of the cell are further than distance away, extra cells are inserted such that
        the cell's new edges are at distance away from the centre.

        Parameters
        ----------
        distance : float
            Distance to mask
        values : array_like, optional
            If given, values will be remapped to the masked mesh.

        Returns
        -------
        out : RectilinearMesh1D
            Masked mesh
        indices : ints
            Location of the original centres in the expanded mesh
        out_values : array_like, optional
            If values is given, values will be remapped to the masked mesh.

        """
        w = self.widths
        distance2 = distance
        iBig = where(w >= distance)
        n_large = size(iBig)
        new_edges = full((self.nEdges + 2*n_large), fill_value=nan)
        indices = zeros(self.nCells.item(), dtype=int32)

        k = 0

        for i in range(self.nCells.item()):
            new_edges[k] = self.edges[i]
            k += 1
            modified = False
            if (abs(self.centres[i] - self.edges[i]) > distance2):
                new_edges[k] = self.centres[i] - distance2
                indices[i] = k-1
                modified = True
                k += 1
            else:
                indices[i] = k-1

            if (abs(self.edges[i+1] - self.centres[i]) > distance2):
                new_edges[k] = self.centres[i] + distance2
                indices[i] = k-1
                mdified = True
                k += 1
            else:
                indices[i] = k-1

        new_edges[k] = self.edges[-1]
        new_edges = new_edges[~isnan(new_edges)]
        out = RectilinearMesh1D(edges=new_edges)

        if values is not None:
            out_values = full(out.nCells.item(), fill_value=nan)
            out_values[indices] = values
            return out, indices, out_values

        return out, indices

    def pad(self, size):
        """Copies the properties of a mesh including all priors or proposals, but pads memory to the given size

        Parameters
        ----------
        size : int
            Create memory upto this size.

        Returns
        -------
        out : RectilinearMesh1D
            Padded mesg

        """
        out = deepcopy(self)
        # out._centres = self.centres.pad(size)
        out.edges = self.edges.pad(size+1)
        return out

    def pcolor(self, values, **kwargs):
        """Create a pseudocolour plot.

        Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

        Parameters
        ----------
        values : array_like
            The value of each cell.

        Other Parameters
        ----------------
        alpha : scalar or array_like, optional
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

        See Also
        --------
        geobipy.plotting.pcolor : For non matplotlib keywords.
        matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

        """

        # assert isinstance(values, DataArray), TypeError("arr must be a StatArray")
        assert values.size == self.nCells, ValueError(
            "arr must have size nCell {}".format(self.nCells))

        values = DataArray(values)

        kwargs['y'] = kwargs.pop('y', self.plotting_edges)

        if self.log is not None:
            if 'transpose' in kwargs:
                kwargs['yscale'] = 'log'
            else:
                kwargs['xscale'] = 'log'

        ax = values.pcolor(**kwargs)

        return ax

    def perturb(self, values=None):
        """Perturb the mesh

        Generates a new mesh by perturbing the current mesh based on four probabilities.
        The probabilities correspond to
        * Birth, the insertion of a new interface
        * Death, the deletion of an interface
        * Change, change one the existing interfaces
        * No change, do nothing and return the original

        The methods.set_priors and setProposals must be used before calling self.perturb.

        If an interface is created, or an interface perturbed, any resulting cell width must be greater than the minimum width :math:`h_{min}`.
        If the new cell width test fails, the birth or perturbation tries again. If the cycle fails after 10 tries, the entire process begins again
        such that a death, or no change is possible thus preventing any never-ending cycles.

        Returns
        -------
        out : RectilinearMesh1D
            The perturbed mesh

        See Also
        --------
        RectilinearMesh1D.set_priors : Must be used before calling self.perturb
        RectilinearMesh1D.setProposals : Must be used before calling self.perturb

        """
        assert (not self.event_proposal is None), ValueError('Please set the proposals with RectilinearMesh1D.setProposals()')

        # event_proposal = self.event_proposal
        # if probability is not None:
        #     event_proposal = Distribution('Categorical',
        #                                   probability,
        #                                   ['insert', 'delete', 'perturb', 'none'],
        #                                   prng=self.nCells.prior.prng)

        prng = self.event_proposal.prng

        nTries = 10
        # This outer loop will allow the perturbation to change types. e.g. if the loop is stuck in a birthing
        # cycle, the number of tries will exceed 10, and we can try a different perturbation type.
        tryAgain = True  # Temporary to enter the loop
        while (tryAgain):
            tryAgain = False
            goodAction = False

            # Choose an action to perform, and make sure its legitimate
            # i.e. don't delete a single layer mesh, or add a layer to a mesh that is at the priors max on number of cells.
            while not goodAction:
                goodAction = True
                # Get a random probability from 0-1
                event = self.event_proposal.rng()

                if ((self.nCells.item() == 1) and (event == 1 or event == 2)):
                    goodAction = False
                elif ((self.nCells.item() == self.nCells.prior.max) and (event == 0)):
                    goodAction = False

            # Return if no change
            if (event == 3):
                out = deepcopy(self)
                out._action = ['none', 0, 0.0]

                if values is not None:
                    return out, deepcopy(values)
                return out

            # Otherwise enter life-death-perturb cycle
            if (event == 0):  # Create a new layer
                suitable_width = False
                tries = 0
                while (not suitable_width):  # Continue while the new layer is smaller than the minimum
                    # Get the new edge
                    # new_edge = exp(prng.uniform(low=nplog(self.min_edge), high=nplog(self.max_edge), size=1))
                    # new_edge = prng.uniform(low=self.min_edge, high=self.max_edge, size=1)
                    new_edge = self.edges.proposal.rng(size=1)

                    # Insert the new depth
                    i = self.edges.searchsorted(new_edge)
                    z = self.edges.insert(i, new_edge)
                    # Get the thicknesses
                    h = min(diff(z))
                    tries += 1
                    suitable_width = (h > self.min_width)
                    if (tries == nTries):
                        suitable_width = True  # just to exit.
                        tryAgain = True
                if (not tryAgain):
                    return self.insert_edge(new_edge, values=values)

            if (event == 1):  # Remove an edge
                # Get the layer to remove
                iDeleted = int64(prng.uniform(low=0, high=self.nCells.item()-1, size=1)[0]) + 1
                # Remove the layer and return
                return self.delete_edge(iDeleted, values=values)

            if (event == 2):  # Perturb an edge
                suitable_width = False
                tries = 0
                while (not suitable_width):  # Continue while the perturbed layer is suitable
                    z = self.edges.copy()
                    # Get the internal edge to perturb
                    i = int32(prng.uniform(low=1, high=self.nEdges-1, size=1)[0])
                    # Get the perturbation amount
                    dz = sign(prng.normal()) * self.min_width * prng.uniform()
                    # Perturb the layer
                    z[i] += dz
                    # Get the minimum thickness
                    h = min(diff(z))
                    tries += 1

                    # Exit if the thickness is big enough, and we stayed within
                    # the depth bounds
                    suitable_width = ((h > self.min_width) and (z[1] > self.min_edge) and (z[-2] < self.max_edge))

                    if (tries == nTries):
                        suitable_width = True
                        tryAgain = True

                if (not tryAgain):
                    out = deepcopy(self)
                    out.edges = z
                    out._action = ['perturb', int32(i), dz]
                    if values is not None:
                        return out, deepcopy(values)
                    return out

        assert False, Exception("Should not be here, file a bug report....")

    def piecewise_constant_interpolate(self, values, other, bound=False, axis=0):
        """Interpolate values of the cells to another RectilinearMesh in a piecewise constant manner.

        Parameters
        ----------
        values : geobipy.StatArray
            The values to interpolate. Has size self.nCells
        mesh : geobipy.RectilinearMeshND for N = 1, 2, 3.
            A mesh to interpolate to.
            If 2D, axis must be given to specify which axis to interpolate against.
        bound : bool, optional
            Interpolated values above the top of the model are nan.
        axis : int, optional
            Axis to interpolate the value to.

        Returns
        -------
        out : array
            The interpolated values at the cell centres of the other mesh.

        """
        # assert isinstance(other, (RectilinearMesh1D, RectilinearMesh2D)), TypeError('mesh must be a RectilinearMesh1D or RectilinearMesh2D')

        if not isinstance(other, RectilinearMesh1D):
            other = other.axis(axis)

        edges = self.plotting_edges
        bounds = [maximum(other.edges[0], edges[0]), minimum(other.edges[-1], edges[-1])]

        y = other.centres

        if (self.nCells.item() == 1):
            out = interp(y, bounds, kron(values, [1, 1]))
        else:
            xp = kron(asarray(edges), [1, 1.000001])[1:-1]
            fp = kron(values, [1, 1])
            out = interp(y, xp, fp)

        if bound:
            i = where((y < bounds[0]) & (y > bounds[-1]))
            out[i] = nan

        return out

    def plot(self, values, **kwargs):
        """Plots values using the mesh as a line

        Parameters
        ----------
        reciprocateX : bool, optional
            Take the reciprocal of the x axis
        xscale : str, optional
            Scale the x axis? e.g. xscale = 'linear' or 'log'
        yscale : str, optional
            Scale the y axis? e.g. yscale = 'linear' or 'log'
        flipX : bool, optional
            Flip the X axis
        flipY : bool, optional
            Flip the Y axis
        noLabels : bool, optional
            Do not plot the labels

        """
        reciprocateX = kwargs.pop("reciprocateX", False)

        # Repeat the last entry since we are plotting against edges
        par = values.append(values[-1])
        if (reciprocateX):
            par = 1.0 / par

        kwargs.pop('line', None)

        ax, stp = cp.step(x=par, y=self.plotting_edges, **kwargs)
        # if self.hasHalfspace:
        #     h = 0.99*z[-1]
        #     if (self.nCells == 1):
        #         h = 0.99*self.max_edge
        #     plt.text(0, h, s=r'$\downarrow \infty$', fontsize=12)

    def bar(self, values, **kwargs):

        if self.log is not None:
            kwargs['xscale'] = 'log'

        return cp.bar(values, self.plotting_edges, **kwargs)

    def plot_grid(self, **kwargs):
        """ Plot the grid lines of the mesh.

        See Also
        --------
        geobipy.StatArray.pcolor : For additional plotting arguments

        """

        kwargs['grid'] = True
        kwargs['colorbar'] = False

        values = DataArray(full(self.nCells.item(), nan))

        self.pcolor(values=values, **kwargs)

        # if self.open_left:
        #     h = y[-1] if kwargs.get('flip', False) else y[0]
        #     s = 'down' if transpose else 'left'
        #     if transpose:
        #         plt.text(0, 0.99*h, s=r'$\{}arrow \infty$'.format(s))
        #     else:
        #         plt.text(0.99*h, 0, s=r'$\{}arrow \infty$'.format(s))

        # if self.open_right:
        #     h = y[-1] if kwargs.get('flip', False) else y[0]
        #     s = 'up' if transpose else 'right'
        #     if transpose:
        #         plt.text(0, 0.99*h, s=r'$\{}arrow \infty$'.format(s))
        #     else:
        #         plt.text(0.99*h, 0, s=r'$\{}arrow \infty$'.format(s))

    def plot_line(self, value, **kwargs):
        kwargs.pop('axis', None)

        geobipy_kwargs, kwargs = cp.filter_plotting_kwargs(kwargs)
        color_kwargs, kwargs = cp.filter_color_kwargs(kwargs)

        if value.size == 1:
            value = value.item()

        ax = geobipy_kwargs['ax']

        f = ax.axhline if geobipy_kwargs['transpose'] else ax.axvline

        linecolor = kwargs.pop('linecolor', 'y')

        if size(value) > 1:
            if isinstance(linecolor, list):
                [f(l, color=c, **kwargs) for l, c in zip(value, linecolor)]
            else:
                kwargs['color'] = linecolor
                [f(l, **kwargs) for l in value]
        else:
            kwargs['color'] = linecolor
            f(value, **kwargs)

    @property
    def n_posteriors(self):
        return sum([x.hasPosterior for x in [self.nCells, self.edges]])

    def _init_posterior_plots(self, gs, values=None, sharex=None, sharey=None):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """

        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        if values is None:
            splt = gs.subgridspec(2, 1)
            ax = {'ncells':plt.subplot(splt[0, 0]),
                  'edges': plt.subplot(splt[1, 0], sharey=sharey)}
        else:
            splt = gs.subgridspec(2, 1, height_ratios=[1, 4])
            ax = {'ncells': plt.subplot(splt[0, :])}

            splt2 = splt[1, :].subgridspec(1, 2, width_ratios=[2, 1])
            ax['edges'] = plt.subplot(splt2[1], sharey=sharey)
            if sharey is None:
                sharey = ax['edges']
            ax['values'] = plt.subplot(splt2[0], sharex=sharex, sharey=sharey)

        for k, a in ax.items():
            cp.pretty(a)

        return ax

    def plot_posteriors(self, axes=None, values=None, **kwargs):
        # assert len(axes) == 2, ValueError("Must have length 2 list of axes for the posteriors. self.init_posterior_plots can generate them")

        if axes is None:
            axes = kwargs.pop('fig', plt.gcf())

        if not isinstance(axes, dict):
            axes = self._init_posterior_plots(axes, values=values)

        assert len(axes) >= 2, ValueError("axes must have length >= 2")

        ncells_kwargs = kwargs.get('ncells_kwargs', {})
        edges_kwargs = kwargs.get('edges_kwargs', {})
        values_kwargs = kwargs.get('values_kwargs',{})

        overlay = kwargs.pop('overlay', None)
        if overlay is not None:
            tmp = overlay
            if values is not None:
                tmp = overlay.mesh
            ncells_kwargs['overlay'] = tmp.nCells
            edges_kwargs['overlay'] = tmp.edges

        if self.nCells.hasPosterior:
            ax = axes['ncells']; ax.cla()
            self.nCells.plot_posteriors(ax = ax, **ncells_kwargs)

        if self.edges.hasPosterior:
            ax = axes['edges']; ax.cla()
            self.edges.plot_posteriors(ax = ax, **edges_kwargs)

        if values is not None:
            ax = axes['values']
            values.plot_posteriors(ax=ax, **values_kwargs)

            if overlay is not None:
                overlay.plot(ax=ax, xscale=values_kwargs.get('xscale', 'linear'),
                        reciprocateX=values_kwargs.get('reciprocateX', None),
                        labels=False,
                        linewidth=1,
                        color=cp.wellSeparated[3])

            doi = values.posterior.opacity_level(percent=90.0, log=values_kwargs.get('logX', None), axis=values_kwargs.get('axis', 0))
            ax.axhline(doi, color = '#5046C8', linestyle = 'dashed', alpha = 0.6)
        return axes

    @property
    def probability(self):
        r"""Evaluate the prior probability for the mesh.

        The following equation describes the components of the prior that correspond to the Model1D,

        .. math::
            p(k | I)p(\boldsymbol{e}| k, I),

        where :math:`k, I, \boldsymbol{e}` are the number of cells, prior information, edge location respectively.

        The multiplication here can be turned into a summation by taking the log of the components.

        Parameters
        ----------
        components : bool, optional
            Return all components used in the final probability as well as the final probability

        Returns
        -------
        probability : numpy.float64
            The probability
        components : array_like, optional
            Return the components of the probability, i.e. the individually evaluated priors as a second return argument if comonents=True on input.

        """
        # Probability of the number of layers
        probability = self.nCells.probability(log=True)

        # Probability of depth given nCells
        # probability += self.edges.probability(x=self.nCells.item()-1, log=True)

        return probability

    def proposal_probability(self):

        prng = self.nCells.prior.prng
        action = self.action[0]

        proposal = 1.0
        proposal1 = 1.0
        if action == 'insert':
            k = self.nCells - 1

            forward = Distribution('Uniform', 0.0, self.remainingSpace(k), prng=prng)
            reverse = Distribution('Uniform', 0.0, k, prng=prng)

            proposal = reverse.probability(1, log=True)
            proposal1 = forward.probability(0.0, log=True)

        if action == 'delete':
            k = self.nCells

            forward = Distribution('Uniform', 0.0, self.remainingSpace(k), prng=prng)
            reverse = Distribution('Uniform', 0.0, k, prng=prng)

            proposal = forward.probability(0.0, log=True)
            proposal1 = reverse.probability(1, log=True)

        return proposal, proposal1

    def remainingSpace(self, n_cells):
        return (self.max_edge - self.min_edge) - (n_cells * self.min_width)

    def resample(self, dx, values, kind='cubic'):
        x = arange(self.edges[0], self.edges[-1]+dx, dx)

        mesh = RectilinearMesh1D(edges=x)

        f = interpolate.interp1d(self.centres, values, kind=kind, fill_value="extrapolate")
        return mesh, f(mesh.centres)

    def interpolate_centres_to_nodes(self, values, kind='linear', **kwargs):
        kwargs['fill_value'] = kwargs.pop('fill_value', 'extrapolate')
        if values.size < 4:
            kind = 'linear'

        if ndim(values) == 1:
            f = interpolate.interp1d(self.centres, values, kind=kind, **kwargs)
            return f(self.edges)
        elif ndim(values) == 2:
            raise NotImplementedError('Cant interpolate 2D things yet. ')

    def reset_posteriors(self):
        self.nCells.reset_posteriors()
        self.edges.reset_posteriors()

    def set_posteriors(self, nCells_posterior=None, edges_posterior=None, **kwargs):

        # Initialize the posterior histogram for the number of layers
        if nCells_posterior is None:
            mesh = RectilinearMesh1D(centres=DataArray(arange(0.0, self.max_cells + 1.0), name="# of Layers"))
            self.nCells.posterior = Histogram.Histogram(mesh=mesh)

        if edges_posterior is None:

            assert not self.max_cells is None, ValueError(
                "No priors are set, user self.set_priors().")

            number_of_edge_bins = kwargs.get('number_of_edge_bins', int(self.max_edge / (0.5 * self.min_width)))

            # Discretize the parameter values
            edges = DataArray(linspace(0.0, 1.01 * self.max_edge, number_of_edge_bins+1), self.edges.name, self.edges.units)

            mesh = RectilinearMesh1D(edges=edges)
            # Initialize the interface Depth Histogram
            self.edges.posterior = Histogram.Histogram(mesh=mesh)

    def set_priors(self, n_cells_prior=None, edges_prior=None, **kwargs):
        r"""Setup the priors of the mesh.

        By default the following priors are set unless explictly specified.

        **Prior on the number of cells**

        Uninformative prior using a uniform distribution.

        .. math::
            :label: layers

            p(k | I) =
            \begin{cases}
            \frac{1}{k_{max} - 1} & \quad 1 \leq k \leq k_{max} \newline
            0 & \quad otherwise
            \end{cases}.

        **Prior on the cell edges**

        We use order statistics for the prior on cell edges.

        .. math::
            :label: depth

            p(\boldsymbol{e} | k, I) = \frac{(k -1)!}{\prod_{i=0}^{k-1} \Delta e_{i}},

        where the numerator describes the number of ways that :math:`(k - 1)` interfaces can be ordered and
        :math:`\Delta e_{i} = (e_{max} - e_{min}) - 2 i h_{min}` describes the interval that is available to place an edge when there are already i edges in the model

        Parameters
        ----------
        min_edge : float64
            Minimum edge possible
        max_edge : float64
            Maximum edge possible
        max_cells : int
            Maximum number of cells allowable
        min_width : float64, optional
            Minimum width of any layer. If min_width = None, min_width is computed from min_edge, max_edge, and max_cells (recommended).
        prng : numpy.random.RandomState(), optional
            Random number generator, if none is given, will use numpy's global generator.
        n_cells_prior : geobipy.Distribution, optional
            Distribution describing the prior on the number of cells. Overrides the default.
        edge_prior : geobipy.Distribution, optional
            Distribution describing the prior on the cell edges. Overrides the default.

        See Also
        --------
        RectilinearMesh1D.perturb : For a description of the perturbation cycle.

        """

        if edges_prior is None:
            if kwargs.get('solve_edges', True):

                assert all(x in kwargs for x in ('max_cells', 'min_edge', 'max_edge')), ValueError("No edges_prior distribution given, must specify keywords 'max_cells', 'min_edge', 'max_edge', 'min_width'")

                self.max_cells = kwargs['max_cells']
                self.min_edge = kwargs.pop('min_edge')  # Assign the log of the min depth
                self.max_edge = kwargs.pop('max_edge')  # Assign the log of the max depth
                self.min_width = kwargs.pop('min_width', None)

                min_edge = self.min_edge; max_edge = self.max_edge
                min_width = self.min_width; max_cells = self.max_cells

                assert (self.min_width * self.max_cells) < self.max_edge, ValueError(f"{min_width * max_cells=} is bigger than the max_edge {max_edge}.  Either reduce max_cells to {int32((max_edge-min_edge)/min_width)} or min_width")

                # Set priors on the depth interfaces, given a number of layers
                dz = self.remainingSpace(arange(self.max_cells))

                self.edges.prior = Distribution('Order', denominator=dz)

        if n_cells_prior is None:
            if kwargs.get('solve_n_cells', True):
                self._nCells = StatArray(1, 'Number of cells', dtype=int32) + self.centres.size
                self.nCells.prior = Distribution('Uniform', 1, kwargs.pop('max_cells'), prng=kwargs.get('prng'))

        self.set_n_cells_prior(n_cells_prior)
        self.set_edges_prior(edges_prior)

    def set_axis(self, axis, value):
            self = value

    def set_n_cells_prior(self, prior):
        if prior is not None:
            self.nCells.prior = prior

    def set_edges_prior(self, prior):
        if prior is not None:
            self.edges.prior = prior

    def set_proposals(self, probabilities, **kwargs):
        """Setup the proposal distibution.

        Parameters
        ----------
        probabilities : array_like
            Probability of birth, death, perturb, and no change for the model
            e.g. probabilities = [0.5, 0.25, 0.15, 0.1]
        parameterProposal : geobipy.Distribution
            The proposal  distribution for the parameter.
        prng : numpy.random.RandomState(), optional
            Random number generator, if none is given, will use numpy's global generator.

        See Also
        --------
        geobipy.Model1D.perturb : For a description of the perturbation cycle.

        """
        probabilities = asarray(probabilities).copy()
        self._event_proposal = Distribution('Categorical',
                                            probabilities,
                                            ['insert', 'delete', 'perturb', 'none'],
                                            prng=kwargs.get('prng', None))

        self.edges.proposal = Distribution('Uniform',
                                           min=self.min_edge,
                                           max=self.max_edge,
                                        #    log=True,
                                           prng=kwargs.get('prng', None))

    def unperturb(self):
        """After a mesh has had its structure perturbed, remap back its previous state. Used for the reversible jump McMC step.

        """
        if self.action[0] == 'none':
            return deepcopy(self)

        if self.action[0] == 'perturb':
            other = deepcopy(self)
            other.edges[self.action[1]] -= self.action[2]
            return other

        if self.action[0] == 'insert':
            return self.delete_layer(self.action[1])

        if self.action[0] == 'delete':
            return self.insert_layer(self.action[2])

    def update_posteriors(self, values=None, ratio=None):

        # Update the number of layeres posterior
        self.nCells.update_posterior()

        # Update the layer interface histogram
        if (self.nCells.item() > 1):
            d = self.edges[~isinf(self.edges)][1:]
            if not values is None:
                r = exp(diff(nplog(values)))
                m1 = r <= 1.0 - ratio
                m2 = r >= 1.0 + ratio
                keep = logical_not(masked_invalid(ratio).mask) & mask_or(m1,m2)
                d = d[keep]

            if (d.size > 0):
                self.edges.posterior.update(d, trim=True)

    def hdfName(self):
        """ Reprodicibility procedure """
        return('RectilinearMesh1D()')

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None, upcast=True):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # If another axis is given, upcast to a 2D mesh
        if (add_axis is not None and upcast):
            return self._create_hdf_2d(parent, name, withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        # Otherwise mesh is 1D.
        grp = self.create_hdf_group(parent, name)

        if not self.log is None:
            grp.create_dataset('log', data=self.log)

        if self._nCells is not None:
            self.nCells.createHdf(grp, 'nCells', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        if self._relative_to is not None:
            self.relative_to.createHdf(grp, 'relative_to', add_axis=add_axis, fillvalue=fillvalue, withPosterior=False)

        # self.centres.toHdf(grp, 'centres', withPosterior=withPosterior)
        self.edges.toHdf(grp, 'edges', withPosterior=withPosterior)

        data = self.dimension if add_axis is None else self.dimension + 1
        grp.create_dataset('dimension', data=data, shape=(1, ))

        return grp

    # def _create_hdf_2d_stitched(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
    #     if isinstance(add_axis, (int, int_)):
    #         x = arange(add_axis, dtype=float64)
    #     else:
    #         x = add_axis
    #     relative_to = None if self._relative_to is None else StatArray(zeros(x.size))

    #     mesh = RectilinearMesh2D_stitched.RectilinearMesh2D_stitched(x=x, y=self, max_cells=self.max_cells, relative_toCentres=relative_to)
    #     out = mesh.createHdf(parent, name, withPosterior=withPosterior, fillvalue=fillvalue)
    #     mesh.x.writeHdf(out, 'x')

    def _create_hdf_2d(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):

        from .RectilinearMesh2D_stitched import RectilinearMesh2D_stitched
        from .RectilinearMesh2D import RectilinearMesh2D

        if isinstance(add_axis, (int, int_)):
            x = arange(add_axis, dtype=float64)
        else:
            x = add_axis
        if not isinstance(x, RectilinearMesh1D):
            x = RectilinearMesh1D(centres=x, dimension=0)

        if self._nCells is None:
            out = self.create_hdf_group(parent, name, hdf_name='RectilinearMesh2D')
            x.toHdf(out, 'x', withPosterior=withPosterior)
            self.createHdf(out, 'y', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=False)

        else:
            mesh = RectilinearMesh2D_stitched(x=x, relative_to=self.relative_to, max_cells=self.max_cells)
            if self.nCells.hasPosterior:
                mesh.nCells.posterior = Histogram.Histogram(mesh=RectilinearMesh2D(x=mesh.x, y=self.nCells.posterior.mesh))
            if self.edges.hasPosterior:
                mesh.y_edges.posterior = Histogram.Histogram(mesh=RectilinearMesh2D(x=mesh.x, y=self.edges.posterior.mesh))

            out = mesh.createHdf(parent, name, withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=False)
            x.writeHdf(out, 'x', withPosterior=withPosterior)

        return out

    def writeHdf(self, parent, name, withPosterior=True, index=None, upcast=True):

        if (index is not None) and upcast:
            return self._write_hdf_2d(parent, name, withPosterior=withPosterior, index=index)
        else:
            self._write_hdf_1d(parent, name, withPosterior=withPosterior, index=index)

    def _write_hdf_1d(self, parent, name, withPosterior=True, index=None):

        grp = parent.get(name)

        if index is None:
            assert ('1D' in grp.attrs['repr']), Exception('Writing 1D mesh to 2D HDF entry requires an index')

        if self._nCells is not None:
            self.nCells.writeHdf(grp, 'nCells',  withPosterior=withPosterior, index=index)

        if self._relative_to is not None:
            self.relative_to.writeHdf(grp, 'relative_to', index=index, withPosterior=False)

        # Edges can have a posterior
        self.edges.writeHdf(grp, 'edges',  withPosterior=withPosterior)

        # grp['dimension'][0] = self.dimension

    def _write_hdf_2d(self, parent, name, withPosterior=True, index=None):
        grp = parent.get(name)

        if self._relative_to is not None:
            self.relative_to.writeHdf(grp, 'y/relative_to', index=index, withPosterior=False)

        ind = None
        if self._nCells is not None:
            self.nCells.writeHdf(grp, 'nCells',  withPosterior=withPosterior, index=index)
            ind = index

        # Edges can have a posterior
        self.edges.writeHdf(grp, 'y/edges',  withPosterior=withPosterior, index=ind)

    @classmethod
    def fromHdf(cls, grp, index=None, skip_posterior=False):
        """ Reads in the object froma HDF file """

        if '1D' in grp.attrs['repr']:
            out = RectilinearMesh1D._1d_from_1d(grp, index, skip_posterior=skip_posterior)

        elif 'stitched' in grp.attrs['repr']:
            out = RectilinearMesh1D._1d_from_stitched(grp, index, skip_posterior=skip_posterior)
        else:
            if '2D' in grp.attrs['repr']:
                out = RectilinearMesh1D._1d_from_2d(grp, index, skip_posterior=skip_posterior)
            # if '3D' in grp.attrs['repr']:
            #     out = RectilinearMesh1D._1d_from_2d(grp, index)

        if 'min_width' in grp:
            out._min_width = array(grp.get('min_width'))
        if 'min_edge' in grp:
            out._min_edge = array(grp.get('min_edge'))
        if 'max_edge' in grp:
            out._max_edge = array(grp.get('max_edge'))
        if 'max_cells' in grp:
            out._max_cells = array(grp.get('max_cells'))

        return out

    @classmethod
    def _1d_from_1d(cls, grp, index, skip_posterior=False):
        # Get the number of cells of the mesh.
        # If present, its a perturbable mesh
        nCells = None
        if 'nCells' in grp:
            i = index if grp['nCells/data'].size > 1 else None
            tmp = StatArray.fromHdf(grp['nCells'], index=i, skip_posterior=skip_posterior)
            nCells = tmp.astype(int32)
            nCells.copyStats(tmp)
            assert nCells.size == 1, ValueError("Mesh was created with expanded memory\nIndex must be specified")

        # If no index, we are reading in multiple models side by side.
        log = None
        if 'log' in grp:
            log = asarray(grp['log']).item()

        dimension = 0
        if 'dimension' in grp:
            dimension = asarray(grp['dimension']).item()
            if index is not None:
                dimension -= 1

        # If relative_to is present, the edges/centres should be 1 dimensional
        relative_to = None
        key = 'relative_to' if 'relative_to' in grp else 'relativeTo'

        if key in grp:
            i = index if grp[f'{key}/data'].size > 1 else None
            relative_to = DataArray.fromHdf(grp[key], index=i, skip_posterior=skip_posterior)

            s = index if nCells is None else s_[:nCells.item() + 1]
            pi = None #if nCells is None else s_[:]
            # edges = StatArray.fromHdf(grp['edges'], index=index, skip_posterior=skip_posterior, posterior_index=pi)

        else:
            s = None if nCells is None else s_[:nCells.item() + 1]
            pi = None #if nCells is None else s_[:]

        edges = StatArray.fromHdf(grp['edges'], skip_posterior=skip_posterior, posterior_index=pi)

        out = cls(edges=edges, dimension=dimension)

        # centres = None
        # if (edges is None) or (npall(edges == 0.0)):
        #     s = None if nCells is None else s_[:nCells.item()]
        #     centres = StatArray.fromHdf(grp['centres'], skip_posterior=skip_posterior)
            # if not npall(centres == 0.0):
            #     out.centres = centres

        out.relative_to = relative_to
        if nCells is not None:
            out._nCells = nCells

        out.log = log

        return out

    @classmethod
    def _1d_from_2d(cls, grp, index, skip_posterior=False):

        if index is None:
            assert False, ValueError("RectilinearMesh1D cannot be read from a RectilinearMesh2D without an index")

        grp = grp['y']

        # If no index, we are reading in multiple models side by side.
        log = None
        if 'log' in grp:
            log = asarray(grp['log']).item()

        # If relative_to is present, the edges/centres should be 1 dimensional
        relative_to = None
        key = 'relative_to' if 'relative_to' in grp else 'relativeTo'
        if key in grp:
            relative_to = StatArray.fromHdf(grp[key], index=index, skip_posterior=skip_posterior)
        #     edges = StatArray.fromHdf(grp['edges'], skip_posterior=skip_posterior)
        # else:

        edges = StatArray.fromHdf(grp['edges'], skip_posterior=skip_posterior)

        out = cls(edges=edges)

        # centres = None
        if (edges is None) or (npall(edges == 0.0)):
            # if 'relative_to' in grp:
            centres = DataArray.fromHdf(grp['centres'], skip_posterior=skip_posterior)
            # else:
            #     centres = StatArray.fromHdf(grp['centres'], index=index, skip_posterior=skip_posterior)
            if not npall(centres == 0.0):
                out.centres = centres

        out._relative_to = relative_to
        out.log = log
        return out

    @classmethod
    def _1d_from_stitched(cls, grp, index, skip_posterior=False):

        if index is None:
            assert False, ValueError("RectilinearMesh1D cannot be read from a RectilinearMesh2D_stitched without an index")
        #     from .RectilinearMesh2D_stitched import RectilinearMesh2D_stitched
        #     return RectilinearMesh2D_stitched.fromHdf(grp)

        # Get the number of cells of the mesh.
        # If present, its a perturbable mesh
        tmp = StatArray.fromHdf(grp['nCells'], index=index, skip_posterior=skip_posterior)
        nCells = tmp.astype(int32)
        nCells.copyStats(tmp)

        # If no index, we are reading in multiple models side by side.
        log = None
        if 'y/log' in grp:
            log = asarray(grp['y/log']).item()

        # If relative_to is present, the edges/centres should be 1 dimensional
        relative_to = None
        key = 'relative_to' if 'relative_to' in grp else 'relativeTo'
        if key in grp:
            relative_to = DataArray.fromHdf(grp[key], index=index, skip_posterior=skip_posterior)
            if relative_to == 0.0:
                relative_to = None


        s = (index, s_[:nCells.item() + 1])
        posterior_index = index

        edges = StatArray.fromHdf(grp['y/edges'], index=s, skip_posterior=skip_posterior, posterior_index=index)

        out = cls(edges=edges)

        out.relative_to = relative_to
        out._nCells = nCells
        out.log = log

        return out

    def fromHdf_cell_values(self, grp, key, index=None, skip_posterior=False):

        s = s_[:self.nCells.item()]
        if index is not None:
            s = (index, s)

        return StatArray.fromHdf(grp[key], index=s, skip_posterior=skip_posterior)

    @property
    def summary(self):
        """Summary of self """
        msg =  "{}\n".format(type(self).__name__)
        if self._nCells is not None:
            msg += "Number of Cells:\n{}\n".format("|   "+(self._nCells.summary.replace("\n", "\n|   "))[:-4])
        else:
            msg += "Number of Cells:\n{}\n".format("|   "+str(self.nCells))
        msg += "Cell Centres:\n{}\n".format("|   "+(self._centres.summary.replace("\n", "\n|   "))[:-4])
        msg += "Cell Edges:\n{}\n".format("|   "+(self._edges.summary.replace("\n", "\n|   "))[:-4])
        # msg = msg[:-1]
        msg += "log:\n{}".format("|   "+str(self.log)+'\n')
        msg += "relative_to:\n{}\n".format("|   "+(self.relative_to.summary.replace("\n", "\n|   "))[:-4])

        return msg

    def Bcast(self, world, root=0):

        if world.rank == root:
            edges = self.edges
        else:
            edges = StatArray(0)

        edges = self.edges.Bcast(world, root=root)

        return RectilinearMesh1D(edges=edges)