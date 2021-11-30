""" @RectilinearMesh1D_Class
Module describing a 1D Rectilinear Mesh class
"""
from .Mesh import Mesh
from ...classes.core import StatArray
from copy import deepcopy
import numpy as np
from ...base import utilities as cF
from ...base import plotting as cp
from . import RectilinearMesh2D
from ..statistics.Distribution import Distribution
from ..statistics import Histogram1D
from scipy.sparse import diags
from scipy import interpolate
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class RectilinearMesh1D(Mesh):
    """Class defining a 1D rectilinear mesh with cell centres and edges.

    Contains a simple 1D mesh with cell edges, widths, and centre locations.

    RectilinearMesh1D(centres, edges, edgesMin, edgesMax)

    Parameters
    ----------
    centres : geobipy.StatArray, optional
        The locations of the centre of each cell. Only centres, edges, or widths can be given.
    edges : geobipy.StatArray, optional
        The locations of the edges of each cell, including the outermost edges. Only centres, edges, or widths can be given.
    widths : geobipy.StatArray, optional
        The widths of the cells.
    log : 'e' or float, optional
        Entries are given in linear space, but internally cells are logged.
        Plotting is in log space.
    relativeTo : float, optional
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
        centres must be a geobipy.StatArray.
    TypeError
        edges must be a geobipy.StatArray.

    """
    def __init__(self, centres=None, edges=None, widths=None, log=None, relativeTo=None):
        """ Initialize a 1D Rectilinear Mesh"""
        self._centres = None
        self._edges = None
        self._widths = None

        self.log = log
        self.relativeTo = relativeTo

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
        out._nCells = deepcopy(self._nCells)
        out._centres = deepcopy(self._centres)
        out._edges = deepcopy(self._edges)
        out._widths = deepcopy(self._widths)
        out.log = self.log
        out._relativeTo = self._relativeTo

        out._min_width = self.min_width
        out._min_edge = self.min_edge
        out._max_edge = self.max_edge
        out._max_cells = self.max_cells

        out._event_proposal = self.event_proposal
        out._action = deepcopy(self.action)

        return out

    def __getitem__(self, slic):
        """Slice into the class. """

        assert np.shape(slic) == (), ValueError(
            "slic must have one dimension.")

        s2stop = None
        if isinstance(slic, slice):
            if not slic.stop is None:
                s2stop = slic.stop + 1 if slic.stop > 0 else slic.stop
            s2 = slice(slic.start, s2stop, slic.step)
        else:
            s2 = slice(slic, slic + 2, 1)

        tmp = self._edges[s2]
        assert tmp.size > 1, ValueError("slic must contain at least one cell.")
        return type(self)(edges=tmp)

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
    def bounds(self):
        return np.r_[self.edges[0], self.edges[-1]]

    @property
    def centres(self):
        return self._centres

    @property
    def centres_absolute(self):
        return self._centres + self.relativeTo.value

    @centres.setter
    def centres(self, values):
        if not isinstance(values, StatArray.StatArray):
            values = StatArray.StatArray(values, cF.getName(self._centres), cF.getUnits(self._centres))

        values, _ = cF._log(values, log=self.log)
        values -= self.relativeTo

        values.name = cF._logLabel(self.log) + values.name

        # assert np.ndim(values) == 1, ValueError("centres must be 1D")
        # StatArray of the x axis values

        self._centres = deepcopy(values)
        self._edges = self._centres.edges()
        self._widths = self._edges.diff()

        self._nCells[0] = self.centres.size

    @property
    def centreTocentre(self):
        return np.diff(self.centres)

    @property
    def displayLimits(self):
        dx = 0.02 * self.range
        return (self.plotting_edges[0] - dx, self.plotting_edges[-1] + dx)

    @property
    def edges(self):
        return self._edges

    @property
    def edges_absolute(self):
        return self.edges + self.relativeTo.value
        # return cF._power(edges, self.log)

    @edges.setter
    def edges(self, values):
        if not isinstance(values, StatArray.StatArray):
            values = StatArray.StatArray(values, cF.getName(self._edges), cF.getUnits(self._edges))

        values, _ = cF._log(values, log=self.log)
        values -= self.relativeTo

        values.name = cF._logLabel(self.log) + values.name
        # assert np.ndim(values) == 1, ValueError("edges must be 1D")
        self._edges = deepcopy(values)
        self._centres = values.internalEdges()
        self._widths = values.diff()

        self._nCells[0] = self.centres.size

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
    def max_cells(self):
        return self._max_cells

    @property
    def max_edge(self):
        return self._max_edge

    @property
    def min_edge(self):
        return self._min_edge

    @property
    def min_width(self):
        return self._min_width

    @property
    def nCells(self):
        return self._nCells

    @nCells.setter
    def nCells(self, value):
        if value is None:
            self._nCells = StatArray.StatArray(1, '# of Cells', dtype=np.int32) + 1
        else:
            assert isinstance(value, (int, np.integer, StatArray.StatArray)), TypeError("nCells must be an integer, or StatArray")
            assert np.size(value) == 1, ValueError("nCells must be scalar or length 1")
            assert (value >= 1), ValueError('nCells must >= 1')
            if isinstance(value, int):
                self._nCells = StatArray.StatArray(1, '# of Cells', dtype=np.int32) + value
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
        return self.edges[0] == -np.inf

    @property
    def open_right(self):
        return self.edges[-1] == np.inf

    @property
    def range(self):
        """Get the difference between end edges."""
        return np.abs(self._edges[-1] - self._edges[0])

    @property
    def relativeTo(self):
        if self._relativeTo is None:
            return StatArray.StatArray(0.0)
        return self._relativeTo

    @relativeTo.setter
    def relativeTo(self, value):
        if value is None:
            self._relativeTo = None
            return

        if np.all(value > 0.0):
            value, _ = cF._log(value, self.log)
        self._relativeTo = StatArray.StatArray(value)

    @property
    def shape(self):
        return (self.nCells.value, )

    @property
    def units(self):
        return self._centres.units

    @property
    def widths(self):
        return self._widths

    @widths.setter
    def widths(self, values):
        if not isinstance(values, StatArray.StatArray):
            values = StatArray.StatArray(values)

        assert np.all(values > 0.0), ValueError(
            "widths must be entirely positive")

        self._widths = deepcopy(values)
        self.edges = np.hstack([0.0, np.cumsum(values)])

    def cellIndex(self, values, clip=False, trim=False):
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
        values, dum = cF._log(np.atleast_1d(values).flatten(), self.log)
        values = values - self.relativeTo

        # Get the bin indices for all values
        iBin = np.atleast_1d(self.edges.searchsorted(values, side='right') - 1)

        # Remove indices that are out of bounds
        if trim:
            iBin = iBin[(values >= self.edges[0]) &
                        (values < self.edges[-1])]
        else:
            # Force out of bounds to be in bounds if we are clipping
            if clip:
                iBin = np.maximum(iBin, 0)
                iBin = np.minimum(iBin, self.nCells.value - 1)
            # Make sure values outside the lower edge are -1
            else:
                iBin[values < self.edges[0]] = -1
                iBin[values >= self.edges[-1]] = self.nCells

        return np.squeeze(iBin)

    def delete_edge(self, i, **kwargs):
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
            return self

        assert 1 <= i <= (
            self.nEdges - 1), ValueError("Required  1 <= i <= {}".format(self.nEdges-1))

        # Deepcopy the 1D Model to ensure priors and proposals are passed
        out = deepcopy(self)
        # Remove the interface depth
        out.edges = out.edges.delete(i)

        out._action = ['delete', np.int32(i), np.squeeze(self.edges[i])]
        return out

    def gradientMatrix(self):
        tmp = 1.0/np.sqrt(self.centreTocentre)
        return diags([tmp, -tmp], [0, 1], shape=(self.nCells.value-1, self.nCells.value))

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
        values, _ = cF._log(values, self.log)
        return (values >= self._edges[0]) & (values < self._edges[-1])

    def insert_edge(self, value, **kwargs):
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
        out._action = ['insert', np.int32(i), value]
        return out

    def is_left(self, value):
        value, _ = cF._log(value, self.log)
        return value < self.edges[0]

    def is_right(self, value):
        value, _ = cF._log(value, self.log)
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
        iBig = np.where(w >= distance2)
        n_large = np.size(iBig)
        new_edges = np.full((self.nEdges + 2*n_large), fill_value=np.nan)
        indices = np.zeros(self.nCells.value, dtype=np.int32)

        x = np.asarray([-distance, +distance])
        k = 0

        for i in range(self.nCells.value):
            new_edges[k] = self.edges[i]
            k += 1
            modified = False
            if ((self.centres[i] - self.edges[i]) > distance):
                new_edges[k] = self.centres[i] - distance
                indices[i] = k-1
                modified = True
                k += 1
            else:
                indices[i] = k-1

            if ((self.edges[i+1] - self.centres[i]) > distance):
                new_edges[k] = self.centres[i] + distance
                indices[i] = k-1
                mdified = True
                k += 1
            else:
                indices[i] = k-1

        new_edges[k] = self.edges[-1]

        new_edges = new_edges[~np.isnan(new_edges)]
        out = RectilinearMesh1D(edges=new_edges)

        if not values is None:
            out_values = np.full(out.nCells.value, fill_value=np.nan)
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
        out.edges = self.edges.pad(size)
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

        # assert isinstance(values, StatArray.StatArray), TypeError("arr must be a StatArray")
        assert values.size == self.nCells, ValueError(
            "arr must have size nCell {}".format(self.nCells))

        values = StatArray.StatArray(values)

        kwargs['y'] = kwargs.pop('y', self.plotting_edges)

        # if not self.log is None:
        #     kwargs['y'] = cF._power(kwargs['y'], self.log)
        #     kwargs['yscale'] = 'log'

        ax = values.pcolor(**kwargs)

        return ax

    def perturb(self, verbose=False):
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
        if verbose: print('perturb')
        assert (not self.event_proposal is None), ValueError(
            'Please set the proposals with RectilinearMesh1D.setProposals()')
        prng = self.nCells.prior.prng

        nTries = 10
        # This outer loop will allow the perturbation to change types. e.g. if the loop is stuck in a birthing
        # cycle, the number of tries will exceed 10, and we can try a different perturbation type.
        tryAgain = True  # Temporary to enter the loop
        if verbose: print(self.edges)
        while (tryAgain):
            tryAgain = False
            goodAction = False

            # Choose an action to perform, and make sure its legitimate
            # i.e. don't delete a single layer mesh, or add a layer to a mesh that is at the priors max on number of cells.
            while not goodAction:
                goodAction = True
                # Get a random probability from 0-1
                event = self.event_proposal.rng()

                if ((self.nCells.value == 1) and (event == 1 or event == 2)):
                    goodAction = False
                elif ((self.nCells.value == self.nCells.prior.max) and (event == 0)):
                    goodAction = False

            if verbose: print('event', event)

            # Return if no change
            if (event == 3):
                out = deepcopy(self)
                out._action = ['none', 0, 0.0]
                return out

            # Otherwise enter life-death-perturb cycle
            if (event == 0):  # Create a new layer
                suitable_width = False
                tries = 0
                while (not suitable_width):  # Continue while the new layer is smaller than the minimum
                    # Get the new edge
                    new_edge = np.exp(prng.uniform(np.log(self.min_edge), np.log(self.max_edge), 1))
                    # Insert the new depth
                    i = self.edges.searchsorted(new_edge)
                    z = self.edges.insert(i, new_edge)
                    # Get the thicknesses
                    h = np.min(np.diff(z))
                    tries += 1
                    suitable_width = (h > self.min_width)
                    if (tries == nTries):
                        suitable_width = True  # just to exit.
                        tryAgain = True
                if (not tryAgain):
                    out = self.insert_edge(new_edge, update_priors=True)
                    if verbose: print('insert', out.edges)
                    return out

            if (event == 1):  # Remove an edge
                # Get the layer to remove
                iDeleted = np.int64(prng.uniform(0, self.nCells.value-1, 1)[0]) + 1
                # Remove the layer and return
                out = self.delete_edge(iDeleted, update_priors=True)
                if verbose: print('death', out.edges)
                return out

            if (event == 2):  # Perturb an edge
                suitable_width = False
                tries = 0
                while (not suitable_width):  # Continue while the perturbed layer is suitable

                    z = self.edges.copy()
                    # Get the internal edge to perturb
                    i = np.int64(prng.uniform(1, self.nEdges-1, 1)[0])
                    # Get the perturbation amount
                    dz = np.sign(prng.randn()) * self.min_width * prng.uniform()
                    # Perturb the layer
                    z[i] += dz
                    # Get the minimum thickness
                    h = np.min(np.diff(z))
                    tries += 1

                    # Exit if the thickness is big enough, and we stayed within
                    # the depth bounds
                    suitable_width = ((h > self.min_width) and (z[1] > self.min_edge) and (z[-2] < self.max_edge))

                    if (tries == nTries):
                        suitable_width = True
                        tryAgain = True

                if (not tryAgain):
                    out = deepcopy(self)
                    # z0[i] += dz  # Perturb the depth in the model
                    out.edges = z
                    out._action = ['perturb', np.int32(i), dz]
                    if verbose: print('perturb', out.edges)
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
        assert isinstance(other, (RectilinearMesh1D, RectilinearMesh2D.RectilinearMesh2D)), TypeError('mesh must be a RectilinearMesh1D or RectilinearMesh2D')

        if isinstance(other, RectilinearMesh2D.RectilinearMesh2D):
            other = other.axis(axis)

        edges = self.plotting_edges
        bounds = [np.maximum(other.edges[0], edges[0]), np.minimum(other.edges[-1], edges[-1])]

        y = other.centres

        if (self.nCells.value == 1):
            out = np.interp(y, bounds, np.kron(values, [1, 1]))
        else:
            xp = np.kron(np.asarray(edges), [1, 1.000001])[1:-1]
            fp = np.kron(values, [1, 1])
            out = np.interp(y, xp, fp)

        if bound:
            i = np.where((y < bounds[0]) & (y > bounds[-1]))
            out[i] = np.nan

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

        ax, stp = cp.step(x=par, y=self.plotting_edges, **kwargs)
        # if self.hasHalfspace:
        #     h = 0.99*z[-1]
        #     if (self.nCells == 1):
        #         h = 0.99*self.max_edge
        #     plt.text(0, h, s=r'$\downarrow \infty$', fontsize=12)

    def plotGrid(self, **kwargs):
        """ Plot the grid lines of the mesh.

        See Also
        --------
        geobipy.StatArray.pcolor : For additional plotting arguments

        """

        kwargs['grid'] = True
        kwargs['noColorbar'] = True

        values = StatArray.StatArray(np.full(self.nCells.value, np.nan))
        RectilinearMesh1D.pcolor(self, values=values, y=self.plotting_edges, **kwargs)

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

    @property
    def n_posteriors(self):
        return np.sum([x.hasPosterior for x in [self.nCells, self.edges]])

    def init_posterior_plots(self, gs):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """

        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        splt = gs.subgridspec(2, 1, height_ratios=[1, 4])
        ax = [plt.subplot(splt[0, 0]), plt.subplot(splt[1, 0])]

        for a in ax:
            cp.pretty(a)

        return ax

    def plot_posteriors(self, axes=None, ncells_kwargs={}, edges_kwargs={}, **kwargs):
        assert len(axes) == 2, ValueError("Must have length 2 list of axes for the posteriors. self.init_posterior_plots can generate them")

        edges_kwargs['rotate'] = edges_kwargs.get('rotate', True)

        best = kwargs.get('best', None)
        if not best is None:
            ncells_kwargs['line'] = best.nCells
            edges_kwargs['line'] = best.edges[1:]
        self.nCells.plotPosteriors(ax = axes[0], **ncells_kwargs)
        self.edges.plotPosteriors(ax = axes[1] ,**edges_kwargs)

        # ax = self.currentModel.nCells.posterior.plot(**kwargs)
        # plt.axvline(self.bestModel.nCells, color=cP.wellSeparated[3], linewidth=1)
        # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    def priorProbability(self, log=True, verbose=False):
        """Evaluate the prior probability for the mesh.

        The following equation describes the components of the prior that correspond to the Model1D,

        .. math::
            p(k | I)p(\\boldsymbol{e}| k, I),

        where :math:`k, I, \\boldsymbol{e}` are the number of cells, prior information, edge location respectively.

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
        P_nCells = self.nCells.probability(log=log)

        # Probability of depth given nCells
        P_edges = self.edges.probability(x=self.nCells.value-1, log=log)

        return (P_nCells + P_edges) if log else (P_nCells * P_edges)

    def remainingSpace(self, n_cells):
        return (self.max_edge - self.min_edge) - n_cells * self.min_width

    def resample(self, dx, values, kind='cubic'):
        x = np.arange(self.edges[0], self.edges[-1]+dx, dx)

        mesh = RectilinearMesh1D(edges=x)
        f = interpolate.interp1d(self.centres, values, kind=kind)
        return mesh, f(mesh.centres)

    def interpolate_centres_to_nodes(self, values, kind='cubic', **kwargs):
        kwargs['fill_value'] = kwargs.pop('fill_value', 'extrapolate')
        f = interpolate.interp1d(self.centres, values, kind=kind, **kwargs)
        return f(self.edges)

    def set_posteriors(self, nCells_posterior=None, edges_posterior=None):

        # Initialize the posterior histogram for the number of layers
        if nCells_posterior is None:
            self.nCells.posterior = Histogram1D.Histogram1D(centres=StatArray.StatArray(np.arange(0.0, self.max_cells + 1.0), name="# of Layers"))

        if edges_posterior is None:

            assert not self.max_cells is None, ValueError(
                "No priors are set, user self.set_priors().")

            # Discretize the parameter values
            zGrd = StatArray.StatArray(np.arange(
                0.9 * self.min_edge, 1.1 * self.max_edge, 0.5 * self.min_width), self.edges.name, self.edges.units)

            # Initialize the interface Depth Histogram
            self.edges.posterior = Histogram1D.Histogram1D(edges=zGrd)

    def set_priors(self, min_edge, max_edge, max_cells, min_width=None, prng=None, n_cells_prior=None, edge_prior=None):
        """Setup the priors of the mesh.

        By default the following priors are set unless explictly specified.

        **Prior on the number of cells**

        Uninformative prior using a uniform distribution.

        .. math::
            :label: layers

            p(k | I) =
            \\begin{cases}
            \\frac{1}{k_{max} - 1} & \\quad 1 \leq k \leq k_{max} \\newline
            0 & \\quad otherwise
            \\end{cases}.

        **Prior on the cell edges**

        We use order statistics for the prior on cell edges.

        .. math::
            :label: depth

            p(\\boldsymbol{e} | k, I) = \\frac{(k -1)!}{\prod_{i=0}^{k-1} \Delta e_{i}},

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
        assert min_edge > 0.0, ValueError("min_edge must be > 0.0")
        assert max_edge > 0.0, ValueError("max_edge must be > 0.0")
        assert max_cells > 0, ValueError("max_cells must be > 0")

        if (min_width is None):
            # Assign a minimum possible thickness
            self._min_width = (max_edge - min_edge) / (2 * max_cells)
        else:
            self._min_width = min_width

        self._min_edge = min_edge  # Assign the log of the min depth
        self._max_edge = max_edge  # Assign the log of the max depth
        self._max_cells = np.int32(max_cells)

        # Assign a uniform distribution to the number of layers
        self.nCells.prior = Distribution('Uniform', 1, self.max_cells, prng=prng)

        # Set priors on the depth interfaces, given a number of layers
        i = np.arange(self.max_cells)
        dz = self.remainingSpace(i)

        self.edges.prior = Distribution('Order', denominator=dz)  # priZ

    def set_proposals(self, probabilities, prng=None):
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
        probabilities = np.asarray(probabilities)
        self._event_proposal = Distribution('Categorical', probabilities, [
                                            'insert', 'delete', 'perturb', 'none'], prng=prng)

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
        self.nCells.updatePosterior()

        # Update the layer interface histogram
        if (self.nCells > 1):
            d = self.edges[~np.isinf(self.edges)][1:]
            if not values is None:
                r = np.exp(np.diff(np.log(values)))
                m1 = r <= 1.0 - ratio
                m2 = r >= 1.0 + ratio
                keep = np.logical_not(np.ma.masked_invalid(ratio).mask) & np.ma.mask_or(m1,m2)
                d = d[keep]

            if (d.size > 0):
                self.edges.posterior.update(d, trim=True)

    def hdfName(self):
        """ Reprodicibility procedure """
        return('RectilinearMesh1D()')

    def createHdf(self, parent, name, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = self.create_hdf_group(parent, name)

        self.nCells.createHdf(grp, 'nCells', withPosterior=withPosterior, nRepeats=nRepeats)

        if not self.log is None:
            grp.create_dataset('log', data=self.log)

        if self._relativeTo is not None:
            self.relativeTo.createHdf(grp, 'relativeTo', nRepeats=nRepeats, fillvalue=fillvalue)

        self.centres.createHdf(grp, 'centres', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        self.edges.createHdf(grp, 'edges', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)

        # Instantiate extra parameters for Markov chain perturbations.
        if not self.min_edge is None: grp.create_dataset('min_edge', data=self.min_edge)
        if not self.max_edge is None: grp.create_dataset('max_edge', data=self.max_edge)
        if not self.max_cells is None: grp.create_dataset('max_cells', data=self.max_cells)
        if not self.min_width is None: grp.create_dataset('min_width', data=self.min_width)
        # if not self.event_proposal is None: grp.create_dataset('event_proposal', data=self.event_proposal)

        return grp

    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        grp = parent.get(name)
        self.nCells.writeHdf(grp, 'nCells',  withPosterior=withPosterior, index=index)
        if self._relativeTo is not None:
            self.relativeTo.writeHdf(grp, 'relativeTo', index=index)

        self.centres.writeHdf(grp, 'centres',  withPosterior=withPosterior, index=index)
        self.edges.writeHdf(grp, 'edges',  withPosterior=withPosterior, index=index)
        

    @classmethod
    def fromHdf(cls, grp, index=None):
        """ Reads in the object froma HDF file """

        # Get the number of cells of the mesh
        if 'nCells' in grp:
            tmp = StatArray.StatArray.fromHdf(grp['nCells'], index=index)
            nCells = tmp.astype(np.int32)
            nCells.copyStats(tmp)
            assert nCells.size == 1, ValueError("Mesh was created with expanded memory\nIndex must be specified")

        log = None
        if 'log' in grp:
            log = np.asscalar(np.asarray(grp['log']))

        # If relativeTo is present, the edges/centres should be 1 dimensional
        relativeTo = None
        if 'relativeTo' in grp:
            relativeTo = StatArray.StatArray.fromHdf(grp['relativeTo'], index=index)
        else:
            if 'top' in grp:
                relativeTo = StatArray.StatArray.fromHdf(grp['top'], index=index)

        if relativeTo == 0.0:
            relativeTo = None
            
        edges = None
        if (('edges' in grp) or ('bins' in grp)):
            if 'edges' in grp:
                key = 'edges'
            elif 'bins' in grp:
                key = 'bins'

            if (np.ndim(grp[key+'/data']) == 2):
                                
                edges = StatArray.StatArray.fromHdf(grp[key], index=(index, np.s_[:nCells.value+1]))
            else:
                edges = StatArray.StatArray.fromHdf(grp[key], index=np.s_[:nCells.value+1])
                
        centres = None
        # if edges is None and (('centres' in grp) or ('x' in grp)):
        #     key = 'centres' if not 'x' in grp else 'x'
        #     if np.ndim(grp[key+'/data']) == 2:
        #         centres = StatArray.StatArray.fromHdf(grp[key], index=(index, np.s_[:nCells.value]))
        #     else:
        #         centres = StatArray.StatArray.fromHdf(grp[key], index=np.s_[:nCells.value+1])

        out = cls(centres=centres, edges=edges, widths=None)

        out.log = log
        out.relativeTo = relativeTo
        out._nCells = nCells

        if 'min_width' in grp: 
            out._min_width = np.array(grp.get('min_width'))
        if 'min_edge' in grp:
            out._min_edge = np.array(grp.get('min_edge'))
        if 'max_edge' in grp: 
            out._max_edge = np.array(grp.get('max_edge'))
        if 'max_cells' in grp: 
            out._max_cells = np.array(grp.get('max_cells'))

        # if 'event_proposal' in grp: self._event_proposal = np.array(grp.get('event_proposal'))

        # if 'depth' in grp: # Old Model1D class
        #     i = np.s_[index, :nCells.value]
        #     tmp = StatArray.StatArray.fromHdf(grp['depth'], index=i)
        #     edges = tmp.prepend(0.0)
        #     edges[-1] = np.inf
        #     edges.copyStats(tmp)

        #     out  = cls(edges=edges)
        #     out.nCells = nCells
        #     # self.edges = edges

        #     if 'minThk' in grp: out._min_width = np.array(grp.get('minThk'))
        #     if 'hmin'   in grp: out._min_width = np.array(grp.get('hmin'))
        #     if 'zmin'   in grp: out._min_edge = np.array(grp.get('zmin'))
        #     if 'zmax'   in grp: out._max_edge = np.array(grp.get('zmax'))
        #     if 'kmax'   in grp: out._max_cells = np.array(grp.get('kmax'))

        return out

    @property
    def summary(self):
        """ Print a summary of self """
        msg = ("Cell Centres\n"
               "    {}"
               "Cell Edges \n"
               "    {}").format(
                   self._centres.summary,
                   self._edges.summary
        )

        return msg

    def Bcast(self, world, root=0):

        if world.rank == root:
            edges = self.edges
        else:
            edges = StatArray.StatArray(0)

        edges = self.edges.Bcast(world, root=root)

        return RectilinearMesh1D(edges=edges)
