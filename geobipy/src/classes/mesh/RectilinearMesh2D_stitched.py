""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
from copy import copy, deepcopy
from matplotlib.figure import Figure
from ...classes.core import StatArray
from .RectilinearMesh2D import RectilinearMesh2D
import numpy as np
import matplotlib.cm as mplcm
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from ...base import plotting as cP
from ...base import utilities
# from ...base import geometry
from scipy.sparse import (kron, diags)

class RectilinearMesh2D_stitched(RectilinearMesh2D):
    """Class defining stitched 1D rectilinear meshes.
    """

    def __init__(self, max_cells, x=None, relativeTo=None, nCells=None, **kwargs):
        """ Initialize a 2D Rectilinear Mesh"""

        self._max_cells = np.int32(max_cells)
        self.x = kwargs if x is None else x
        self.nCells = nCells
        self.y_edges = None
        self.y_log = kwargs.get('y_log')
        self.relativeTo = relativeTo

    @property
    def max_cells(self):
        return self._max_cells

    @property
    def nCells(self):
        return self._nCells

    @nCells.setter
    def nCells(self, values):
        if values is None:
            values = self.x.nCells
        else:
            assert np.size(values) == self.x.nCells, ValueError("Size of nCells must be {}".format(self.x.nCells))

        self._nCells = StatArray.StatArray(values, 'Number of cells', dtype=np.int32)

    @property
    def shape(self):
        return (self.x.nCells.item(), self.max_cells)

    @property
    def summary(self):
        """ Display a summary of the 3D Point Cloud """

        msg = ("2D Stitched Rectilinear Mesh: \n"
              "nCells:{}\n"
              "x\n{}"
              "y\n{}").format(self.nCells.summary, self.x.summary, self.y_edges.summary)

        return msg

    @property
    def y_edges(self):
        return self._y_edges

    @y_edges.setter
    def y_edges(self, values):

        if values is None:
            values = (self.x.nCells, self.max_cells+1)
        else:

            assert np.ndim(values) == 2, ValueError("y_edges must have 2 dimensions")
            assert np.shape(values)[0] == self.x.nCells, ValueError("First dimension of y_edges must have size {}".format(self.x.nCells))

            values, dum = utilities._log(values, log=self.y_log)

            for i in range(self.x.nCells):
                values[i, self.nCells[i]+1:] = np.nan

        self._y_edges = StatArray.StatArray(values)

    @property
    def plotting_edges(self):
        out = copy(self.y_edges)
        if np.any(self.y_edges[:, -1] == np.inf):
            i = np.argwhere(self.y_edges[:, -1] == np.inf)
            out[i, -1] = 1.1 * np.max(out[i, -2])
        return out

    def __getitem__(self, slic):
        """Allow slicing of the histogram."""
        from .RectilinearMesh1D import RectilinearMesh1D
        # assert np.shape(slic) == (1,), ValueError("slic must be over 1 dimensions.")

        if isinstance(slic, (int, np.integer)):
            relativeTo = self.x._relativeTo[slic] if not self.x._relativeTo is None else None
            return RectilinearMesh1D(edges=self.y_edges[slic, :self.nCells[slic]+2], relativeTo=relativeTo)

        else:
            slic0 = slic
            if isinstance(slic.stop, (int, np.integer)):
                # If a slice, add one to the end for bins.
                slic0 = slice(slic.start, slic.stop+1, slic.step)

            relativeTo = self.relativeTo[slic] if not self.relativeTo is None else None
            if self.xyz:
                out = type(self)(x_edges=self._x.edges[slic0], y_edges=self._y.edges[slic0], z_edges=self._z_edges[slic0, :], relativeTo=relativeTo)
            else:
                out = type(self)(x_edges=self._x.edges[slic0], y_edges=self._z_edges[slic0, :], relativeTo=relativeTo)
            out.nCells = self.nCells[slic]
            return out

    def n_posteriors(self):
        return np.sum([self.nCells.hasPosterior, self.y.edges.hasPosterior])

    def pcolor(self, values, **kwargs):

        assert np.all(np.shape(values) == self.shape), ValueError("values must have shape {}".format(self.shape))

        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        cP.pretty(ax)

        xscale = kwargs.pop('xscale', 'linear')
        if self.x.log is not None:
            xscale = 'log'
        yscale = kwargs.pop('yscale', 'linear')
        if self.y_log is not None:
            yscale = 'log'
        flipX = kwargs.pop('flipX', False)
        flipY = kwargs.pop('flipY', False)

        equalize = kwargs.pop('equalize', False)
        clim_scaling = kwargs.pop('clim_scaling', None)
        colorbar = kwargs.pop('colorbar', True)
        cl = kwargs.pop('clabel', None)
        cax = kwargs.pop('cax', None)
        cmap = kwargs.pop('cmap', 'viridis')
        cmapIntervals = kwargs.pop('cmapIntervals', None)
        cmap = copy(mplcm.get_cmap(cmap, cmapIntervals))
        cmap.set_bad(color='white')
        orientation = kwargs.pop('orientation', 'vertical')
        log = kwargs.pop('log', False)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)

        grid = kwargs.pop('grid', False)
        if 'edgecolor' in kwargs:
            grid = True
        if grid:
            kwargs['edgecolor'] = kwargs.pop('edgecolor', 'k')
            kwargs['linewidth'] = kwargs.pop('linewidth', 2)

        if (log):
            values, logLabel = utilities._log(values, log)

        if vmin is not None:
            values[values < vmin] = vmin
        if vmax is not None:
            values[values > vmax] = vmax

        if equalize:
            nBins = kwargs.pop('nbins', 256)
            assert nBins > 0, ValueError('nBins must be greater than zero')
            values, dummy = utilities.histogramEqualize(values, nBins=nBins)

        rescale = lambda y: (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))
        v = rescale(values)

        y_edges = self.y_edges

        y_edges = utilities._power(y_edges, self.y_log)

        if np.any(y_edges == np.inf):
            max_edge = 2.0 * np.max(y_edges[np.isfinite(y_edges)])
        elif np.any(y_edges == -np.inf):
            max_edge = 2.0 * np.min(y_edges[np.isfinite(y_edges)])
        else:
            max_edge = np.max(y_edges[np.isfinite(y_edges)])

        i = 0

        bottom = y_edges[:, i] #+ self.relativeTo
        while np.any(i < self.nCells):
            active = np.where(i < self.nCells)

            top = y_edges[:, i+1] #+ self.relativeTo

            width = np.zeros(self.x.nCells)
            width[active] = top[active] - bottom[active]
            pm = plt.bar(self.x.centres, width, self.x.widths, bottom=bottom, color=cmap(v[:, i]), **kwargs)
            i += 1
            bottom = top

        plt.xscale(xscale)
        plt.yscale(yscale)

        cP.xlabel(self.x.label)
        cP.ylabel(self.y_edges.label)

        if flipX:
            ax.invert_xaxis()

        if flipY:
            ax.invert_yaxis()

        cbar = None
        if (colorbar):

            sm = mplcm.ScalarMappable(cmap=cmap, norm=plt.Normalize(np.nanmin(values), np.nanmax(values)))
            sm.set_array([])

            if (equalize):
                cbar = plt.colorbar(sm, extend='both', cax=cax, orientation=orientation)
            else:
                cbar = plt.colorbar(sm, cax=cax, orientation=orientation)

            if cl is None:
                if (log):
                    cP.clabel(cbar, logLabel + utilities.getNameUnits(values))
                else:
                    cP.clabel(cbar, utilities.getNameUnits(values))
            else:
                cP.clabel(cbar, cl)

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
            splt = gs.subgridspec(1, 2)

        else:
            shape = (4, 2)
            splt = gs.subgridspec(*shape)

        ax = [plt.subplot(splt[0, 0], sharex=sharex, sharey=sharey)] # n_cells
        sharex = ax[0] if sharex is None else sharex
        ax.append(plt.subplot(splt[0, 1], sharex=sharex, sharey=sharey)) # y_edges
        sharey = ax[1] if sharey is None else sharey

        if values is not None:
            ax += [plt.subplot(splt[np.unravel_index(i, shape)], sharex=sharex, sharey=sharey) for i in range(2, 8)]

        for a in ax:
            cP.pretty(a)

        return ax

    def plot_posteriors(self, axes=None, values=None, value_kwargs={}, sharex=None, sharey=None, **kwargs):
        # assert len(axes) == 2, ValueError("Must have length 2 list of axes for the posteriors. self.init_posterior_plots can generate them")

        # best = kwargs.get('best', None)
        # if best is not None:
        #     ncells_kwargs['line'] = best.nCells
        #     edges_kwargs['line'] = best.edges[1:]

        if axes is None:
            axes = kwargs.pop('fig', plt.gcf())

        if not isinstance(axes, list):
            axes = self._init_posterior_plots(axes, values=values, sharex=sharex, sharey=sharey)

        ncells_kwargs = kwargs.get('ncells_kwargs', {})
        y_edges_kwargs = kwargs.get('y_edges_kwargs', {})

        self.nCells.plot_posteriors(ax = axes[0], **ncells_kwargs)

        if kwargs.pop('flipY', False) :
            y_edges_kwargs['flipY'] = True
        self.y_edges.plot_posteriors(ax = axes[1], **y_edges_kwargs)

        if values is not None:
            axis = value_kwargs.pop('axis', 1)
            mean = values.posterior.mean(axis=axis)
            mean.pcolor(ax=axes[2], **value_kwargs)
            tmp = values.posterior.percentile(percent=5.0, axis=axis)
            tmp.pcolor(ax=axes[4], **value_kwargs)
            tmp = values.posterior.percentile(percent=95.0, axis=axis)
            tmp.pcolor(ax=axes[6], **value_kwargs)
            tmp = values.posterior.entropy(axis=axis)
            tmp.pcolor(ax=axes[3])
            tmp = values.posterior.opacity(axis=axis)
            a, b, cb = tmp.pcolor(axis=axis, ax=axes[5], ticks=[0.0, 0.5, 1.0], cmap='plasma')

            if cb is not None:
                labels = ['Less', '', 'More']
                cb.ax.set_yticklabels(labels)
                cb.set_label("Confidence")

            mean.pcolor(ax=axes[7], alpha = tmp.values, **value_kwargs)

        return axes


    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None, upcast=False):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = self.create_hdf_group(parent, name)
        self.x.createHdf(grp, 'x', withPosterior=withPosterior, fillvalue=fillvalue, upcast=upcast)
        self.y_edges.createHdf(grp, 'y/edges', withPosterior=withPosterior, fillvalue=fillvalue, upcast=upcast)

        if not self.relativeTo is None:
            self.relativeTo.createHdf(grp, 'y/relativeTo', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=upcast)

        if self._nCells is not None:
            self.nCells.createHdf(grp, 'nCells', withPosterior=withPosterior, fillvalue=fillvalue, upcast=upcast)

        return grp

    @classmethod
    def fromHdf(cls, grp, index=None, skip_posterior=False):
        """ Reads in the object from a HDF file """
        from .RectilinearMesh1D import RectilinearMesh1D
        if index is None:

            x = RectilinearMesh1D.fromHdf(grp['x'], skip_posterior=skip_posterior)
            nCells = StatArray.StatArray.fromHdf(grp['nCells'], skip_posterior=skip_posterior)
            edges = StatArray.StatArray.fromHdf(grp['y/edges'], skip_posterior=skip_posterior)

            relativeTo = None
            if 'y/relativeTo' in grp:
                relativeTo = StatArray.StatArray.fromHdf(grp['y/relativeTo'], skip_posterior=skip_posterior)
                if np.all(np.isnan(relativeTo)):
                    relativeTo = None

            out = cls(max_cells=edges.shape[1]-1,  x=x, relativeTo=relativeTo, nCells=nCells)
            out.y_edges = edges

        else:
            return RectilinearMesh1D.fromHdf(grp, index=index, skip_posterior=skip_posterior)

        return out
