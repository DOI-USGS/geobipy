""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
from copy import copy, deepcopy

from numpy import argwhere
from numpy import inf, int32, integer, isfinite
from numpy import isinf, isnan, max, min, nan, nanmax, nanmin, ndim, outer, ravel_multi_index
from numpy import shape, size, unravel_index
from numpy import where, zeros
from numpy import all as npall
from numpy import any as npany

from matplotlib.figure import Figure
from ..core.DataArray import DataArray
from ..statistics.StatArray import StatArray
from .RectilinearMesh2D import RectilinearMesh2D

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

    def __init__(self, max_cells, x=None, relative_to=None, nCells=None, **kwargs):
        """ Initialize a 2D Rectilinear Mesh"""

        self._max_cells = int32(max_cells)
        self.x = kwargs if x is None else x
        self.nCells = nCells
        self.y_edges = None
        self.y_log = kwargs.get('y_log')
        self.relative_to = relative_to

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
            assert size(values) == self.x.nCells, ValueError("Size of nCells must be {}".format(self.x.nCells))

        self._nCells = StatArray(values, 'Number of cells', dtype=int32)

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

            assert ndim(values) == 2, ValueError("y_edges must have 2 dimensions")
            assert shape(values)[0] == self.x.nCells, ValueError("First dimension of y_edges must have size {}".format(self.x.nCells))

            values, dum = utilities._log(values, log=self.y_log)

            for i in range(self.x.nCells):
                values[i, self.nCells[i]+1:] = nan

        self._y_edges = StatArray(values)

    @property
    def plotting_edges(self):
        out = copy(self.y_edges)
        if npany(self.y_edges[:, -1] == inf):
            i = argwhere(self.y_edges[:, -1] == inf)
            out[i, -1] = 1.1 * max(out[i, -2])
        return out

    def __getitem__(self, slic):
        """Allow slicing of the histogram."""
        from .RectilinearMesh1D import RectilinearMesh1D
        # assert shape(slic) == (1,), ValueError("slic must be over 1 dimensions.")

        if isinstance(slic, (int, integer)):
            relative_to = self.x._relative_to[slic] if not self.x._relative_to is None else None
            return RectilinearMesh1D(edges=self.y_edges[slic, :self.nCells[slic]+2], relative_to=relative_to)

        else:
            slic0 = slic
            if isinstance(slic.stop, (int, integer)):
                # If a slice, add one to the end for bins.
                slic0 = slice(slic.start, slic.stop+1, slic.step)

            relative_to = self.relative_to[slic] if not self.relative_to is None else None
            if self.xyz:
                out = type(self)(x_edges=self._x.edges[slic0], y_edges=self._y.edges[slic0], z_edges=self._z_edges[slic0, :], relative_to=relative_to)
            else:
                out = type(self)(x_edges=self._x.edges[slic0], y_edges=self._z_edges[slic0, :], relative_to=relative_to)
            out.nCells = self.nCells[slic]
            return out

    def n_posteriors(self):
        return sum([self.nCells.hasPosterior, self.y.edges.hasPosterior])

    def pcolor(self, values, **kwargs):

        assert npall(shape(values) == self.shape), ValueError("values must have shape {}".format(self.shape))

        geobipy_kwargs, kwargs = cP.filter_plotting_kwargs(kwargs)
        color_kwargs, kwargs = cP.filter_color_kwargs(kwargs)

        ax = geobipy_kwargs['ax']

        cP.pretty(ax)

        xscale = kwargs.pop('xscale', 'linear' if self.x.log is None else 'log')
        yscale = kwargs.pop('yscale', 'linear' if self.y_log is None else 'log')

        # color_kwargs['cmap'].set_bad(color='white')

        if 'edgecolor' in kwargs:
            geobipy_kwargs['grid'] = True

        if geobipy_kwargs['grid']:
            kwargs['edgecolor'] = kwargs.pop('edgecolor', 'k')
            kwargs['linewidth'] = kwargs.pop('linewidth', 2)

        if (geobipy_kwargs['log']):
            values, logLabel = utilities._log(values, geobipy_kwargs['log'])

        vmin = kwargs.pop('vmin', None)
        if vmin is not None:
            values[values < vmin] = vmin

        vmax = kwargs.pop('vmax', None)
        if vmax is not None:
            values[values > vmax] = vmax

        if color_kwargs['equalize']:
            assert color_kwargs['nBins'] > 0, ValueError('nBins must be greater than zero')
            values, dummy = utilities.histogramEqualize(values, nBins=color_kwargs['nBins'])

        if color_kwargs['clim_scaling'] is not None:
            values = utilities.trim_by_percentile(values, color_kwargs['clim_scaling'])

        if geobipy_kwargs['hillshade'] is not None:
            kw = geobipy_kwargs['hillshade'] if isinstance(geobipy_kwargs['hillshade'], dict) else {}
            values = cP.hillshade(values, azimuth=kw.get('azimuth', 30), altitude=kw.get('altitude', 30))

        rescale = lambda y: (y - nanmin(y)) / (nanmax(y) - nanmin(y))
        v = rescale(values)

        y_edges = self.y_edges
        y_edges = utilities._power(y_edges, self.y_log)

        min_edge = min(y_edges[isfinite(y_edges)])
        max_edge = max(y_edges[isfinite(y_edges)])
        if npany(isinf(y_edges)):
            if nanmin(y_edges) == -inf:
                min_edge = 2.0 * min(y_edges[isfinite(y_edges)])
            elif nanmax(y_edges) == inf:
                max_edge = 2.0 * max(y_edges[isfinite(y_edges)])

        alpha = ndim(color_kwargs['alpha'] > 0) if color_kwargs['alpha'] is not None else False

        width = zeros(self.x.nCells)

        relative_to = 0.0 if self.relative_to is None else self.relative_to
        for i in range(max(self.nCells)):

            active = where(self.nCells > i)[0]

            bottom = y_edges[:, i] + relative_to
            top = y_edges[:, i+1] + relative_to

            top[top == -inf] = min_edge
            top[top == inf] = max_edge

            bottom[bottom == -inf] = min_edge
            bottom[bottom == inf] = max_edge

            width[:] = 0.0
            width[active] = top[active] - bottom[active]

            colour = color_kwargs['cmap'](v[:, i])

            if alpha:
                colour[:, -1] = color_kwargs['alpha'][:, i]

            pm = ax.bar(self.x.centres, width, self.x.widths, bottom=bottom, color=colour)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        cP.xlabel(ax, self.x.label, wrap=geobipy_kwargs['wrap_xlabel'])
        cP.ylabel(ax, self.y_edges.label, wrap=geobipy_kwargs['wrap_ylabel'])

        if geobipy_kwargs['flipX']:
            ax.invert_xaxis()

        if geobipy_kwargs['flipY']:
            ax.invert_yaxis()

        if geobipy_kwargs['xlim'] is not None:
            ax.set_xlim(geobipy_kwargs['xlim'])

        if geobipy_kwargs['ylim'] is not None:
            ax.set_ylim(geobipy_kwargs['ylim'])

        cbar = None
        if (color_kwargs['colorbar']):

            sm = mplcm.ScalarMappable(cmap=color_kwargs['cmap'], norm=plt.Normalize(nanmin(values), nanmax(values)))
            sm.set_array([])

            if (color_kwargs['equalize']):
                cbar = plt.colorbar(sm, extend='both', ax=ax, cax=color_kwargs['cax'], orientation=color_kwargs['orientation'])
            else:
                cbar = plt.colorbar(sm, ax=ax, cax=color_kwargs['cax'], orientation=color_kwargs['orientation'])

            if color_kwargs['clabel'] != False:
                color_kwargs['clabel'] = utilities.getNameUnits(values)
                if (geobipy_kwargs['log']):
                    color_kwargs['clabel'] = logLabel + color_kwargs['clabel']

            cP.clabel(cbar, color_kwargs['clabel'], wrap=color_kwargs['wrap_clabel'])

        return ax, pm, cbar

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

        ax = {'ncells': cP.pretty(plt.subplot(splt[0, 0], sharex=sharex, sharey=sharey))}

        sharex = ax['ncells'] if sharex is None else sharex
        ax['edges'] = cP.pretty(plt.subplot(splt[0, 1], sharex=sharex, sharey=sharey))
        sharey = ax['edges'] if sharey is None else sharey

        if values is not None:
            ax['values'] = [cP.pretty(plt.subplot(splt[unravel_index(i, shape)], sharex=sharex, sharey=sharey)) for i in range(2, 8)]

        # for a in ax:
        #     cP.pretty(a)

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

        self.nCells.plot_posteriors(ax = axes['ncells'], **ncells_kwargs)

        if kwargs.pop('flipY', False) :
            y_edges_kwargs['flipY'] = True
        self.y_edges.plot_posteriors(ax = axes['edges'], **y_edges_kwargs)

        if values is not None:
            axis = value_kwargs.pop('axis', 1)
            mean = values.posterior.mean(axis=axis)
            mean.pcolor(ax=axes['values'][0], **value_kwargs)
            tmp = values.posterior.percentile(percent=5.0, axis=axis)
            tmp.pcolor(ax=axes['values'][2], **value_kwargs)
            tmp = values.posterior.percentile(percent=95.0, axis=axis)
            tmp.pcolor(ax=axes['values'][4], **value_kwargs)
            tmp = values.posterior.entropy(axis=axis)
            tmp.pcolor(ax=axes['values'][1])
            tmp = values.posterior.opacity(axis=axis)
            a, b, cb = tmp.pcolor(axis=axis, ax=axes['values'][3], ticks=[0.0, 0.5, 1.0], cmap='plasma')

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

        if not self.relative_to is None:
            self.relative_to.createHdf(grp, 'y/relative_to', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=upcast)

        if self._nCells is not None:
            self.nCells.createHdf(grp, 'nCells', withPosterior=withPosterior, fillvalue=fillvalue, upcast=upcast)

        return grp

    @classmethod
    def fromHdf(cls, grp, index=None, skip_posterior=False):
        """ Reads in the object from a HDF file """
        from .RectilinearMesh1D import RectilinearMesh1D
        if index is None:
            x = RectilinearMesh1D.fromHdf(grp['x'], skip_posterior=skip_posterior)
            nCells = StatArray.fromHdf(grp['nCells'], skip_posterior=skip_posterior)
            edges = StatArray.fromHdf(grp['y/edges'], skip_posterior=skip_posterior)

            relative_to = None
            if 'y/relative_to' in grp:
                relative_to = DataArray.fromHdf(grp['y/relative_to'], skip_posterior=skip_posterior)
                if npall(isnan(relative_to)):
                    relative_to = None

            out = cls(max_cells=edges.shape[1]-1,  x=x, relative_to=relative_to, nCells=nCells)
            out.y_edges = edges

        else:
            return RectilinearMesh1D.fromHdf(grp, index=index, skip_posterior=skip_posterior)

        return out
